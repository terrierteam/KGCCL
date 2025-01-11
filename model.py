import os
import world
import torch
from dataloader import BasicDataset
import dataloader
from dataloader import load_data
from torch import nn
from GAT import GAT
import numpy as np
from utils import _L2_loss_mean
import torch.nn.functional as F
import time
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
import os
import world
import torch
from dataloader import BasicDataset
import dataloader
from dataloader import load_data
from torch import nn
from GAT import GAT
import numpy as np
from utils import _L2_loss_mean
import torch.nn.functional as F
import time
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class Aggregator(nn.Module):
    def __init__(self, n_usersX):
        super(Aggregator, self).__init__()
        self.n_usersX = n_usersX

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat,
                weight):
        from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

        n_entities = entity_emb.shape[0]

        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  

        neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail], weight[edge_type - 1])
        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
                                                                     neigh_relation_emb.shape[1])

        neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
        neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        user_agg = torch.sparse.mm(interact_mat, entity_emb)

        score = torch.mm(user_emb, weight.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):

        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights
    
class GraphConv(nn.Module):

    def __init__(self, channel, n_hops, n_usersX,
                  n_relationsX, interact_mat,
                 ind, node_dropout_rate=0.0, mess_dropout_rate=0.0):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relationsX = n_relationsX
        self.n_usersX = n_usersX

        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 10
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relationsX - 1, channel))
        self.weight = nn.Parameter(weight) 
        self.device = torch.device("cuda:" + str(0))

        for i in range(n_hops):
           
            self.convs.append(Aggregator(n_usersX=n_usersX))
            

        self.dropout = nn.Dropout(p=mess_dropout_rate)  

    def _edge_sampling(self, edge_index, edge_type, rate=0.0):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]
    

    def _sparse_dropout(self, x, rate=0.0):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)    

        entity_res_emb = entity_emb  
        user_res_emb = user_emb 
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight)

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb
    

class KGCCL(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset,
                 kg_dataset, data_config, args_config, graphX, relation_dict, adj_mat):
        super(KGCCL, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
        self.eps = 0.1
        self.layer_cl = 2
        self.n_usersX = data_config['n_usersX']
        self.n_itemsX = data_config['n_itemsX']
        self.n_relationsX = data_config['n_relationsX']
        self.n_entitiesX = data_config['n_entitiesX']  
        self.n_nodesX = data_config['n_nodesX'] 
        self.emb_size = args_config.dim
        self.adj_mat = adj_mat
        self.graphX = graphX
        self.relation_dict = relation_dict
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodesX, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat)
        self.all_embed = nn.Parameter(self.all_embed)
        self.itemuie_emb = None
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate        
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.edge_index, self.edge_type = self._get_edges(graphX)        
        self.__init_weight()
        self._init_weightX()
        self.gcnX = self._init_model()
        self.itemuie_emb = None
        self.lightgcnX_layer = 2
        self.n_item_layer = 1

    def _init_weightX(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = torch.nn.Parameter(initializer(torch.empty(self.n_nodesX, self.emb_size)))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(world.device)
    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_usersX=self.n_usersX,
                         n_relationsX=self.n_relationsX,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  

    def _get_edges(self, graphX):
        graphX_tensor = torch.tensor(list(graphX.edges)) 
        index = graphX_tensor[:, :-1]  
        type = graphX_tensor[:, -1]  
        return index.t().long().to(world.device), type.long().to(world.device)

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config['pretrain'] == 0:
            world.cprint('use NORMAL distribution UI')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution ENTITY')
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    
    
    def _convert_sp_mat_to_sp_tensor(self, adj_mat):
        coo = adj_mat.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)    


    def uie_embedding(self,kg_dict):
        user_emb = self.all_embed[:self.n_usersX, :]
        item_emb = self.all_embed[self.n_usersX:, :]
        entity_gcn_emb, user_gcn_emb = self.gcnX(user_emb,
                                 item_emb,
                                 self.edge_index,
                                 self.edge_type,
                                 self.interact_mat,
                                 mess_dropout=self.mess_dropout,
                                 node_dropout=self.node_dropout)
        item_indices = list(kg_dict.keys())
        useruie_emb = user_gcn_emb
        itemuie_emb = entity_gcn_emb[item_indices]
        return useruie_emb, itemuie_emb

    def view_computer(self, g_droped, kg_droped):    
        useruie_emb, itemuie_emb = self.uie_embedding(kg_droped)
        users_emb = self.embedding_user.weight
        items_emb = itemuie_emb
        ego_embeddings = torch.cat([users_emb, items_emb])
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(g_droped, ego_embeddings)
            random_noise = torch.rand_like(ego_embeddings).cuda()
            ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        users, items = torch.split(final_embeddings, [self.num_users, self.num_items])
       
        users_cl, items_cl = torch.split(all_embeddings_cl, [self.num_users, self.num_items])     
        
        return users, items, users_cl, items_cl   
    


    def computer(self):
    
        useruie_emb, itemuie_emb = self.uie_embedding(self.kg_dict)
        users_emb = self.embedding_user.weight
        items_emb = itemuie_emb
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss
       
       
    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):

        r_embed = self.embedding_relation(r)                 
        h_embed = self.embedding_item(h)              
        pos_t_embed = self.embedding_entity(pos_t)      
        neg_t_embed = self.embedding_entity(neg_t)      

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     
       
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        
        loss = kg_loss + 1e-3 * l2_loss
        
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):

        r_embed = self.embedding_relation(r)                
        W_r = self.W_R[r]                               

        h_embed = self.embedding_item(h)              
        pos_t_embed = self.embedding_entity(pos_t)      
        neg_t_embed = self.embedding_entity(neg_t)     
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)    
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)    

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)    

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)      

        return gamma

