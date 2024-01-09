from cppimport import imp
from numpy import negative, positive
from torch_sparse.tensor import to
from model import KGCCL
from random import random, sample
from shutil import make_archive
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
from utils import randint_choice
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import world
from dataloader import BasicDataset
import dataloader
from dataloader import load_data

def drop_edge_random(edge_index, p):
    drop_mask = torch.empty((edge_index.size(1),), dtype=torch.float32, device=edge_index.device).uniform_(0, 1) < p
    x = edge_index.clone()
    x[:, drop_mask] = 0
    return x

def drop_edge_random(item2entities, p_drop, padding):
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if(random()>p_drop):
                new_es.append(e)
            else:
                new_es.append(e)
        res[item] = torch.IntTensor(new_es).to(world.device)
    return res

def pad_item_entities(kg, padding):
    max_len = max([len(es) for es in kg.values()])
    for item, es in kg.items():
        if len(es) < max_len:
            es += [padding] * (max_len - len(es))
    return kg

class Contrast(nn.Module):
  
    def __init__(self, gcn_model, tau=world.kgc_temp):
        super(Contrast, self).__init__()
        self.gcn_model : KGCL = gcn_model
        self.tau = tau

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1,z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss_overall(self, z1, z2, z_all):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        positive_pairs = between_sim
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.pair_sim(z1, z1))
        between_sim = f(self.pair_sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = False, batch_size: int = 0):
        h1 = z1
        h2 = z2

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def get_uie_views(self):
        kgX = self.gcn_model.relation_dict        
        XX = self.gcn_model.n_entitiesX
        view1 = drop_edge_random(kgX, world.kg_p_drop, XX)
        view2 = drop_edge_random(kgX, world.kg_p_drop, XX)
        return view1, view2

    def get_ui_views(self, p_drop):
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        user_np = self.gcn_model.dataset.trainUser
        item_np = self.gcn_model.dataset.trainItem
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(world.device)
        g.requires_grad = False
        return g
    
    def get_views(self, aug_side="both"):      
        uiev1, uiev2 = self.get_kg_views()            
        if aug_side=="kg" or world.uicontrast=="NO" or world.uicontrast=="ITEM-BI":
            uiv1, uiv2 = None, None
        else:
            if world.uicontrast=="NEW":
                uiv1 = self.get_ui_views(stability, 1)  
                uiv2 = self.get_ui_views(stability, 1)
        contrast_views = {
            "uiv1":uiv1,
            "uiv2":uiv2,
            "uiev1":uiev1,
            "uiev2":uiev2
        }
        return contrast_views
