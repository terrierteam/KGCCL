import world
import dataloader
import model
import utils
from pprint import pprint
from os.path import join
import os

if world.dataset in ['movielens', 'last-fm', 'MIND', 'yelp2018', 'amazon-book', 'LAS', 'ALB', 'af', 'lf', 'mf']:
    dataset = dataloader.Loader(path=join(world.DATA_PATH,world.dataset))
    kg_dataset = dataloader.KGDataset()
print("KGC: {} @ d_prob:{} @ joint:{} @ from_pretrain:{}".format(world.kgc_enable, world.kg_p_drop, world.kgc_joint, world.use_kgc_pretrain))
print("UIC: {} @ d_prob:{} @ temp:{} @ reg:{}".format(world.uicontrast, world.ui_p_drop, world.kgc_temp, world.ssl_reg))
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
MODELS = {
    'kgccl': model.KGCCL,
}
