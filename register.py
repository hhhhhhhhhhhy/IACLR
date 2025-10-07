import world
import dataloader
from models.p1_resid import LightGCN_Residual
from models.p2_intent import LightGCN_Intent 

if world.dataset in ['yelp2018', 'amazon', 'steam']:
    dataset = dataloader.Loader(path=f"{world.DATA_PATH}/{world.dataset}")
else:
    raise NotImplementedError

from models.model import LightGCN
RECMODELS = {
    'lgn_intent': LightGCN_Intent      
}

print('===========config================')
print("dataset:", world.dataset)
print("device:", world.device)
print("model_name:", world.model_name)
print("topks:", world.topks)
print('===========end===================')