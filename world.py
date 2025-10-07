import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(__file__)
CODE_PATH = ROOT_PATH
DATA_PATH = join(CODE_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['yelp2018', 'amazon','steam']
all_models  = ['lgn_intent']

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

config['use_semantic_item_init'] = args.use_semantic_item_init
config['llm_emb_path'] = f"/hy-tmp/data/{dataset}/embeddings/item_emb.pt"

config['intent_num'] = args.intent_num
config['topk_intent'] = args.topk_intent
config['warmup_epochs'] = args.warmup_epochs
config['intent_lr'] = args.intent_lr
config['intent_weight_step'] = args.intent_weight_step

config['intent_schedule'] = 'fixed'
config['intent_weight_fixed'] = 0.05
config['intent_weight_max'] = 0.2
config['intent_sigmoid_scale'] = 20.0

config['lambda_align'] = args.lambda_align

config['intent_lr'] = 0.0005
config['normalize_node_emb'] = False

config['multi_view'] = args.multi_view
config['behavior_view'] = args.behavior_view
config['semantic_view'] = args.semantic_view
config['fusion'] = args.fusion
config['semantic_intent_init'] = args.semantic_intent_init

config['iva_kind'] = args.iva_kind
config['lambda_iva'] = args.lambda_iva
config['iva_dim'] = args.iva_dim
config['iva_delay'] = args.iva_delay

config['lambda_nca'] = args.lambda_nca
config['nca_weight_ui'] = args.nca_weight_ui
config['nca_weight_ii'] = args.nca_weight_ii
config['nca_temperature'] = args.nca_temperature
config['nca_delay'] = args.nca_delay

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
config['device'] = device
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

config['csv_dir'] = args.csv_dir





TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)




logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
