import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='amazon',
                        help="available datasets: [yelp2018, amazon, steam]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[5,10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn_intent', help='rec-model, support [mf, lgn, lgn_resid, lgn_intent]')
    parser.add_argument('--csv_dir', type=str,default="./log/",
                        help="csv_dir")
    
    parser.add_argument('--use_semantic_item_init', action='store_true', help="whether to use semantic features to init item embeddings")
    parser.add_argument('--intent_num', type=int, default=64,
                        help="number of intent nodes")
    parser.add_argument('--topk_intent', type=int, default=10,
                        help="top-k intent connections per user/item")
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help="number of warmup epochs before introducing intent nodes")
    parser.add_argument('--intent_lr', type=float, default=1e-5,
                        help="learning rate for intent-related parameters")
    parser.add_argument('--intent_weight_step', type=float, default=0.02,
                        help="step size for gradually increasing intent weight")
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help="lambda_align")
    
    parser.add_argument('--multi_view', action='store_true', help="enable multi-view intent")
    parser.add_argument('--behavior_view', action='store_true', help="enable behavior_view")
    parser.add_argument('--semantic_view', action='store_true', help="enable semantic_view")
    parser.add_argument('--fusion', type=str, default='gate',
                        choices=['gate', 'avg'],
                        help='fusion strategy for multi-view intents')
    parser.add_argument('--semantic_intent_init', type=str, default=None,
                    help='path to semantic init npy file for semantic view prototypes')
    
    parser.add_argument('--iva_kind', type=str, default='kl', choices=['kl', 'infonce'],
                    help='the kind of iva loss')
    parser.add_argument('--lambda_iva', type=float, default=0.0,
                    help='lambda_iva, if 0 then no iva')
    parser.add_argument('--iva_dim', type=int, default=64,
                    help='iva_dim')
    parser.add_argument('--iva_delay', type=int, default=50,
                    help='iva_delay')
    

    parser.add_argument('--lambda_nca', type=float, default=0.0,
                    help='lambda_nca, if 0 then no nca')
    parser.add_argument('--nca_weight_ui', type=float, default=1.0,
                    help='nca_weight_ui')
    parser.add_argument('--nca_weight_ii', type=float, default=1.0,
                    help='nca_weight_ii')
    parser.add_argument('--nca_temperature', type=float, default=0.07,
                    help='nca_temperature')
    parser.add_argument('--nca_delay', type=int, default=50,
                    help='nca_delay')
    
    return parser.parse_args()
