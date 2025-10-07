import world
import utils.utils as utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import register
from register import dataset
import csv
import os


# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

Recmodel = register.RECMODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
print("Model class:", type(Recmodel))

bpr = utils.BPRLoss(Recmodel, world.config)


weight_file = utils.getFileName()
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    print("not enable tensorflowboard")

# best_recall = 0
best_results = {
    'recall@5': 0, 'recall@10': 0, 'recall@20': 0,
    'ndcg@5': 0,  'ndcg@10': 0,  'ndcg@20': 0
}
best_epoch = 0
last_update_epoch = {
    'recall@5': 0, 'recall@10': 0, 'recall@20': 0,
    'ndcg@5': 0, 'ndcg@10': 0, 'ndcg@20': 0
}


save_dir = world.config['csv_dir']
os.makedirs(save_dir, exist_ok=True)

def _save_intent_snapshot(recmodel, prefix, out_dir):
    """
    Save per-view raw intent embeddings and projected/fused versions.
    prefix: str, e.g., 'init' or 'final'
    """
    if not hasattr(recmodel, "intent_embeddings"):
        return
    # per-view raw intents
    for v, view_name in enumerate(getattr(recmodel, "view_order", [f"view{v}" for v in range(len(recmodel.intent_embeddings))])):
        emb = recmodel.intent_embeddings[v].detach().cpu().numpy()
        np.save(os.path.join(out_dir, f"intent_emb_{view_name}_{prefix}.npy"), emb)
    try:
        proj_list = []
        for v in range(recmodel.num_views):
            with torch.no_grad():
                proj = recmodel.intent_projections[v](recmodel.intent_embeddings[v]).detach().cpu().numpy()
            np.save(os.path.join(out_dir, f"intent_emb_{recmodel.view_order[v]}_{prefix}_proj.npy"), proj)
            proj_list.append(proj)
        fused = np.mean(np.stack(proj_list, axis=0), axis=0)  # [K, dim]
        np.save(os.path.join(out_dir, f"intent_emb_fused_{prefix}.npy"), fused)
    except Exception as e:
        print(f"[SaveIntent] projection/fused save failed: {e}")

def _save_topk_assignments(recmodel, out_dir, topk=None):
    topk = topk or world.config.get('topk_intent', 8)
    with torch.no_grad():
        users_emb, items_emb = recmodel.computer()  # [n_users, dim], [n_items, dim]
        users_emb = users_emb.detach().cpu().numpy()
        items_emb = items_emb.detach().cpu().numpy()

        try:
            fused_pool = recmodel._fused_intent_pool().detach().cpu().numpy()  # [K, dim]
        except Exception:
            fused_list = []
            for v in range(getattr(recmodel, "num_views", 1)):
                try:
                    p = recmodel.intent_projections[v](recmodel.intent_embeddings[v]).detach().cpu().numpy()
                    fused_list.append(p)
                except Exception:
                    pass
            if len(fused_list) > 0:
                fused_pool = np.mean(np.stack(fused_list, axis=0), axis=0)
            else:
                print("[TopK] fused intent pool unavailable, aborting topk save.")
                return

        def _norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n==0] = 1.0
            return x / n

        users_n = _norm(users_emb)
        items_n = _norm(items_emb)
        intents_n = _norm(fused_pool)

        def _topk_for_matrix(A, B, topk):
            sim = A.dot(B.T)  # [N, K]
            idx = np.argpartition(-sim, topk-1, axis=1)[:, :topk]
            rows = np.arange(sim.shape[0])[:, None]
            top_sims = sim[rows, idx]
            order = np.argsort(-top_sims, axis=1)
            idx_sorted = idx[rows, order]
            weights = top_sims[rows, order]
            # convert weights to softmax over selected (numeric-safe)
            expw = np.exp(weights - np.max(weights, axis=1, keepdims=True))
            wnorm = expw / (np.sum(expw, axis=1, keepdims=True) + 1e-12)
            return idx_sorted.squeeze(), wnorm.squeeze()

        print("[TopK] computing user->intent topk ...")
        user_topk_idx, user_topk_w = _topk_for_matrix(users_n, intents_n, topk)
        print("[TopK] computing item->intent topk ...")
        item_topk_idx, item_topk_w = _topk_for_matrix(items_n, intents_n, topk)


        return {
            "user_topk_idx": user_topk_idx, "user_topk_w": user_topk_w,
            "item_topk_idx": item_topk_idx, "item_topk_w": item_topk_w
        }

def _save_intent_top_neighbors(recmodel, out_dir, topn=10):
    try:
        with torch.no_grad():
            users_emb, items_emb = recmodel.computer()
            users_np = users_emb.detach().cpu().numpy()
            items_np = items_emb.detach().cpu().numpy()
            fused_pool = recmodel._fused_intent_pool().detach().cpu().numpy()  # [K, dim]
        def _n(x): 
            n = np.linalg.norm(x, axis=1, keepdims=True); n[n==0]=1; return x / n
        un = _n(users_np); in_ = _n(items_np); zn = _n(fused_pool)
        sim_i_items = zn.dot(in_.T)  # [K, M]
        sim_i_users = zn.dot(un.T)   # [K, N]
        top_items = np.argpartition(-sim_i_items, topn-1, axis=1)[:, :topn]
        top_users = np.argpartition(-sim_i_users, topn-1, axis=1)[:, :topn]
    except Exception as e:
        print(f"[IntentNeighbors] failed: {e}")

try:
    for epoch in range(world.TRAIN_epochs):
        Recmodel.current_epoch = epoch
        start = time.time()
        if world.model_name != 'lgn':
            if epoch == Recmodel.warmup_epochs:
                torch.save(Recmodel.state_dict(), f'warmup_ckpt_{Recmodel.warmup_epochs}.pth')  # 先存
            
        if hasattr(Recmodel, 'update_intent_weight'):
            Recmodel.update_intent_weight()
        if epoch % 10 == 0:
            print("[TEST]")
            Recmodel.current_epoch = epoch
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            cur_recall5, cur_recall10, cur_recall20 = results['recall']
            cur_ndcg5,  cur_ndcg10,  cur_ndcg20  = results['ndcg']
            cur_metrics = {
                'recall@5': cur_recall5, 'recall@10': cur_recall10, 'recall@20': cur_recall20,
                'ndcg@5': cur_ndcg5, 'ndcg@10': cur_ndcg10, 'ndcg@20': cur_ndcg20
            }
            improved = False
            for key, val in cur_metrics.items():
                if val > best_results[key]:
                    best_results[key] = val
                    last_update_epoch[key] = epoch
                    improved = True

            if improved:
                torch.save(Recmodel.state_dict(), weight_file)
                print(f"【Best Model Saved】Epoch {epoch}: "
                    f"Recall@5={best_results['recall@5']:.6f}, "
                    f"Recall@10={best_results['recall@10']:.6f}, "
                    f"Recall@20={best_results['recall@20']:.6f}, "
                    f"NDCG@5={best_results['ndcg@5']:.6f}, "
                    f"NDCG@10={best_results['ndcg@10']:.6f}, "
                    f"NDCG@20={best_results['ndcg@20']:.6f}")
        
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
  
        if world.model_name == 'lgn_intent':
            if epoch == Recmodel.warmup_epochs and not Recmodel.intent_initialized:
                Recmodel.init_intent_nodes()
                print("Warmup completed, intent nodes initialized")
                
                print("[TEST AFTER INTENT INIT]")
                results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                print(f"Results after intent init: {results}")
                
                print("Optimizer parameters:")
                total_params, intent_params = 0, 0
                for name, param in Recmodel.named_parameters():
                    total_params += param.numel()
                    if 'intent' in name or 'attn_param' in name:
                        intent_params += param.numel()
                        print(f"  {name}: {param.shape} (requires_grad: {param.requires_grad})")
                print(f"Total parameters: {total_params}, Intent-related parameters: {intent_params}")

                    
        patience = 20
        if all(epoch - last_update_epoch[key] > patience for key in best_results.keys()):
            print(f"Early stopping at epoch {epoch}...")
            torch.save(Recmodel.state_dict(), f'{save_dir}/ckpt_final_stopat{epoch}.pth')
            break
        
        epoch_time = time.time() - start
        if 'epoch_time_list' not in globals():
            epoch_time_list = []
        epoch_time_list.append(epoch_time)
        
        if epoch % world.config.get('ckpt_interval', 200) == 0:
            torch.save(Recmodel.state_dict(), f'{save_dir}/ckpt_epoch{epoch}.pth')
        
        
finally:
    if world.tensorboard:
        w.close()
    print("【Training Finished】Best Results:")
    for k, v in best_results.items():
        print(f"  {k}: {v:.6f} (last updated at epoch {last_update_epoch[k]})")
