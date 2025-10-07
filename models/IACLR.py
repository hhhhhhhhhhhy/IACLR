import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from models.model import BasicModel
from dataloader import BasicDataset
import world, os

class LightGCN_Intent(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN_Intent, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        
        self.intent_num = config['intent_num']
        self.top_k = config['topk_intent']
        self.warmup_epochs = config['warmup_epochs']
        self.current_epoch = 0
        
        self.intent_weight = config.get('intent_weight_fixed', 0.05)
        self.intent_weight_step = config.get('intent_weight_step', 0.0)
        self.intent_weight_max = config.get('intent_weight_max', 0.2)
        
        self.multi_view = config.get('multi_view', True)
        self.view_order = []
        if self.multi_view:
            self.view_order = ['behavioral','semantic']
        else:
            if config.get('behavior_view', False):
                self.view_order += ['behavioral']
            elif config.get('semantic_view', False):
                self.view_order += ['semantic']
            else:
                self.view_order = ['behavioral']
        self.num_views = len(self.view_order)
        self.fusion_type = config.get('fusion', 'gate')
            print(f"[MultiView] enabled. views={self.view_order}, fusion={self.fusion_type}")
        else:
            print("[MultiView] disabled, single view (behavioral)")

        self.user_intent_scales = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_views)])
        self.item_intent_scales = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_views)])

        self.intent_embeddings = nn.ParameterList()
        for v in range(self.num_views):
            p = nn.Parameter(torch.Tensor(self.intent_num, self.latent_dim))
            nn.init.normal_(p, mean=0.0, std=0.01)
            self.intent_embeddings.append(p)

        self.intent_projections = nn.ModuleList()
        for v in range(self.num_views):
            proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            try:
                nn.init.eye_(proj.weight)
            except Exception:
                nn.init.normal_(proj.weight, std=0.01)
            self.intent_projections.append(proj)

        self.intent_anchors = [None for _ in range(self.num_views)]
        
        self.intent_freeze_epochs = self.config.get('intent_freeze_epochs', 5)
        self.intent_frozen_until_epoch = [self.current_epoch + self.intent_freeze_epochs for _ in range(self.num_views)]
        print(f"Intent embedding frozen until epoch {self.intent_frozen_until_epoch[0]}")
        for v in range(self.num_views):
            self.intent_embeddings[v].requires_grad = False
        
        self.intent_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

        self.intent_initialized = False
        

        if self.num_views == 1:
            self.intent_embedding = self.intent_embeddings[0]
            self.intent_projection = self.intent_projections[0]
            self.user_intent_scale = self.user_intent_scales[0]
            self.item_intent_scale = self.item_intent_scales[0]

        self.iva_kind = self.config.get('iva_kind', 'kl')
        self.iva_delay = self.config.get('iva_delay', 50)
        hidden_dim = self.latent_dim
        proj_dim = self.config.get('contrastive_dim', self.latent_dim)
        self.proj_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.nca_delay = self.config.get('nca_delay', 50)
        

        print(f"IntentGCN is ready to go (warmup: {self.warmup_epochs} epochs)")

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            print('user NORMAL distribution initilizer')
            if self.config.get('use_semantic_item_init', False):
                llm_emb_path = self.config.get('llm_emb_path', None)
                if llm_emb_path is not None and os.path.exists(llm_emb_path):
                    llm_item_emb_cpu = torch.load(llm_emb_path, map_location='cpu')
                    llm_embedding_dim = llm_item_emb_cpu.shape[1]
                    llm_projection = nn.Linear(llm_embedding_dim, self.latent_dim, bias=False).to(world.device)

                    item_llm_emb = F.normalize(llm_item_emb_cpu, p=2, dim=1).to(world.device)
                    projected_item_emb = llm_projection(item_llm_emb)
                    self.embedding_item.weight.data.copy_(projected_item_emb)
                    print(f"[Init] item embeddings initialized from {llm_emb_path}")
        
                    user_llm_emb_cpu = torch.zeros((self.num_users, llm_embedding_dim))
                    user_item_net = self.dataset.UserItemNet.tocoo()
                    rows = torch.from_numpy(user_item_net.row).long()
                    cols = torch.from_numpy(user_item_net.col).long()
                    for u in range(self.num_users):
                        items_of_u = cols[rows == u]
                        if len(items_of_u) > 0:
                            user_llm_emb_cpu[u] = llm_item_emb_cpu[items_of_u].mean(dim=0)
                    user_llm_emb = F.normalize(user_llm_emb_cpu, p=2, dim=1).to(world.device)
                    projected_user_emb = llm_projection(user_llm_emb)
                    self.embedding_user.weight.data.copy_(projected_user_emb)
                    print(f"[Init] user embeddings initialized from LLM item avg")
                else:
                    nn.init.normal_(self.embedding_item.weight, std=0.1)
                    print("[Init] item semantic init requested but file missing, fallback to random")
            else:
                nn.init.normal_(self.embedding_item.weight, std=0.1)
                print("[Init] item embeddings randomly initialized")
            
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    # -------------------- multi-view-aware init --------------------
    def init_intent_nodes(self):
        print("Initializing intent nodes for each view...")
        for v, view_name in enumerate(self.view_order):
            if view_name == 'behavioral':
                try:
                    item_embeddings = self.embedding_item.weight.detach().cpu().numpy()
                    kmeans = KMeans(n_clusters=self.intent_num, random_state=2024, n_init=10)
                    kmeans.fit(item_embeddings)
                    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(world.device)
                    noise = torch.randn_like(centers) * 0.01
                    self.intent_embeddings[v].data = centers + noise
                    self.intent_anchors[v] = centers.detach().cpu().clone()
                    print(f"[Init] view {v} behavioral KMeans done")
                except Exception as e:
                    print(f"[Init] behavioral view KMeans failed: {e}, use random")
                    nn.init.normal_(self.intent_embeddings[v].data, mean=0.0, std=0.01)
                    self.intent_anchors[v] = None

            elif view_name == 'semantic':
                sem_path = self.config.get('semantic_intent_init', None)
                if sem_path is not None and os.path.exists(sem_path):
                    sem = np.load(sem_path)  # expect shape [intent_num, latent_dim]
                    assert sem.shape == (self.intent_num, self.latent_dim), \
                        f"semantic_init shape mismatch: got {sem.shape}, expect ({self.intent_num},{self.latent_dim})"
                    emb = torch.tensor(sem, dtype=torch.float32).to(world.device)
                    noise = torch.randn_like(emb) * 0.01
                    self.intent_embeddings[v].data = emb + noise
                    self.intent_anchors[v] = emb.detach().cpu().clone()
                    print(f"[Init] view {v} semantic loaded from {sem_path}")
                else:
                    llm_emb_path = self.config.get('llm_emb_path', None)
                    if llm_emb_path is not None and os.path.exists(llm_emb_path):
                        try:
                            llm_item = torch.load(llm_emb_path, map_location='cpu')  # tensor [m_items, d_sem]
                            d_sem = llm_item.shape[1]
                            if d_sem != self.latent_dim:
                                proj = nn.Linear(d_sem, self.latent_dim, bias=False).to(world.device)
                                with torch.no_grad():
                                    eye = torch.eye(min(d_sem, self.latent_dim), device=proj.weight.device)
                                    proj.weight.copy_(torch.zeros_like(proj.weight))
                                    proj.weight[:eye.size(0), :eye.size(1)].copy_(eye)
                                llm_proj = proj(llm_item.to(world.device))
                                arr = llm_proj.detach().cpu().numpy()
                            else:
                                arr = llm_item.numpy()
                            km = KMeans(n_clusters=self.intent_num, random_state=2024, n_init=10).fit(arr)
                            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(world.device)
                            noise = torch.randn_like(centers) * 0.01
                            self.intent_embeddings[v].data = centers + noise
                            self.intent_anchors[v] = centers.detach().cpu().clone()
                            print(f"[Init] view {v} semantic KMeans (from llm_emb) done")
                        except Exception as e:
                            print(f"[Init] semantic fallback failed: {e}, using random")
                            nn.init.normal_(self.intent_embeddings[v].data, mean=0.0, std=0.01)
                            self.intent_anchors[v] = None
                    else:
                        print(f"[Init] semantic_init not provided and llm_emb_path missing; view {v} random init")
                        nn.init.normal_(self.intent_embeddings[v].data, mean=0.0, std=0.01)
                        self.intent_anchors[v] = None
            else:
                print(f"[Init] view {v} ({view_name}) random init")
                nn.init.normal_(self.intent_embeddings[v].data, mean=0.0, std=0.01)
                self.intent_anchors[v] = None

        self.adapter_pretrain()
        self.intent_initialized = True
        print("Intent nodes initialized successfully")

    def adapter_pretrain(self):
        for param in self.embedding_user.parameters():
            param.requires_grad = False
        for param in self.embedding_item.parameters():
            param.requires_grad = False

        num_batches = 100
        batch_size = 1024
        for v in range(self.num_views):
            optimizer = torch.optim.Adam(self.intent_projections[v].parameters(), lr=0.001)
            for i in range(num_batches):
                users = torch.randint(0, self.num_users, (batch_size,), device=world.device)
                items = torch.randint(0, self.num_items, (batch_size,), device=world.device)
                with torch.no_grad():
                    users_emb = self.embedding_user(users)
                    items_emb = self.embedding_item(items)
                    original_scores = torch.sum(users_emb * items_emb, dim=1)

                intent_emb = self.intent_projections[v](self.intent_embeddings[v])
                user_intent_scores = torch.matmul(users_emb, intent_emb.t())
                item_intent_scores = torch.matmul(items_emb, intent_emb.t())
                user_attention = F.softmax(user_intent_scores, dim=1)
                item_attention = F.softmax(item_intent_scores, dim=1)
                user_intent_emb = torch.matmul(user_attention, intent_emb)
                item_intent_emb = torch.matmul(item_attention, intent_emb)
                users_enhanced = users_emb + 0.1 * user_intent_emb
                items_enhanced = items_emb + 0.1 * item_intent_emb
                enhanced_scores = torch.sum(users_enhanced * items_enhanced, dim=1)
                loss = F.mse_loss(enhanced_scores, original_scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for param in self.embedding_user.parameters():
            param.requires_grad = True
        for param in self.embedding_item.parameters():
            param.requires_grad = True
        print("Projection layer pretraining for all views completed")
    
    def update_intent_weight(self):
        if not (self.intent_initialized and self.current_epoch >= self.warmup_epochs):
            return

        schedule = self.config.get('intent_schedule', 'fixed')
        if schedule == 'fixed':
            target = self.config.get('intent_weight_fixed', self.intent_weight)
            if abs(self.intent_weight - target) > 1e-9:
                self.intent_weight = target
        elif schedule == 'sigmoid':
            t = float(self.current_epoch - self.warmup_epochs)
            scale_epoch = float(self.config.get('intent_sigmoid_scale', 20.0))
            max_w = float(self.config.get('intent_weight_max', self.intent_weight_max))
            s = 1.0 / (1.0 + torch.exp(- (t/scale_epoch - 4.0)))
            new_w = (s * max_w).item()
            self.intent_weight = max(0.0, min(self.intent_weight_max, new_w))

        for v in range(self.num_views):
            if (self.current_epoch >= self.intent_frozen_until_epoch[v]) and (not self.intent_embeddings[v].requires_grad):
                self.intent_embeddings[v].requires_grad = True
                print(f"Unfroze intent_embeddings view{v} at epoch {self.current_epoch}")
    
    @staticmethod
    def gumbel_softmax_topk(logits: torch.Tensor, k: int, tau: float = 1.0, eps=1e-10):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        gumbel_logits = (logits + gumbel_noise) / tau
        topk_val, topk_idx = torch.topk(gumbel_logits, k, dim=1)
        topk_weight = F.softmax(topk_val, dim=1)
        row = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand(-1, k).flatten()
        col = topk_idx.flatten()
        val = topk_weight.flatten()
        sparse_weight = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=val,
            size=logits.shape,
            device=logits.device).coalesce()
        return sparse_weight
    
    def _compute_attention(self, node_emb, intent_emb, scale_param):
        scores = torch.matmul(node_emb, intent_emb.t())
        sqrt_d = np.sqrt(node_emb.size(1))
        scores = scores * scale_param / sqrt_d
        return self.gumbel_softmax_topk(scores, k=self.top_k, tau=1.0)

    def _sparse_colsum(self, mat: torch.sparse.FloatTensor):
        ones = torch.ones((mat.size(0), 1), device=mat.device)
        colsum = torch.sparse.mm(mat.t(), ones)  # [M,1]
        return colsum.squeeze(1)  # [M]

    def computer(self):

        if self.current_epoch < self.warmup_epochs or not self.intent_initialized:
            return self._original_lightgcn_forward()

        users_emb, items_emb = self._original_lightgcn_forward()

        if self.config.get('normalize_node_emb', False):
            users_emb = F.normalize(users_emb, dim=1)
            items_emb = F.normalize(items_emb, dim=1)

        view_user_from = []
        view_item_from = []

        for v in range(self.num_views):
            intent_emb = self.intent_projections[v](self.intent_embeddings[v])
            user_intent_attn = self._compute_attention(users_emb, intent_emb, self.user_intent_scales[v])
            item_intent_attn = self._compute_attention(items_emb, intent_emb, self.item_intent_scales[v])

            deg_u = self._sparse_colsum(user_intent_attn).clamp_min(1e-6).unsqueeze(1)  # [K,1]
            deg_i = self._sparse_colsum(item_intent_attn).clamp_min(1e-6).unsqueeze(1)  # [K,1]

            intent_from_user = torch.sparse.mm(user_intent_attn.t(), users_emb) / deg_u  # [K, dim]
            intent_from_item = torch.sparse.mm(item_intent_attn.t(), items_emb) / deg_i  # [K, dim]

            intent_emb_updated = (intent_from_user + intent_from_item) / 2  # [K, dim]

            user_from_intent = torch.sparse.mm(user_intent_attn, intent_emb_updated) * self.intent_weight * 0.5  # [N, dim]
            item_from_intent = torch.sparse.mm(item_intent_attn, intent_emb_updated) * self.intent_weight * 0.5  # [M, dim]

            view_user_from.append(user_from_intent)
            view_item_from.append(item_from_intent)

            if self.training and self.config.get('intent_update_via_ema', False):
                with torch.no_grad():
                    self.intent_embeddings[v].mul_(0.9).add_(0.1 * intent_emb_updated.to(self.intent_embeddings[v].device))
            
        if self.num_views == 1:
            users_enhanced = users_emb + view_user_from[0]
            items_enhanced = items_emb + view_item_from[0]
        else:
            if self.fusion_type == 'avg':
                users_enhanced = users_emb + sum(view_user_from) / float(self.num_views)
                items_enhanced = items_emb + sum(view_item_from) / float(self.num_views)
            elif self.fusion_type == 'gate':
                gate = torch.sigmoid(self.intent_gate)
                
                users_enhanced = (1 - gate) * users_emb + gate * (users_emb + sum(view_user_from)/float(self.num_views))
                items_enhanced = (1 - gate) * items_emb + gate * (items_emb + sum(view_item_from)/float(self.num_views))
            else:
                raise ValueError(f"Unknown fusion type {self.fusion_type}")

        return users_enhanced, items_enhanced

    def _original_lightgcn_forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        if self.config['dropout'] and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items

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

    def iva_kl_loss(self, temp_t=0.07, temp_m=0.1):
        v0, v1 = 0, 1  # view0=behavioral, view1=semantic
        z_b = F.normalize(self.proj_mlp(self.intent_embeddings[v0]), dim=1)  # [K, d]
        z_s = F.normalize(self.proj_mlp(self.intent_embeddings[v1]), dim=1)  # [K, d]

        sim = torch.matmul(z_b, z_s.t())  # [K,K]
        p = F.softmax(sim / temp_t, dim=1)
        q = F.softmax(sim / temp_m, dim=1)
        kl_b2s = F.kl_div(q.log(), p, reduction='batchmean')

        p_s = F.softmax(sim.t() / temp_t, dim=1)
        q_s = F.softmax(sim.t() / temp_m, dim=1)
        kl_s2b = F.kl_div(q_s.log(), p_s, reduction='batchmean')

        return kl_b2s + kl_s2b


    def _fused_intent_pool(self):
        device = self.intent_embeddings[0].device
        accum = None
        for v in range(self.num_views):
            proj = self.intent_projections[v](self.intent_embeddings[v])  # [K, dim]
            if accum is None:
                accum = proj
            else:
                accum = accum + proj
        fused = accum / float(self.num_views)
        return fused  # [K, dim]

    def compute_nca_loss_batch(self, users_batch_emb, pos_items_batch_emb, temperature=0.07):
        nca_weight_ui = float(self.config.get('nca_weight_ui', 1.0))
        nca_weight_ii = float(self.config.get('nca_weight_ii', 1.0))
        tau = float(self.config.get('nca_temperature', temperature))

        device = users_batch_emb.device
        B = users_batch_emb.size(0)

        intent_pool = self._fused_intent_pool().to(device)  # [K, d]
        K = intent_pool.size(0)
        d = intent_pool.size(1)

        sqrt_dim = torch.sqrt(torch.tensor(d, dtype=torch.float32, device=device))
        scores_u = torch.matmul(users_batch_emb, intent_pool.t()) / sqrt_dim  # [B, K]
        scores_v = torch.matmul(pos_items_batch_emb, intent_pool.t()) / sqrt_dim  # [B, K]

        topk = min(self.top_k, K)
        topv_u, topi_u = torch.topk(scores_u, topk, dim=1)  # both [B, topk]
        topv_v, topi_v = torch.topk(scores_v, topk, dim=1)

        attn_u_pos = F.softmax(topv_u / tau, dim=1)  # [B, topk]
        attn_v_pos = F.softmax(topv_v / tau, dim=1)  # [B, topk]

        proj_users = F.normalize(self.proj_mlp(users_batch_emb), dim=1)  # [B, d_proj]
        proj_pos_items = F.normalize(self.proj_mlp(pos_items_batch_emb), dim=1)
        proj_intents = F.normalize(self.proj_mlp(intent_pool), dim=1)  # [K, d_proj]

        sim_u_all = torch.matmul(proj_users, proj_intents.t()) / tau
        sim_v_all = torch.matmul(proj_pos_items, proj_intents.t()) / tau

        exp_sim_u_all = torch.exp(sim_u_all)  # [B, K]
        denom_u = exp_sim_u_all.sum(dim=1)  # [B]

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, topk)  # [B, topk]
        sel_sim_u = sim_u_all[batch_idx, topi_u]  # [B, topk]
        num_u = (attn_u_pos * torch.exp(sel_sim_u)).sum(dim=1)  # [B]

        denom_u = denom_u.clamp_min(1e-12)
        L_UI = - torch.log((num_u / denom_u).clamp_min(1e-12)).mean()

        exp_sim_v_all = torch.exp(sim_v_all)  # [B, K]
        denom_v = exp_sim_v_all.sum(dim=1)  # [B]

        sel_sim_v = sim_v_all[batch_idx, topi_v]  # [B, topk]
        num_v = (attn_v_pos * torch.exp(sel_sim_v)).sum(dim=1)  # [B]
        denom_v = denom_v.clamp_min(1e-12)
        L_II = - torch.log((num_v / denom_v).clamp_min(1e-12)).mean()

        L_nca = nca_weight_ui * L_UI + nca_weight_ii * L_II
        return L_nca



    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2))/float(len(users))
        
        intent_reg = 0
        if self.intent_initialized and self.current_epoch >= self.warmup_epochs:
            # sum over views
            for v in range(self.num_views):
                intent_reg = intent_reg + (1/2)*(self.intent_embeddings[v].norm(2).pow(2) + 
                               self.user_intent_scales[v].norm(2).pow(2) +
                               self.item_intent_scales[v].norm(2).pow(2))/float(len(users))
            reg_loss = reg_loss + intent_reg
            
            
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        bpr_loss_val = torch.mean(F.softplus(neg_scores - pos_scores))
        
        align_lambda = float(self.config.get('lambda_align', 0.0))
        align_loss = 0.0
        if align_lambda > 0:
            for v in range(self.num_views):
                if self.intent_anchors[v] is not None:
                    anchor = torch.tensor(self.intent_anchors[v], dtype=torch.float32).to(self.intent_embeddings[v].device)
                    align_loss = align_loss + F.mse_loss(self.intent_embeddings[v], anchor) * align_lambda

        
        total_loss = bpr_loss_val + align_loss
        
        iva_lambda = float(self.config.get('lambda_iva', 0.0))

        if iva_lambda > 0 and self.num_views >= 2 and self.current_epoch >= self.warmup_epochs + self.iva_delay:
            if self.iva_kind == 'kl':
                total_loss = total_loss + iva_lambda * self.iva_kl_loss()
            elif self.iva_kind == 'infonce':
                total_loss = total_loss + iva_lambda * self.iva_infonce_loss()
        
        nca_lambda = float(self.config.get('lambda_nca', 0.0))
        if nca_lambda > 0 and self.intent_initialized and self.current_epoch >= self.warmup_epochs+ self.nca_delay:
            nca_loss_batch = self.compute_nca_loss_batch(users_emb, pos_emb, temperature=float(self.config.get('nca_temperature', 0.07)))
            total_loss = total_loss + nca_lambda * nca_loss_batch

        return total_loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
