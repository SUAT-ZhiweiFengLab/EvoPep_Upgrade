import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 氨基酸词表
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_net = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        B, N, H = x.shape
        # 创建节点对特征
        v_expand = x.unsqueeze(2).expand(B, N, N, H)
        u_expand = x.unsqueeze(1).expand(B, N, N, H)
        msg_in = torch.cat([v_expand, u_expand], dim=-1) 
        msgs = F.relu(self.message_net(msg_in))          
        
        # 使用邻接矩阵 Mask 掉不相连的边
        adj_mask = adj.unsqueeze(-1)                     
        masked_msgs = msgs * adj_mask
        agg_msgs = masked_msgs.sum(dim=2)                
        
        # GRU 状态更新 (继承自 EpiMII 核心思想)
        agg_msgs_flat = agg_msgs.view(B * N, H)
        x_flat = x.view(B * N, H)
        updated_x_flat = self.update_net(agg_msgs_flat, x_flat)
        
        return updated_x_flat.view(B, N, H)

class EvoPepHybridGenerator(nn.Module):
    def __init__(self, vocab_size=20, d_model=64, nhead=4, mpnn_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 分支 1: 继承自 EpiMII (MPNN 捕捉拓扑图信息)
        self.mpnn_layers = nn.ModuleList([MPNNLayer(d_model) for _ in range(mpnn_layers)])
        
        # 分支 2: 优化增强模块 (CNN + Attention)
        self.cnn = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x):
        B, N = x.shape
        emb = self.embedding(x) 
        
        # 针对环肽构建拓扑图的邻接矩阵
        adj = torch.zeros(B, N, N, device=x.device)
        for i in range(N):
            adj[:, i, (i+1)%N] = 1.0 # 模拟环肽首尾相连
            adj[:, i, (i-1)%N] = 1.0 
            adj[:, i, i] = 1.0       
            
        mpnn_out = emb
        for layer in self.mpnn_layers:
            mpnn_out = layer(mpnn_out, adj) 
            
        cnn_in = emb.transpose(1, 2)
        cnn_out = F.relu(self.cnn(cnn_in)).transpose(1, 2) 
        attn_out, _ = self.attention(cnn_out, cnn_out, cnn_out) 
        
        fused_features = torch.cat([mpnn_out, attn_out], dim=-1) 
        logits = self.fc_out(fused_features)
        
        return logits

    def generate_variants(self, seed_sequences, num_variants=5, temperature=1.0, mutation_rate=0.2):
        """
        [架构优化的体现]: 真正利用训练好的混合神经网络计算 Logits 来引导序列变异，
        而不是完全随机突变。
        """
        self.eval()
        variants = set()
        rng = random.Random(42)
        gen = torch.Generator()
        gen.manual_seed(42)
        
        for seq in seed_sequences:
            variants.add(seq) # 保留精英种子
            
            # 转为张量输入网络
            seq_idx = [AA_TO_IDX.get(aa, 0) for aa in seq]
            x = torch.tensor([seq_idx], dtype=torch.long)
            
            with torch.no_grad():
                logits = self.forward(x) # 网络前向计算得到 [1, N, Vocab] 的输出
                
            probs = F.softmax(logits / temperature, dim=-1)[0] # 转为概率分布
            
            # 根据神经网络算出的概率进行定向采样变异
            for _ in range(num_variants):
                new_seq = []
                for i in range(len(seq)):
                    if rng.random() < mutation_rate:
                        sampled_idx = torch.multinomial(probs[i], 1, generator=gen).item()
                        new_seq.append(IDX_TO_AA[sampled_idx])
                    else:
                        new_seq.append(seq[i])
                variants.add("".join(new_seq))
                
        return sorted(variants)

class ConformalQSARPredictor:
    AMINO_HYDROPHOBICITY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    TARGET_PROFILES = {
        'GLP-1R': (0.055, 0.30, 0.12, 0.60, 6.5),
        'GRIN2A': (0.020, 0.15, 0.05, 0.80, 5.8),
        'HDAC6': (0.015, 0.10, 0.10, 0.50, 5.6),
        'CHRNA7': (0.025, 0.20, 0.15, 0.70, 6.0),
        'SIGMAR1': (0.010, 0.45, -0.05, 1.00, 5.5),
        'PI3K': (0.018, 0.12, 0.08, 0.60, 5.7),
        'MEK1': (0.015, 0.18, 0.06, 0.55, 5.6),
    }
    TEMPLATE_SEEDS = [
        'HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS',
        'HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG',
        'HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPSKKKKK',
        'HAEGTFTSDVSISYLEGQAAKEFIAWLVKGR',
    ]

    def __init__(self, target_names, alpha=0.1, n_calib=300):
        self.target_names = target_names
        self.alpha = alpha
        self.qhat = self._calibrate(n_calib)

    def _features(self, seq):
        n = len(seq)
        hydro = sum(self.AMINO_HYDROPHOBICITY[aa] for aa in seq) / n
        charge = sum(1 if aa in 'KRH' else -1 if aa in 'DE' else 0 for aa in seq)
        arom = sum(1 for aa in seq if aa in 'FWY') / n
        return n, hydro, charge, arom

    def _reference_score(self, target, seq):
        w_len, w_hydro, w_charge, w_arom, bias = self.TARGET_PROFILES[target]
        n, hydro, charge, arom = self._features(seq)
        return bias + w_len * n + w_hydro * hydro + w_charge * charge + w_arom * arom

    def _simple_score(self, seq, target):
        n, hydro, charge, arom = self._features(seq)
        w_len, w_hydro, w_charge, w_arom, bias = self.TARGET_PROFILES[target]
        return bias + w_len * n + 0.5 * w_hydro * hydro + w_charge * charge + 0.5 * w_arom * arom

    def _calibrate(self, n_calib):
        rng = random.Random(42)
        residuals = {target: [] for target in self.target_names}
        for _ in range(n_calib):
            seed = rng.choice(self.TEMPLATE_SEEDS)
            seq = ''.join(aa if rng.random() > 0.15 else rng.choice(AMINO_ACIDS) for aa in seed)
        for target in self.target_names:
            simple = self._simple_score(seq, target)
            residuals[target].append(abs(self._reference_score(target, seq) - simple))
        qhat = {}
        for target in self.target_names:
            residuals[target].sort()
            idx = math.ceil((len(residuals[target]) + 1) * (1 - self.alpha)) - 1
            qhat[target] = residuals[target][min(idx, len(residuals[target]) - 1)]
        return qhat

    def predict(self, sequences):
        results = []
        for seq in sequences:
            scores = {target: self._reference_score(target, seq) for target in self.target_names}
            intervals = {target: self.qhat[target] for target in self.target_names}
            results.append({'sequence': seq, 'scores': scores, 'intervals': intervals})
        return results