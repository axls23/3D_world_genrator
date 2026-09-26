import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        # Director Agent Architecture from insights
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim//2), # 256 -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        
        # Dimensions from insights: State=32, Action=4, Hidden=256
        input_dim = getattr(config, 'INPUT_DIM', 32)
        hidden_dim = getattr(config, 'HIDDEN_DIM', 256)
        action_dim = getattr(config, 'ACTION_DIM', 4)
        
        self.q_net = QNetwork(input_dim, hidden_dim, action_dim).to(self.device)
        self.q_net.eval()
        
    def load(self, path):
        if torch.cuda.is_available():
            self.q_net.load_state_dict(torch.load(path))
        else:
            self.q_net.load_state_dict(torch.load(path, map_location='cpu'))
            
    def predict(self, state_tensor):
        """
        Predict action index for a single state or batch.
        state_tensor: [Batch, 32] or [32]
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            q_values = self.q_net(state_tensor)
            action_idx = torch.argmax(q_values, dim=1)
            
        return action_idx

def _quantile(x, q):
    """q-quantile (q in [0, 1]) of a 1-D tensor; sort-based so it works beyond
    torch.quantile's 16M-element limit."""
    x = x.flatten()
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    v, _ = torch.sort(x.float())
    pos = q * (v.numel() - 1)
    lo = int(pos)
    hi = min(lo + 1, v.numel() - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _robust_z(x):
    """(x - median) / (1.4826 * MAD): a z-score that a few extreme splats can't inflate."""
    med = _quantile(x, 0.5)
    mad = _quantile((x - med).abs(), 0.5)
    return (x - med) / (1.4826 * mad + 1e-6)


def _knn_indices(positions, k):
    """(N, k) neighbour indices (self excluded) via CPU sklearn."""
    from sklearn.neighbors import NearestNeighbors
    pos = positions.detach().cpu().numpy()
    n = pos.shape[0]
    k = min(k, max(n - 1, 1))
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), n_jobs=-1).fit(pos)
    _, idx = nbrs.kneighbors(pos)
    return torch.from_numpy(idx[:, 1:].astype(np.int64))


class HeuristicAgent:
    """
    Fallback agent that uses strict thresholds (Rule-Based)
    Instead of learned Q-values, it checks:
    1. Is opacity too low?
    2. Is point too isolated?

    Thresholds are percentiles of the model's own opacity / isolation distributions
    (HEURISTIC_*_PCT); the absolute PRUNE_OPACITY_THR (logit) and PRUNE_ISOLATION_THR
    (scene units) are used only when a percentile is set to None.
    """
    def __init__(self, config):
        self.config = config

    def thresholds(self, opacities, isolation):
        """(opacity_thr, isolation_thr, ghost_isolation_thr) for this model."""
        cfg = self.config
        op_pct = getattr(cfg, 'HEURISTIC_OPACITY_PCT', None)
        iso_pct = getattr(cfg, 'HEURISTIC_ISOLATION_PCT', None)
        ghost_pct = getattr(cfg, 'HEURISTIC_GHOST_ISOLATION_PCT', None)
        op_thr = _quantile(opacities, op_pct / 100.0) if op_pct is not None else cfg.PRUNE_OPACITY_THR
        iso_thr = _quantile(isolation, iso_pct / 100.0) if iso_pct is not None else cfg.PRUNE_ISOLATION_THR
        ghost_thr = (_quantile(isolation, ghost_pct / 100.0) if ghost_pct is not None
                     else cfg.PRUNE_ISOLATION_THR * 0.5)
        return op_thr, iso_thr, ghost_thr

    def predict(self, states, raw_data):
        # Features mapping from StateExtractor:
        # 0: Opacity (logit)
        # 1-3: Scale Norm
        # 4-6: Color
        # 7: Isolation (mean kNN distance)
        # 8-10: Pos Norm
        opacities = states[:, 0]
        isolation = states[:, 7]

        OPACITY_THR, ISOLATION_THR, GHOST_ISOLATION_THR = self.thresholds(opacities, isolation)

        # Prune if (Transparent AND Isolated) OR (Very Isolated)
        is_floater = isolation > ISOLATION_THR
        is_ghost = (opacities < OPACITY_THR) & (isolation > GHOST_ISOLATION_THR)

        to_prune = is_floater | is_ghost

        return to_prune.long() # 0 = KEEP (False), 1 = PRUNE (True)


class ContextAwareAgent:
    """
    Context-aware pruning agent. Each Gaussian gets a suspicion score in [0, 1] from:
    1. Isolation (mean kNN distance, percentile-normalized)
    2. Low opacity (sigmoid of the logit)
    3. Scale anomalies, in linear units (PLY scales are log-scales):
       a. needles: max / middle axis (flat discs are legitimate surface splats)
       b. giants: largest axis relative to the local neighbour spacing
    4. Colour inconsistency with its kNN neighbours (quarter-nearest in colour) (not with the global mean colour, which
       flagged every saturated-but-coherent region)
    5. Depth outlier (distance from the scene centre)
    6. Combined suspicious pattern (isolated AND another anomaly)

    Decision (percentile budget instead of a fixed score cutoff):
      - candidates are the top PRUNE_BUDGET_FRAC (default 3%) scores of the model;
      - of those, only Gaussians that are isolated (mean kNN distance above the model's
        PRUNE_MIN_ISOLATION_PCT percentile) AND not opaque (opacity at or below the model's
        PRUNE_PROTECT_OPACITY_PCT percentile) are pruned.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')

    def feature_scores(self, states, raw_data):
        """Per-feature scores in [0, 1] (dict of (N,) tensors) and the total score."""
        cfg = self.config
        states = states.to(self.device)
        opacities = states[:, 0]
        colors = states[:, 4:7]
        isolation = states[:, 7]
        pos_norm = states[:, 8:11]

        log_scales = raw_data['scales'].to(self.device).float()
        knn_idx = raw_data.get('knn_idx')
        if knn_idx is None:
            knn_idx = _knn_indices(raw_data['positions'], getattr(cfg, 'DENSITY_K', 16))
        knn_idx = knn_idx.to(self.device)

        f = {}
        # 1. Isolation: 0 at the median spacing, 1 at the 99th percentile
        iso_p50 = _quantile(isolation, 0.5)
        iso_p99 = _quantile(isolation, 0.99)
        f['isolation'] = torch.clamp((isolation - iso_p50) / (iso_p99 - iso_p50 + 1e-9), 0, 1)

        # 2. Opacity: raw values are logits
        f['opacity'] = 1.0 - torch.sigmoid(opacities)

        # 3a. Needle: max / middle linear scale (exp of the log-scales)
        s_sorted, _ = torch.sort(torch.exp(log_scales), dim=1)
        needle = s_sorted[:, 2] / (s_sorted[:, 1] + 1e-12)
        f['needle'] = torch.clamp((needle - 1) / cfg.PRUNE_SCALE_RATIO_THR, 0, 1)

        # 3b. Giant: largest axis vs local spacing, robust z in log space
        spacing = isolation.clamp_min(1e-12)
        rel_size = log_scales.max(dim=1).values - torch.log(spacing)
        f['giant'] = torch.clamp(_robust_z(rel_size) / cfg.PRUNE_SCALE_OUTLIER_THR, 0, 1)

        # 4. Colour vs its kNN neighbours: distance to the neighbour at the 25th percentile
        #    of colour distance, i.e. "does at least a quarter of my neighbourhood look like
        #    me?". Low for coherent saturated regions and for splats on a colour edge; high
        #    only for a splat unlike everything around it.
        if knn_idx.shape[1] > 0:
            dists = torch.norm(colors[knn_idx] - colors[:, None, :], dim=2)  # (N, K)
            q = max(int(np.ceil(0.25 * knn_idx.shape[1])), 1)
            color_dev = torch.kthvalue(dists, q, dim=1).values
            f['color'] = torch.clamp(_robust_z(color_dev) / cfg.PRUNE_COLOR_OUTLIER_THR, 0, 1)
        else:
            f['color'] = torch.zeros_like(opacities)

        # 5. Depth outlier: distance from the scene centre
        depth = torch.norm(pos_norm, dim=1)
        f['depth'] = torch.clamp(_robust_z(depth) / cfg.PRUNE_DEPTH_OUTLIER_THR, 0, 1)

        # 6. Suspicious = isolated AND (low opacity OR giant OR odd colour)
        f['suspicious'] = ((f['isolation'] > 0.3) & (
            (f['opacity'] > 0.5) | (f['giant'] > 0.3) | (f['color'] > 0.5))).float()

        weights = {'isolation': 0.25, 'opacity': 0.20, 'needle': 0.10, 'giant': 0.10,
                   'color': 0.10, 'depth': 0.15, 'suspicious': 0.10}
        total = sum(f[k] * w for k, w in weights.items())
        return f, total

    def predict(self, states, raw_data):
        """
        Features in states tensor (from StateExtractor):
        0: Opacity (logit)
        1-3: Scale (normalized)
        4-6: Color (SH DC)
        7: Isolation (mean neighbor distance)
        8-10: Position (normalized)
        raw_data: 'scales' (log-scales), 'positions', optional 'knn_idx' (N, K)
        """
        cfg = self.config
        N = states.shape[0]
        print(f"[ContextAware] Running multi-feature analysis on {N} Gaussians...")
        _, scores = self.feature_scores(states, raw_data)
        opacities = states[:, 0].to(self.device)
        isolation = states[:, 7].to(self.device)

        # Safety gates from the model's own distribution
        op_cap = _quantile(opacities, cfg.PRUNE_PROTECT_OPACITY_PCT / 100.0)
        iso_min = _quantile(isolation, cfg.PRUNE_MIN_ISOLATION_PCT / 100.0)
        eligible = (opacities <= op_cap) & (isolation >= iso_min) & (scores > 0)

        # Budget: at most PRUNE_BUDGET_FRAC of the highest-scoring Gaussians
        budget = int(np.floor(cfg.PRUNE_BUDGET_FRAC * N))
        to_prune = torch.zeros(N, dtype=torch.bool, device=self.device)
        if budget > 0:
            top = torch.topk(scores, budget).indices
            to_prune[top] = True
            to_prune &= eligible

        n_prune = int(to_prune.sum().item())
        print(f"[ContextAware] Score distribution: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
        print(f"[ContextAware] Gates: opacity<=p{cfg.PRUNE_PROTECT_OPACITY_PCT:g} "
              f"(sigmoid {torch.sigmoid(op_cap).item():.3f}), isolation>=p{cfg.PRUNE_MIN_ISOLATION_PCT:g} "
              f"({iso_min.item():.4g}); budget {budget} ({100 * cfg.PRUNE_BUDGET_FRAC:.1f}%)")
        print(f"[ContextAware] Decisions: KEEP={N - n_prune}, PRUNE={n_prune} ({100*n_prune/max(N, 1):.1f}%)")

        return to_prune.long()  # 0 = KEEP, 1 = PRUNE
