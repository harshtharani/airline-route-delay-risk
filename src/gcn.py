"""Unsupervised GCN that learns a numeric "structural fingerprint" for every airport.

Training signal: link prediction. The model must give connected airport pairs a higher score than random
pairs, so it has to encode who an airport is connected to. No delay labels are used.
The encoder is trained ONLY on the first training month's graph, then applied unchanged to later months.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_undirected
from config import EMBED_DIM, GCN_HIDDEN, GCN_EPOCHS, GCN_LR, SEED


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden=GCN_HIDDEN, out_dim=EMBED_DIM):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.c1(x, edge_index, edge_weight))
        return self.c2(h, edge_index, edge_weight)


def _node_matrix(metrics, nodes):
    m = metrics.set_index("airport").loc[nodes]
    return np.column_stack([np.log1p(m.in_degree), np.log1p(m.out_degree),
                            np.log1p(m.in_flights), np.log1p(m.out_flights),
                            m.pagerank * len(nodes), m.clustering]).astype(np.float32)


def _graph_tensors(G, metrics):
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    src = [idx[u] for u, v in G.edges]
    dst = [idx[v] for u, v in G.edges]
    w = torch.tensor([np.log1p(d["flights"]) for _, _, d in G.edges(data=True)], dtype=torch.float32)
    ei = torch.tensor([src, dst], dtype=torch.long)
    ei, ew = to_undirected(ei, w, num_nodes=len(nodes), reduce="max")
    ew = ew / ew.max()
    return nodes, _node_matrix(metrics, nodes), ei, ew


def train_gcn(G, metrics, epochs=GCN_EPOCHS, lr=GCN_LR, verbose=True):
    torch.manual_seed(SEED); np.random.seed(SEED)
    nodes, X, ei, ew = _graph_tensors(G, metrics)
    scaler = StandardScaler().fit(X)
    x = torch.tensor(scaler.transform(X), dtype=torch.float32)
    enc = Encoder(x.shape[1])
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-5)
    losses = []
    for ep in range(1, epochs + 1):
        enc.train(); opt.zero_grad()
        z = enc(x, ei, ew)
        neg = negative_sampling(ei, num_nodes=len(nodes), num_neg_samples=ei.size(1))
        pos_s = (z[ei[0]] * z[ei[1]]).sum(-1)
        neg_s = (z[neg[0]] * z[neg[1]]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_s, neg_s]),
                                                  torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)]))
        loss.backward(); opt.step()
        losses.append(loss.item())
        if verbose and (ep == 1 or ep % 50 == 0):
            print(f"  GCN epoch {ep:3d}  loss {loss:.4f}")
    return {"encoder": enc, "scaler": scaler, "losses": losses}


@torch.no_grad()
def embed(bundle, G, metrics):
    """Embeddings for any month's graph, using the encoder trained on the first training month."""
    nodes, X, ei, ew = _graph_tensors(G, metrics)
    x = torch.tensor(bundle["scaler"].transform(X), dtype=torch.float32)
    bundle["encoder"].eval()
    z = bundle["encoder"](x, ei, ew).numpy()
    return pd.DataFrame(z, columns=[f"emb{k}" for k in range(z.shape[1])]).assign(airport=nodes)


@torch.no_grad()
def link_auc(bundle, G, metrics):
    """Sanity check: do embeddings rank real routes above random airport pairs?"""
    nodes, X, ei, ew = _graph_tensors(G, metrics)
    x = torch.tensor(bundle["scaler"].transform(X), dtype=torch.float32)
    bundle["encoder"].eval()
    z = bundle["encoder"](x, ei, ew)
    neg = negative_sampling(ei, num_nodes=len(nodes), num_neg_samples=ei.size(1))
    pos_s = (z[ei[0]] * z[ei[1]]).sum(-1).numpy()
    neg_s = (z[neg[0]] * z[neg[1]]).sum(-1).numpy()
    return float(roc_auc_score(np.r_[np.ones_like(pos_s), np.zeros_like(neg_s)], np.r_[pos_s, neg_s]))
