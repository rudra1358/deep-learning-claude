import type { Lesson } from "@/lib/types";

const GNN_NUMPY = String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
# Part 1. One GCN layer and one GAT layer on a 4-node graph, every number shown
# ---------------------------------------------------------------------------
edges = [(0, 1), (0, 2), (0, 3), (2, 3)]          # undirected
n = 4
A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = A[j, i] = 1.0
X = np.array([[1.0, 0.0],                         # node features, one row per node
              [0.0, 1.0],
              [1.0, 1.0],
              [2.0, 0.0]])
W = np.array([[0.5, -1.0],                        # layer weights (2 in -> 2 out)
              [1.0, 0.5]])

A_tilde = A + np.eye(n)                            # add self-loops
d_tilde = A_tilde.sum(axis=1)                      # degrees including the self-loop
D_inv_sqrt = np.diag(1.0 / np.sqrt(d_tilde))
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt          # symmetric normalisation
print("degrees with self-loops:", d_tilde)
print("A_hat =\n", A_hat)
XW = X @ W
H = np.maximum(A_hat @ XW, 0.0)                    # GCN layer: ReLU(A_hat X W)
print("XW =\n", XW)
print("A_hat X W =\n", A_hat @ XW)
print("GCN output ReLU(A_hat X W) =\n", H)


def leaky_relu(z, slope=0.2):
    return np.where(z > 0, z, slope * z)


def gat_layer(A, X, W, a, slope=0.2):
    """Single-head GAT layer (no output non-linearity). Returns new features and attention matrix."""
    Z = X @ W                                      # transformed features, (n, F')
    F = Z.shape[1]
    src = Z @ a[:F]                                # a_l^T z_i  (for the receiving node i)
    dst = Z @ a[F:]                                # a_r^T z_j  (for the sending node j)
    E = leaky_relu(src[:, None] + dst[None, :], slope)        # e_ij for every pair
    mask = (A + np.eye(len(A))) > 0                # attend only over N(i) and i itself
    E = np.where(mask, E, -np.inf)
    E = E - E.max(axis=1, keepdims=True)           # stable softmax per row
    alpha = np.exp(E)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)
    return alpha @ Z, alpha


a = np.array([1.0, 0.0, 0.5, 1.0])                 # attention vector [a_l ; a_r], length 2F'
H_gat, alpha = gat_layer(A, X, W, a)
print("GAT attention alpha (row i = weights node i gives its neighbours) =\n", alpha)
print("GAT output =\n", H_gat)

# ---------------------------------------------------------------------------
# Part 2. Semi-supervised node classification with a 2-layer GCN, manual backprop
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
K, per = 3, 40                                     # 3 communities of 40 nodes
N = K * per
y = np.repeat(np.arange(K), per)
P_IN, P_OUT = 0.10, 0.01                           # edge probabilities inside / between communities
prob = np.where(y[:, None] == y[None, :], P_IN, P_OUT)
upper = np.triu(rng.random((N, N)) < prob, k=1)
A = (upper | upper.T).astype(float)
F_IN = 8
means = rng.normal(0, 1, (K, F_IN))
X = means[y] + rng.normal(0, 2.0, (N, F_IN))       # weak, noisy features (noise std 2 >> signal)
print(f"\nSBM graph: {N} nodes, {int(A.sum() / 2)} edges, mean degree {A.sum() / N:.2f}")

train_idx = np.concatenate([np.where(y == c)[0][:3] for c in range(K)])   # only 3 labels per class
test_mask = np.ones(N, bool)
test_mask[train_idx] = False

A_tilde = A + np.eye(N)
d = A_tilde.sum(1)
A_hat = A_tilde / np.sqrt(d[:, None] * d[None, :])
Y = np.eye(K)[y]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, A_hat, X):
    W1, W2 = params
    AX = A_hat @ X
    Z1 = AX @ W1                                   # layer-1 pre-activation
    H1 = np.maximum(Z1, 0)
    AH = A_hat @ H1
    logits = AH @ W2
    return logits, (AX, Z1, H1, AH)


def loss_and_grads(params, A_hat, X, idx):
    W1, W2 = params
    logits, (AX, Z1, H1, AH) = forward(params, A_hat, X)
    P = softmax(logits)
    loss = -np.mean(np.log(P[idx, y[idx]]))
    dlogits = np.zeros_like(P)                     # cross-entropy only on labelled nodes
    dlogits[idx] = (P[idx] - Y[idx]) / len(idx)
    dW2 = AH.T @ dlogits
    dH1 = A_hat.T @ dlogits @ W2.T                 # A_hat is symmetric, .T kept for clarity
    dZ1 = dH1 * (Z1 > 0)
    dW1 = AX.T @ dZ1
    return loss, [dW1, dW2]


HID = 16
params = [rng.normal(0, np.sqrt(2 / F_IN), (F_IN, HID)), rng.normal(0, np.sqrt(2 / HID), (HID, K))]

# gradient check of dW1[0, 0] with a central difference
loss0, grads = loss_and_grads(params, A_hat, X, train_idx)
eps = 1e-6
params[0][0, 0] += eps
lp, _ = loss_and_grads(params, A_hat, X, train_idx)
params[0][0, 0] -= 2 * eps
lm, _ = loss_and_grads(params, A_hat, X, train_idx)
params[0][0, 0] += eps
print(f"gradient check dW1[0,0]: analytic {grads[0][0, 0]:.8f}  numeric {(lp - lm) / (2 * eps):.8f}")

LR, WD = 0.2, 5e-4                                 # learning rate and L2 weight decay
for epoch in range(201):
    loss, grads = loss_and_grads(params, A_hat, X, train_idx)
    params = [p - LR * (g + WD * p) for p, g in zip(params, grads)]
    if epoch % 50 == 0:
        pred = forward(params, A_hat, X)[0].argmax(1)
        print(f"epoch {epoch:3d}  train loss {loss:.4f}  test acc {np.mean(pred[test_mask] == y[test_mask]):.3f}")

# The same 2-layer network with A_hat replaced by the identity is an MLP that ignores the graph.
params_mlp = [rng.normal(0, np.sqrt(2 / F_IN), (F_IN, HID)), rng.normal(0, np.sqrt(2 / HID), (HID, K))]
I = np.eye(N)
for epoch in range(201):
    loss, grads = loss_and_grads(params_mlp, I, X, train_idx)
    params_mlp = [p - LR * (g + WD * p) for p, g in zip(params_mlp, grads)]
pred = forward(params_mlp, I, X)[0].argmax(1)
print(f"MLP (no graph), same training: test acc {np.mean(pred[test_mask] == y[test_mask]):.3f}")

# ---------------------------------------------------------------------------
# Part 3. Over-smoothing: features after k propagation steps, A_hat^k X
# ---------------------------------------------------------------------------
print("\n    k   energy tr(H^T (I - A_hat) H)   nearest-centroid test acc")
Hk = X.copy()
for k in range(257):
    if k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        energy = np.trace(Hk.T @ (Hk - A_hat @ Hk))
        Hn = Hk / np.linalg.norm(Hk, axis=1, keepdims=True)              # compare directions only
        cent = np.stack([Hn[train_idx[y[train_idx] == c]].mean(0) for c in range(K)])
        pred = (Hn @ cent.T).argmax(1)
        print(f"{k:5d}   {energy:12.3e}                  {np.mean(pred[test_mask] == y[test_mask]):.3f}")
    Hk = A_hat @ Hk`;
const GNN_NUMPY_OUT = String.raw`degrees with self-loops: [4. 2. 3. 3.]
A_hat =
 [[0.25   0.3536 0.2887 0.2887]
 [0.3536 0.5    0.     0.    ]
 [0.2887 0.     0.3333 0.3333]
 [0.2887 0.     0.3333 0.3333]]
XW =
 [[ 0.5 -1. ]
 [ 1.   0.5]
 [ 1.5 -0.5]
 [ 1.  -2. ]]
A_hat X W =
 [[ 1.2002 -0.7949]
 [ 0.6768 -0.1036]
 [ 0.9777 -1.122 ]
 [ 0.9777 -1.122 ]]
GCN output ReLU(A_hat X W) =
 [[1.2002 0.    ]
 [0.6768 0.    ]
 [0.9777 0.    ]
 [0.9777 0.    ]]
GAT attention alpha (row i = weights node i gives its neighbours) =
 [[0.1137 0.5355 0.253  0.0978]
 [0.148  0.852  0.     0.    ]
 [0.2386 0.     0.6487 0.1127]
 [0.2261 0.     0.6146 0.1593]]
GAT output =
 [[ 1.0697 -0.168 ]
 [ 0.926   0.2779]
 [ 1.205  -0.7884]
 [ 1.1942 -0.852 ]]

SBM graph: 120 nodes, 282 edges, mean degree 4.70
gradient check dW1[0,0]: analytic -0.10957188  numeric -0.10957188
epoch   0  train loss 1.6754  test acc 0.324
epoch  50  train loss 0.0263  test acc 0.937
epoch 100  train loss 0.0113  test acc 0.946
epoch 150  train loss 0.0070  test acc 0.946
epoch 200  train loss 0.0050  test acc 0.946
MLP (no graph), same training: test acc 0.396

    k   energy tr(H^T (I - A_hat) H)   nearest-centroid test acc
    0      3.340e+03                  0.568
    1      3.851e+02                  0.748
    2      1.204e+02                  0.883
    4      3.793e+01                  0.928
    8      1.135e+01                  0.955
   16      2.064e+00                  0.946
   32      1.388e-01                  0.919
   64      1.045e-03                  0.892
  128      6.521e-08                  0.892
  256      2.544e-15                  0.351`;
const GNN_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)
rng = np.random.default_rng(1)

# ---- data: a stochastic block model with 4 communities and weak node features ----
K, PER, F_IN = 4, 75, 16
N = K * PER
y_np = np.repeat(np.arange(K), PER)
prob = np.where(y_np[:, None] == y_np[None, :], 0.08, 0.005)
upper = np.triu(rng.random((N, N)) < prob, k=1)
src, dst = np.nonzero(upper | upper.T)                    # both directions of every undirected edge
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)   # shape (2, E): row 0 sends, row 1 receives
means = rng.normal(0, 1, (K, F_IN))
x = torch.tensor(means[y_np] + rng.normal(0, 2.5, (N, F_IN)), dtype=torch.float32)
y = torch.tensor(y_np)
perm = rng.permutation(N)
train_idx = torch.tensor(np.concatenate([perm[y_np[perm] == c][:5] for c in range(K)]))   # 5 labels per class
val_idx = torch.tensor(perm[np.isin(perm, train_idx.numpy(), invert=True)][:60])
test_mask = torch.ones(N, dtype=torch.bool)
test_mask[train_idx] = False
test_mask[val_idx] = False
print(f"{N} nodes, {edge_index.shape[1] // 2} undirected edges, {len(train_idx)} labelled nodes, {int(test_mask.sum())} test nodes")


def add_self_loops(edge_index, n):
    loops = torch.arange(n).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


class GCNLayer(nn.Module):
    """H' = D~^-1/2 (A + I) D~^-1/2 H W, written as message passing over an edge list."""

    def __init__(self, f_in, f_out):
        super().__init__()
        self.lin = nn.Linear(f_in, f_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(f_out))

    def forward(self, h, edge_index):
        n = h.shape[0]
        ei = add_self_loops(edge_index, n)
        s, r = ei                                                   # sender, receiver of every message
        deg = torch.zeros(n).index_add_(0, r, torch.ones(r.shape[0]))   # d~ (self-loop included)
        norm = deg[s].rsqrt() * deg[r].rsqrt()                      # 1 / sqrt(d~_s d~_r) per edge
        z = self.lin(h)                                             # transform first (cheaper if f_out < f_in)
        msg = z[s] * norm[:, None]                                  # message from s to r
        out = torch.zeros(n, z.shape[1]).index_add_(0, r, msg)      # aggregate: sum over incoming edges
        return out + self.bias


class GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al. 2018); heads are concatenated or averaged."""

    def __init__(self, f_in, f_out, heads=4, concat=True, slope=0.2, dropout=0.0):
        super().__init__()
        self.h, self.f, self.concat, self.slope, self.dropout = heads, f_out, concat, slope, dropout
        self.lin = nn.Linear(f_in, heads * f_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, f_out))       # a_r: scores the sending node
        self.a_dst = nn.Parameter(torch.empty(heads, f_out))       # a_l: scores the receiving node
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, edge_index, return_attention=False):
        n = h.shape[0]
        s, r = add_self_loops(edge_index, n)
        z = self.lin(h).view(n, self.h, self.f)                     # (n, heads, f)
        e = F.leaky_relu((z * self.a_dst).sum(-1)[r] + (z * self.a_src).sum(-1)[s], self.slope)   # (E, heads)
        # softmax over the incoming edges of every receiver r (subtract the per-receiver max for stability)
        e_max = torch.full((n, self.h), -1e30).scatter_reduce(0, r[:, None].expand(-1, self.h), e, reduce="amax")
        w = torch.exp(e - e_max[r])
        denom = torch.zeros(n, self.h).index_add_(0, r, w)
        alpha = w / denom[r]                                        # (E, heads), sums to 1 per receiver
        alpha_d = F.dropout(alpha, self.dropout, self.training)
        out = torch.zeros(n, self.h, self.f).index_add_(0, r, alpha_d[:, :, None] * z[s])
        out = out.reshape(n, self.h * self.f) if self.concat else out.mean(1)
        return (out, (s, r, alpha)) if return_attention else out


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = GCNLayer(F_IN, 32), GCNLayer(32, K)

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.l1(F.dropout(x, 0.5, self.training), ei)), 0.5, self.training)
        return self.l2(h, ei)


class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = GATLayer(F_IN, 8, heads=4, concat=True, dropout=0.3)
        self.l2 = GATLayer(32, K, heads=1, concat=False, dropout=0.3)

    def forward(self, x, ei):
        h = F.dropout(F.elu(self.l1(F.dropout(x, 0.5, self.training), ei)), 0.5, self.training)
        return self.l2(h, ei)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.5), nn.Linear(F_IN, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, K))

    def forward(self, x, ei):
        return self.net(x)


def run(model, epochs=200):
    torch.manual_seed(0)
    for p in model.parameters():                       # re-initialise with the fixed seed
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(x, edge_index)[train_idx], y[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(1)
        val = (pred[val_idx] == y[val_idx]).float().mean().item()
        if val > best_val:                              # model selection on the validation nodes
            best_val, best_test = val, (pred[test_mask] == y[test_mask]).float().mean().item()
    return best_val, best_test


for name, model in [("MLP (ignores edges)", MLP()), ("GCN, 2 layers", GCN()), ("GAT, 2 layers, 4 heads", GAT())]:
    v, t = run(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name:<24} params {n_par:5d}   val acc {v:.3f}   test acc {t:.3f}")

# ---- attention weights learned by the GAT for one node ----
gat = GAT()
run(gat)
gat.eval()
with torch.no_grad():
    _, (s, r, alpha) = gat.l1(x, edge_index, return_attention=True)
node = int(train_idx[0])
inc = (r == node).nonzero().flatten()
print(f"\nnode {node} (class {int(y[node])}): attention of head 0 over its {len(inc)} incoming edges (incl. self-loop)")
for k in inc.tolist():
    print(f"   from node {int(s[k]):3d} (class {int(y[s[k]])})  alpha = {alpha[k, 0].item():.3f}")
print(f"   sum = {alpha[inc, 0].sum().item():.3f}")

# ---- depth: over-smoothing of a plain deep GCN (no residuals, no dropout) ----
print("\nlayers  test acc (plain GCN, hidden 32)")
for depth in (1, 2, 4, 8, 16):
    class Deep(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [F_IN] + [32] * (depth - 1) + [K]
            self.layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))

        def forward(self, h, ei):
            for i, layer in enumerate(self.layers):
                h = layer(h, ei)
                if i < len(self.layers) - 1:
                    h = F.relu(h)
            return h

    print(f"{depth:5d}   {run(Deep())[1]:.3f}")`;
const GNN_TORCH_OUT = String.raw`300 nodes, 1096 undirected edges, 20 labelled nodes, 220 test nodes
MLP (ignores edges)      params   676   val acc 0.617   test acc 0.500
GCN, 2 layers            params   676   val acc 1.000   test acc 0.973
GAT, 2 layers, 4 heads   params   712   val acc 0.983   test acc 0.964

node 8 (class 0): attention of head 0 over its 10 incoming edges (incl. self-loop)
   from node   9 (class 0)  alpha = 0.081
   from node  10 (class 0)  alpha = 0.056
   from node  14 (class 0)  alpha = 0.031
   from node  29 (class 0)  alpha = 0.096
   from node  38 (class 0)  alpha = 0.192
   from node  61 (class 0)  alpha = 0.068
   from node  63 (class 0)  alpha = 0.033
   from node 130 (class 1)  alpha = 0.223
   from node 242 (class 3)  alpha = 0.045
   from node   8 (class 0)  alpha = 0.173
   sum = 1.000

layers  test acc (plain GCN, hidden 32)
    1   0.791
    2   0.991
    4   0.973
    8   0.936
   16   0.232`;
const GCN_KERAS = String.raw`import numpy as np
import tensorflow as tf


class GCNLayer(tf.keras.layers.Layer):
    """H' = A_hat H W + b, with A_hat = D~^-1/2 (A + I) D~^-1/2 passed in as a constant dense tensor."""

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        h_shape, _ = input_shape
        self.w = self.add_weight(shape=(h_shape[-1], self.units), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")

    def call(self, inputs):
        h, a_hat = inputs
        return self.activation(tf.matmul(a_hat, tf.matmul(h, self.w)) + self.b)


def normalise(A):
    A_tilde = A + np.eye(len(A))
    d = A_tilde.sum(1)
    return (A_tilde / np.sqrt(np.outer(d, d))).astype("float32")


# A (n x n adjacency), X (n x f features), y (n labels) and train_idx come from your data
A_hat = tf.constant(normalise(A))
X_t = tf.constant(X, dtype=tf.float32)
l1, l2 = GCNLayer(32, "relu"), GCNLayer(int(y.max()) + 1)
drop = tf.keras.layers.Dropout(0.5)
opt = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(200):
    with tf.GradientTape() as tape:
        logits = l2([drop(l1([drop(X_t, training=True), A_hat]), training=True), A_hat])
        loss = loss_fn(tf.gather(y, train_idx), tf.gather(logits, train_idx))
    variables = l1.trainable_variables + l2.trainable_variables
    opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
    if epoch % 50 == 0:
        print(epoch, float(loss))`;

const lesson: Lesson = {
  id: "m15/graph-neural-networks",
  title: "Graph Neural Networks",
  summary:
    "Learning on graphs: adjacency and degree matrices, the message-passing framework (message, aggregate, update), the GCN layer of Kipf & Welling with self-loops and symmetric normalisation, graph attention (GAT), node-, edge- and graph-level tasks, and over-smoothing — derived by hand, implemented from scratch in NumPy with manual backprop and in plain PyTorch with edge lists.",
  minutes: 85,
  level: "advanced",
  prerequisites: ["m02/vectors-matrices", "m02/eigen-svd", "m04/backpropagation", "m10/scaled-dot-product-attention"],
  next: ["m15/multimodal", "m15/deep-rl", "m11/transformer-architecture"],
  terms: ["graph-neural-network", "adjacency-matrix", "node-degree", "message-passing", "gcn", "gat", "over-smoothing", "homophily", "graph-readout"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Much real data is not a grid or a sequence but a **graph**: atoms joined by bonds, users joined by friendships, papers joined by citations, road segments joined at intersections. A graph has a set of **nodes** (vertices) and a set of **edges** connecting pairs of them, and usually a feature vector on every node (and sometimes on every edge). The number of nodes varies, each node has a different number of neighbours, and there is no natural order of the nodes — so a CNN (which needs a grid) or an RNN (which needs an order) cannot be applied directly.",
      },
      {
        t: "p",
        text: "A **graph neural network** (GNN) computes a vector for every node by repeatedly letting each node collect information from its neighbours. One such round is a **message-passing layer**: every node sends a message along its edges, every node aggregates the messages it receives with an order-independent operation (sum, mean, max or an attention-weighted sum), and updates its own vector with a small neural network. After $k$ layers a node's vector depends on everything within $k$ hops. The weights are shared by all nodes, so one model works on graphs of any size and shape, just as a convolution kernel is shared by all pixels.",
      },
      {
        t: "p",
        text: "Two layers dominate practice and this lesson: the **graph convolutional network** (GCN) of Kipf and Welling (2017, \"Semi-Supervised Classification with Graph Convolutional Networks\"), which averages neighbours with weights fixed by the node degrees, and the **graph attention network** (GAT) of Veličković et al. (2018, \"Graph Attention Networks\"), which learns those weights with attention. GNNs are used for molecular property prediction, recommendation, fraud detection, traffic and weather forecasting, and any problem where relations between entities carry information.",
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`GCN layer: $H^{(l+1)} = \sigma\big(\hat A H^{(l)} W^{(l)}\big)$ with $\hat A = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}$. GAT layer: $h_i' = \sigma\big(\sum_{j\in\mathcal N(i)\cup\{i\}} \alpha_{ij} W h_j\big)$ with $\alpha_{ij} = \mathrm{softmax}_j\,\mathrm{LeakyReLU}\big(a^\top[Wh_i \,\|\, Wh_j]\big)$.`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Vectors and matrices](/learn/m02/vectors-matrices) — matrix products: the whole GCN layer is two matrix multiplications, and row $i$ of $AH$ is the sum of the neighbours' feature rows.",
          "[Eigenvalues and SVD](/learn/m02/eigen-svd) — symmetric matrices have real eigenvalues and orthogonal eigenvectors; over-smoothing is explained by the eigen-decomposition of the normalised adjacency matrix.",
          "[Backpropagation](/learn/m04/backpropagation) — we derive the gradients of a two-layer GCN by hand; they are the MLP gradients with an extra multiplication by $\\hat A^\\top$.",
          "[Scaled dot-product attention](/learn/m10/scaled-dot-product-attention) — GAT is attention restricted to a node's neighbours; softmax-normalised scores, weighted sums of values and multiple heads all reappear.",
          "[Convolution](/learn/m08/convolution) (helpful) — weight sharing and local receptive fields are the ideas GNNs generalise from grids to graphs.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Asking your friends",
        text: "You want to guess someone's political leaning, but you only know a little about them. A good strategy is to ask their friends, and their friends' friends: people tend to be connected to people like them. Each round of asking is one GNN layer. In round one, everybody averages what they know with what their immediate friends know; in round two, the information has already been through friends-of-friends. A GCN treats every friend's opinion with a weight that depends on how popular they are (a celebrity with thousands of friends gets a small voice in each conversation); a GAT lets you learn whose opinion to trust. And if you keep asking around for too many rounds, everybody ends up with the opinion of the whole town — over-smoothing.",
      },
      {
        t: "p",
        text: "The key property is that the computation does not care how the nodes are numbered. Renumbering the nodes permutes the rows of the feature matrix and the rows and columns of the adjacency matrix, and the GNN's output is permuted in exactly the same way (it is **permutation-equivariant**). That is what makes it a sensible model for sets of related objects, where an ordering would be arbitrary.",
      },
      {
        t: "p",
        text: "A second intuition: most graphs that GNNs work well on are **homophilous** — connected nodes tend to have similar labels. Averaging over neighbours then acts as denoising: a node with misleading features is corrected by its neighbours. In the From-Scratch experiment, an MLP that ignores the edges reaches 40% accuracy on noisy features while a two-layer GCN using the same features plus the graph reaches 95%.",
      },
    ],
    mechanism: [
      { t: "h", text: "Representing a graph" },
      {
        t: "list",
        items: [
          String.raw`**Adjacency matrix** $A \in \{0,1\}^{n\times n}$: $A_{ij} = 1$ if there is an edge between $i$ and $j$. For an undirected graph $A$ is symmetric; weighted graphs store edge weights instead of 1.`,
          String.raw`**Degree** $d_i = \sum_j A_{ij}$, the number of neighbours; the **degree matrix** $D = \mathrm{diag}(d_1,\dots,d_n)$.`,
          String.raw`**Node features** $X \in \mathbb R^{n\times f}$, one row per node (atom type, user profile, bag-of-words of a paper…). Optional edge features $e_{ij}$.`,
          "**Edge list** (`edge_index`): a $2\\times E$ integer array of (sender, receiver) pairs, each undirected edge stored in both directions. Real graphs are sparse ($E \\ll n^2$), so libraries store edge lists rather than dense $A$.",
        ],
      },
      { t: "h", text: "One message-passing layer" },
      {
        t: "steps",
        items: [
          { title: "Message", text: "For every edge $(j \\to i)$ compute a message from the sender's (and optionally the receiver's and the edge's) features. In a GCN the message is just the transformed sender vector scaled by a constant.", tex: String.raw`m_{j\to i} = \phi\big(h_i, h_j, e_{ij}\big)` },
          { title: "Aggregate", text: "Combine all incoming messages with a **permutation-invariant** operation, so the result does not depend on the order of the neighbours: sum, mean, max, or an attention-weighted sum.", tex: String.raw`a_i = \bigoplus_{j\in\mathcal N(i)} m_{j\to i}` },
          { title: "Update", text: "Combine the aggregate with the node's own vector, usually through a linear map and a non-linearity (or an MLP / GRU). GCN and GAT include the node itself via a self-loop instead of a separate term.", tex: String.raw`h_i' = \psi\big(h_i, a_i\big)` },
          { title: "Stack k layers", text: "After $k$ layers $h_i^{(k)}$ depends on the $k$-hop neighbourhood of $i$ — the receptive field grows by one hop per layer. Most models use 2–4 layers." },
          { title: "Task head", text: "**Node-level**: a linear layer + softmax on each $h_i^{(k)}$ (classify papers, users). **Edge-level**: score a pair, e.g. $\\sigma(h_i^\\top h_j)$ for link prediction (recommendation, knowledge-graph completion). **Graph-level**: a permutation-invariant **readout** such as the mean, sum or max of all node vectors, followed by an MLP (predict a molecule's solubility)." },
        ],
      },
      { t: "h", text: "The GCN layer, operation by operation" },
      {
        t: "steps",
        items: [
          { title: "Add self-loops", text: "Without them a node's new vector would ignore its own old vector.", tex: String.raw`\tilde A = A + I,\qquad \tilde d_i = d_i + 1` },
          { title: "Normalise symmetrically", text: "Divide each entry by the geometric mean of the two degrees, so high-degree nodes neither dominate nor explode the scale of their neighbours.", tex: String.raw`\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2},\qquad \hat A_{ij} = \frac{\tilde A_{ij}}{\sqrt{\tilde d_i\tilde d_j}}` },
          { title: "Transform", text: "Multiply every node's vector by the shared weight matrix (the same for all nodes).", tex: String.raw`Z = H W \in \mathbb R^{n\times f'}` },
          { title: "Aggregate", text: "Row $i$ of $\\hat A Z$ is the normalised sum over $i$'s neighbours and itself.", tex: String.raw`(\hat A Z)_i = \sum_{j\in\mathcal N(i)\cup\{i\}} \frac{z_j}{\sqrt{\tilde d_i\tilde d_j}}` },
          { title: "Non-linearity", text: "ReLU (hidden layers) or nothing (the output layer, followed by softmax cross-entropy).", tex: String.raw`H' = \mathrm{ReLU}(\hat A Z)` },
        ],
      },
      { t: "h", text: "The GAT layer, operation by operation" },
      {
        t: "steps",
        items: [
          { title: "Transform", tex: String.raw`z_i = W h_i` },
          { title: "Score every edge (including self-loops)", text: "A learned vector $a = [a_l; a_r]$ scores the concatenation of receiver and sender; LeakyReLU uses slope 0.2 for negative inputs.", tex: String.raw`e_{ij} = \mathrm{LeakyReLU}\big(a^\top [z_i \,\|\, z_j]\big) = \mathrm{LeakyReLU}\big(a_l^\top z_i + a_r^\top z_j\big)` },
          { title: "Normalise over the neighbourhood", text: "A softmax over $j \\in \\mathcal N(i) \\cup \\{i\\}$ only — non-neighbours get weight 0. This is attention with the graph as a mask.", tex: String.raw`\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal N(i)\cup\{i\}}\exp(e_{ik})}` },
          { title: "Weighted sum", tex: String.raw`h_i' = \sigma\Big(\sum_{j\in\mathcal N(i)\cup\{i\}}\alpha_{ij} z_j\Big)` },
          { title: "Multiple heads", text: "Run $K$ independent heads with their own $W^k, a^k$ and concatenate their outputs (hidden layers) or average them (output layer), exactly like multi-head attention." },
        ],
      },
    ],
    math: [
      { t: "h", text: "1. Notation" },
      {
        t: "p",
        text: String.raw`$n$ nodes, adjacency $A \in \mathbb R^{n\times n}$, features $H^{(l)} \in \mathbb R^{n\times f_l}$ at layer $l$ (with $H^{(0)} = X$), weights $W^{(l)} \in \mathbb R^{f_l\times f_{l+1}}$, element-wise non-linearity $\sigma$, $\mathcal N(i)$ the set of neighbours of node $i$, $\tilde A = A + I$ and $\tilde d_i = \sum_j \tilde A_{ij}$.`,
      },
      { t: "h", text: "2. Why symmetric normalisation" },
      {
        t: "p",
        text: String.raw`Plain summation $\tilde A H$ multiplies the scale of a node's vector by roughly its degree, and repeating it makes high-degree nodes explode. Mean aggregation $\tilde D^{-1}\tilde A H$ (row-normalised, a random-walk matrix) fixes the scale but treats a message from a hub the same as one from a leaf. Kipf and Welling use $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$: it is symmetric, its eigenvalues lie in $(-1, 1]$, and a message from $j$ to $i$ is down-weighted by both degrees. They obtained it as a first-order approximation of spectral graph convolutions (filters defined through the eigenvectors of the graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$), with the self-loop "renormalisation trick" added to keep the eigenvalues bounded for stable stacking.`,
      },
      { t: "math", tex: String.raw`H^{(l+1)} = \sigma\big(\hat A H^{(l)} W^{(l)}\big),\qquad h_i^{(l+1)} = \sigma\Big(\sum_{j\in\mathcal N(i)\cup\{i\}} \frac{1}{\sqrt{\tilde d_i\tilde d_j}}\, W^{(l)\top} h_j^{(l)}\Big)` },
      { t: "h", text: "3. Worked example: one GCN layer" },
      {
        t: "p",
        text: String.raw`Graph with 4 nodes and edges 0–1, 0–2, 0–3, 2–3. Features $X$ (one row per node) and weights $W$:`,
      },
      { t: "math", tex: String.raw`X = \begin{bmatrix}1&0\\0&1\\1&1\\2&0\end{bmatrix},\quad W = \begin{bmatrix}0.5&-1\\1&0.5\end{bmatrix},\quad XW = \begin{bmatrix}0.5&-1\\1&0.5\\1.5&-0.5\\1&-2\end{bmatrix}` },
      {
        t: "steps",
        items: [
          { title: "Degrees with self-loops", tex: String.raw`\tilde d = (3+1,\ 1+1,\ 2+1,\ 2+1) = (4, 2, 3, 3)` },
          { title: "Normalised weights of node 0's neighbourhood", tex: String.raw`\hat A_{00} = \tfrac{1}{\sqrt{4\cdot4}} = 0.25,\quad \hat A_{01} = \tfrac{1}{\sqrt{4\cdot 2}} = 0.353553,\quad \hat A_{02} = \hat A_{03} = \tfrac{1}{\sqrt{4\cdot3}} = 0.288675` },
          { title: "Aggregate, first column", tex: String.raw`0.25(0.5) + 0.353553(1) + 0.288675(1.5) + 0.288675(1) = 0.125 + 0.353553 + 0.433013 + 0.288675 = 1.200241` },
          { title: "Aggregate, second column", tex: String.raw`0.25(-1) + 0.353553(0.5) + 0.288675(-0.5) + 0.288675(-2) = -0.25 + 0.176777 - 0.144338 - 0.577350 = -0.794911` },
          { title: "ReLU", tex: String.raw`h_0' = \mathrm{ReLU}(1.200241,\ -0.794911) = (1.200241,\ 0)` },
        ],
      },
      { t: "p", text: "The full matrix $\\hat A X W$ printed by the From-Scratch program has first row (1.2002, −0.7949), matching. Nodes 2 and 3 get identical outputs (0.9777, −1.122): they have the same neighbourhood {0, 2, 3} and the same degree, so a GCN cannot tell them apart even though their input features differ — an example of the information a degree-weighted average loses." },
      { t: "h", text: "4. Worked example: GAT attention for node 0" },
      {
        t: "p",
        text: String.raw`Same graph and $W$, so $z_j$ are the rows of $XW$. Take $a_l = (1, 0)$ and $a_r = (0.5, 1)$. Then $a_l^\top z_0 = 0.5$ and $a_r^\top z_j = 0.5 z_{j1} + z_{j2}$ is $-0.75, 1.0, 0.25, -1.5$ for $j = 0,1,2,3$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Raw scores", tex: String.raw`e_{0j} = \mathrm{LeakyReLU}(0.5 + a_r^\top z_j) = \mathrm{LeakyReLU}(-0.25,\ 1.5,\ 0.75,\ -1.0) = (-0.05,\ 1.5,\ 0.75,\ -0.2)` },
          { title: "Exponentials", tex: String.raw`(e^{-0.05}, e^{1.5}, e^{0.75}, e^{-0.2}) = (0.951229,\ 4.481689,\ 2.117000,\ 0.818731),\quad \text{sum} = 8.368649` },
          { title: "Attention weights", tex: String.raw`\alpha_{0\cdot} = (0.113666,\ 0.535533,\ 0.252968,\ 0.097833)` },
          { title: "Output (no non-linearity)", tex: String.raw`h_0' = \sum_j \alpha_{0j} z_j = (1.069651,\ -0.168049)` },
        ],
      },
      {
        t: "p",
        text: String.raw`Node 0 now gives more than half of its attention to node 1, whereas GCN gave node 1 a weight of 0.354 — fixed by the degrees, independent of features. Note one property of this scoring function: because LeakyReLU is monotonic, the ranking of neighbours by $e_{ij}$ is decided by $a_r^\top z_j$ alone and is the same for every receiving node $i$. Brody, Alon and Yahav (2022, "How Attentive are Graph Attention Networks?") call this **static attention** and propose GATv2, which applies the non-linearity before the dot product with $a$: $e_{ij} = a^\top\mathrm{LeakyReLU}(W[h_i \,\|\, h_j])$.`,
      },
      { t: "h", text: "5. Backpropagation through a two-layer GCN" },
      {
        t: "p",
        text: String.raw`Model: $Z_1 = \hat A X W_1$, $H_1 = \mathrm{ReLU}(Z_1)$, logits $O = \hat A H_1 W_2$, $P = \mathrm{softmax}(O)$ row-wise, and cross-entropy averaged over the set $\mathcal L$ of labelled nodes only (semi-supervised: all nodes take part in the forward pass, but only a few have labels). With $Y$ the one-hot labels:`,
      },
      { t: "math", tex: String.raw`\frac{\partial \mathcal J}{\partial O} = \frac{1}{|\mathcal L|}M\odot(P - Y),\quad \frac{\partial \mathcal J}{\partial W_2} = (\hat A H_1)^\top\frac{\partial \mathcal J}{\partial O},\quad \frac{\partial \mathcal J}{\partial H_1} = \hat A^\top\frac{\partial \mathcal J}{\partial O}W_2^\top,\quad \frac{\partial \mathcal J}{\partial W_1} = (\hat A X)^\top\Big(\frac{\partial \mathcal J}{\partial H_1}\odot\mathbb 1[Z_1 > 0]\Big)`, caption: "M masks out unlabelled rows. Compared with an MLP, the only change is the extra multiplication by Â (forward) and Âᵀ (backward): gradient from a labelled node flows to every node within two hops." },
      { t: "p", text: "The From-Scratch program checks $\\partial\\mathcal J/\\partial (W_1)_{00}$ against a central difference: both are −0.10957188." },
      { t: "h", text: "6. Over-smoothing, exactly" },
      {
        t: "p",
        text: String.raw`Ignore weights and non-linearities and apply the propagation $k$ times: $H^{(k)} = \hat A^k X$. $\hat A$ is symmetric with eigen-decomposition $\hat A = \sum_m \lambda_m u_m u_m^\top$, and for a connected graph the largest eigenvalue $\lambda_1 = 1$ is simple with eigenvector $u_1 \propto \tilde D^{1/2}\mathbf 1$ (check: $\hat A\,\tilde D^{1/2}\mathbf 1 = \tilde D^{-1/2}\tilde A\mathbf 1 = \tilde D^{-1/2}\tilde d = \tilde D^{1/2}\mathbf 1$). The self-loops keep all other eigenvalues strictly inside $(-1, 1)$, so`,
      },
      { t: "math", tex: String.raw`\hat A^k X = u_1 u_1^\top X + \sum_{m\ge2}\lambda_m^k\, u_m u_m^\top X \ \xrightarrow{k\to\infty}\ u_1 (u_1^\top X),\qquad (u_1)_i = \sqrt{\tilde d_i \,/\, \textstyle\sum_j \tilde d_j}` },
      {
        t: "p",
        text: String.raw`Every node's feature vector converges to the same vector $u_1^\top X$ scaled by $\sqrt{\tilde d_i}$: only the degree survives, and the rate is set by the second-largest $|\lambda_m|$. The distance from this limit is measured by the **normalised Dirichlet energy**`,
      },
      { t: "math", tex: String.raw`E(H) = \operatorname{tr}\big(H^\top(I - \hat A)H\big) = \sum_{(i,j)\in\mathcal E}\Big\|\frac{h_i}{\sqrt{\tilde d_i}} - \frac{h_j}{\sqrt{\tilde d_j}}\Big\|^2`, caption: "Sum over undirected edges; it is 0 exactly when h_i ∝ √d̃_i on every connected component." },
      {
        t: "p",
        text: "In the From-Scratch program (120-node graph, 3 communities) the energy falls from $3.3\\times10^3$ to $10^{-3}$ after 64 propagations and $2.5\\times10^{-15}$ after 256, and a nearest-centroid classifier on the propagated features goes 0.568 → 0.955 (k = 8) → 0.892 (k = 64) → 0.351 (k = 256), barely above chance (1/3). A few propagations denoise; many destroy the information. Li, Han and Wu (2018, \"Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning\") identified this Laplacian-smoothing view of GCN and its over-smoothing problem.",
      },
      { t: "h", text: "7. Expressive power" },
      {
        t: "p",
        text: "A message-passing GNN computes a node's vector from the multiset of its neighbours' vectors, recursively. Two nodes whose $k$-hop neighbourhoods unfold into the same tree (with the same features) receive the same vector. Xu et al. (2019, \"How Powerful are Graph Neural Networks?\") showed that message-passing GNNs are at most as powerful as the 1-dimensional Weisfeiler–Lehman graph-isomorphism test, and that sum aggregation with an MLP update (their GIN) reaches that bound, while mean and max aggregation lose information (mean cannot count how many neighbours share a feature). The worked example shows the mean-like GCN merging nodes 2 and 3.",
      },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator propagates **one scalar feature per node** through up to 32 layers with $W = 1$ and no non-linearity, so you see the aggregation alone. Every number is computed live: GCN uses $\\hat A$, GAT recomputes its attention weights from the current features at every layer.",
      },
      { t: "sim", id: "gnn", caption: "Message passing on three small graphs: GCN's normalised weights or GAT's attention on the edges, the aggregation sum for the selected node, and the energy curve that reveals over-smoothing." },
      {
        t: "list",
        items: [
          "**Two communities, GCN.** Step once. Click node 2 (input −0.3, a noisy member of the positive community): its new value is the sum shown in the side panel, $0.333\\times(-0.3)$ plus contributions from nodes 0, 1 and 3, giving +0.685. Node 7 is corrected in the same way. This is the denoising that makes shallow GNNs work on homophilous graphs.",
          "Keep pressing Step and watch the energy curve fall by orders of magnitude. The single bridge edge slows the mixing between the communities: up to layer 32 the sign still separates them, but by layer 40 it no longer does, and by layer 64 every value is within 0.003 of the limit $c\\sqrt{\\tilde d_i}$ (about −0.02 for every node; the side panel reports the largest distance). The two communities are no longer distinguishable.",
          "**Path (one-hot).** After $k$ layers exactly nodes $0\\dots k$ are non-zero: the receptive field is the $k$-hop neighbourhood. A 2-layer GNN cannot know anything about a node 3 hops away.",
          "**Star.** Read the GCN edge labels: every hub–leaf edge has weight $1/\\sqrt{7\\cdot2} = 0.27$, the hub's self-loop $1/7 = 0.14$ and each leaf's self-loop $1/2$. Over-smoothing is fastest here: the energy reaches $10^{-8}$ within a dozen layers.",
          "**Switch to GAT** and select node 3 on the communities graph. The labels are now $\\alpha_{3j}$; they sum to 1 and change from layer to layer. Move $a_r$ (neighbour weight) from +1.5 to −1.5: node 3 shifts its attention from high-valued to low-valued neighbours. Moving $a_l$ changes the weights less — and never changes which neighbour ranks first (static attention).",
          "With any $a_l, a_r$, GAT also over-smooths: its rows are convex combinations, so repeated layers pull all nodes towards one value. Attention changes the speed, not the destination.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "Three parts in one NumPy program: (1) one GCN and one GAT layer on the 4-node graph of the worked examples; (2) a two-layer GCN trained by full-batch gradient descent with hand-derived gradients on a synthetic **stochastic block model** graph (three communities, edges much more likely inside a community), with noisy features and only 3 labelled nodes per class — the semi-supervised setting of Kipf and Welling — compared against the same network with $\\hat A$ replaced by the identity (an MLP); (3) over-smoothing measured as the number of propagation steps grows.",
      },
      { t: "code", lang: "python", title: "gnn_from_scratch.py", code: GNN_NUMPY, output: GNN_NUMPY_OUT },
      {
        t: "list",
        items: [
          "Part 1 reproduces the worked examples: $\\hat A$, the GCN output (rows 2 and 3 identical) and the GAT attention matrix, whose rows sum to 1 and whose zeros are exactly the non-edges.",
          "`gat_layer` scores all $n^2$ pairs and masks non-edges with $-\\infty$ before the softmax — fine for 4 nodes, quadratic for real graphs; the PyTorch version scores only existing edges.",
          "Part 2: with only 9 labels and features whose noise (std 2) swamps the class signal, the MLP reaches 0.396 test accuracy, the GCN 0.946. The analytic and numerical gradients agree to 8 decimals.",
          "Part 3: accuracy of the propagated features peaks around $k = 8$ and collapses to 0.351 at $k = 256$ as the energy goes to zero. The optimum depth depends on the graph: this graph has few edges between communities, so the second eigenvalue is close to 1 and smoothing is slow.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "Libraries such as PyTorch Geometric and DGL provide optimised GCN/GAT layers, but both layers are short in plain PyTorch. The program stores the graph as an edge list and implements message passing with gathers (`z[s]`) and scatter-adds (`index_add_`) — the same pattern those libraries use internally. The GAT softmax is computed per receiving node with a scatter-max for numerical stability. It compares an MLP, a GCN and a 4-head GAT on a 300-node block-model graph with 5 labels per class, prints the learned attention of one node, and trains plain GCNs of increasing depth.",
              },
              { t: "code", lang: "python", title: "gcn_gat_pytorch.py", code: GNN_TORCH, output: GNN_TORCH_OUT },
              {
                t: "list",
                items: [
                  "Using the graph raises test accuracy from 0.500 (MLP) to 0.973 (GCN) and 0.964 (GAT) with about the same number of parameters. GAT is not automatically better: on a homophilous graph with uninformative edge differences, degree-based averaging is already the right inductive bias.",
                  "The attention of node 8 is not an explanation of the prediction: head 0 puts its largest weight (0.223) on a neighbour from *another* class. Attention weights show what the layer averages, not why the final output is what it is.",
                  "Depth: 2 layers give 0.991, 8 layers 0.936, 16 layers 0.232 (below the 0.25 chance level of 4 classes). The deep failure mixes over-smoothing with ordinary optimisation difficulty of a deep plain network; residual connections, normalisation (e.g. PairNorm) or decoupling propagation from transformation (SGC, APPNP) are the usual remedies.",
                  "Transform-then-aggregate (`self.lin(h)` before the scatter) is cheaper than aggregate-then-transform when $f_{out} < f_{in}$, and gives the same result because both are linear.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "A dense-adjacency GCN layer as a custom Keras layer and a `GradientTape` training loop on the labelled nodes. TensorFlow-GNN and Spektral provide sparse, production-grade versions." },
              { t: "code", lang: "python", title: "gcn_keras.py (not executed in CI)", code: GCN_KERAS, output: "Prints the epoch number and the training cross-entropy on the labelled nodes every 50 epochs." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Model", "Aggregation", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Spectral CNNs (Bruna et al. 2014), ChebNet (Defferrard et al. 2016)", "filters on Laplacian eigenvalues / Chebyshev polynomials of $L$", "principled signal-processing view; K-hop filters", "tied to one graph's spectrum (spectral CNN); costly eigendecomposition", "historical basis of GCN"],
          ["GCN (Kipf & Welling 2017)", "$\\sum_j z_j/\\sqrt{\\tilde d_i\\tilde d_j}$", "simple, fast, strong baseline on homophilous graphs", "fixed weights; over-smoothing; weak on heterophily", "semi-supervised node classification"],
          ["GraphSAGE (Hamilton, Ying & Leskovec 2017)", "mean / max-pool / LSTM over a **sampled** neighbourhood, concatenated with self", "inductive (new nodes at test time); mini-batch training on huge graphs", "sampling variance", "large-scale recommendation"],
          ["GAT (Veličković et al. 2018)", "softmax attention over neighbours, multi-head", "learned, feature-dependent weights; handles varying neighbour relevance", "static attention; more memory per edge", "citation, protein–protein graphs"],
          ["GATv2 (Brody et al. 2022)", "$a^\\top\\mathrm{LeakyReLU}(W[h_i\\|h_j])$", "dynamic attention (ranking can depend on $i$)", "slightly more compute", "drop-in GAT replacement"],
          ["GIN (Xu et al. 2019)", "$\\mathrm{MLP}((1+\\epsilon)h_i + \\sum_j h_j)$", "as expressive as 1-WL; strong for graph classification", "sum can be sensitive to degree scale", "molecules, graph-level tasks"],
          ["MPNN (Gilmer et al. 2017)", "learned edge-conditioned messages, GRU update", "uses edge features (bond types, distances)", "heavier", "quantum chemistry"],
          ["Graph Transformers", "attention over all node pairs + structural encodings", "no over-squashing through bottlenecks; long range", "$O(n^2)$; needs positional/structural encodings", "molecular benchmarks, long-range tasks"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Chemistry and drug discovery** — atoms as nodes, bonds as edges; message-passing networks predict molecular properties (Gilmer et al. 2017 introduced MPNNs on the QM9 quantum-chemistry data set).",
          "**Recommendation** — users and items as a bipartite graph; Pinterest's PinSage (Ying et al. 2018) is a GraphSAGE-style GCN on a graph of billions of pins and boards.",
          "**Traffic and travel time** — road segments as nodes; GNNs were deployed for estimated-time-of-arrival prediction in Google Maps (Derrow-Pinion et al. 2021).",
          "**Weather forecasting** — GraphCast (Lam et al. 2023) runs message passing on a multi-resolution mesh over the globe.",
          "**Fraud and abuse detection** — accounts, devices and transactions as a graph; suspicious nodes are flagged by their neighbourhoods.",
          "**Knowledge graphs and citation networks** — link prediction (missing facts) and node classification (paper topics).",
          "**Physics simulation, chip design, program analysis** — particles, circuit components or code tokens with explicit relations.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Uses relational structure** that tabular models ignore; one model handles graphs of any size; permutation-equivariant by construction.",
          "**Parameter-efficient**: weights are shared across nodes, so parameter count is independent of graph size (676 parameters for a 300-node graph here).",
          "**Over-smoothing** limits depth: repeated averaging makes node vectors indistinguishable (Section 6), so most GNNs have 2–4 layers and a small receptive field.",
          "**Over-squashing**: information from an exponentially growing neighbourhood must pass through fixed-size vectors and bottleneck edges, so long-range dependencies are hard (Alon & Yahav 2021).",
          "**Homophily assumption**: GCN-style averaging helps when neighbours share labels and can hurt on heterophilous graphs (e.g. fraudsters connecting to normal users).",
          "**Limited expressiveness**: message passing cannot distinguish some non-isomorphic graphs (bounded by 1-WL).",
          "**Scalability**: full-batch training needs the whole graph in memory; large graphs need neighbour sampling (GraphSAGE), cluster-based batching or precomputed propagation (SGC). Hub nodes with huge degrees make memory uneven.",
          "**Transductive vs inductive**: a model trained on one fixed graph with spectral or node-ID features may not transfer to new nodes; feature-based message passing does.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting self-loops.** Without $+I$ a node's own features never reach its next representation (and for a node with no neighbours the output is all zeros). Add them before computing degrees.",
          "**Normalising with the degrees of $A$ instead of $\\tilde A$**, or using $D^{-1/2}$ on a graph with isolated nodes (division by zero). Compute $\\tilde d = d + 1$.",
          "**Storing an undirected edge only once in an edge list.** Messages then flow in one direction only; add both $(i, j)$ and $(j, i)$.",
          "**Leaking labels in semi-supervised training**: the loss must be computed on the training nodes only, and model selection on validation nodes — but the forward pass legitimately uses all nodes and edges (transductive setting). Report which setting you use.",
          "**Making the GNN deep 'because deeper is better'.** Plain GCNs degrade beyond a few layers (16 layers: 0.232 here). Increase the receptive field with residual connections, jumping knowledge, or decoupled propagation instead.",
          "**Using a non-permutation-invariant aggregation** (e.g. concatenating neighbours in index order). The output then depends on node numbering; use sum/mean/max/attention.",
          "**Graph-level readout with mean when counts matter.** Mean pooling cannot tell a graph from two copies of itself; use sum (or several readouts) when size or counts are informative.",
          "**Reading GAT attention weights as explanations.** They describe one layer's averaging, can be spread over heads and layers, and in our run the largest weight went to a different-class neighbour.",
          "**Random splits on graphs with leakage**: in link prediction, test edges must be removed from the message-passing graph during training, or the model sees the answer.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why can't you apply a CNN or an MLP directly to a graph?", a: "A CNN needs a regular grid with a fixed neighbourhood shape and ordering; graph nodes have varying numbers of neighbours and no canonical order. An MLP on the flattened adjacency matrix would depend on node numbering, have a fixed input size and not share weights across nodes. A GNN aggregates over neighbours with permutation-invariant operations and shares weights across nodes, so it handles any size and is equivariant to renumbering." },
      { t: "qa", level: "beginner", q: "Describe the message-passing framework.", a: "Each layer: (1) every node computes a message for each of its edges from its features (and possibly edge features); (2) each node aggregates incoming messages with a permutation-invariant function (sum, mean, max, attention); (3) each node updates its vector from its old vector and the aggregate with a learned function. Stacking k layers gives every node information from its k-hop neighbourhood." },
      { t: "qa", level: "beginner", q: "What are node-, edge- and graph-level tasks? Give an example of each.", a: "Node-level: predict a label per node (topic of a paper in a citation graph) — apply a classifier to each final node vector. Edge-level: predict a property of a pair (will this user buy this item? link prediction) — score pairs of node vectors, e.g. a dot product. Graph-level: predict a property of the whole graph (toxicity of a molecule) — pool all node vectors with a permutation-invariant readout, then classify." },
      { t: "qa", level: "intermediate", q: "Write the GCN layer and explain each part of the normalisation.", a: "$H' = \\sigma(\\tilde D^{-1/2}(A + I)\\tilde D^{-1/2} H W)$. $A + I$ adds self-loops so a node keeps its own information. $\\tilde D$ is the degree matrix of $A + I$. Dividing entry $(i,j)$ by $\\sqrt{\\tilde d_i\\tilde d_j}$ keeps the scale bounded (eigenvalues in $(-1, 1]$), so repeated layers neither explode nor vanish, and down-weights messages from and to high-degree nodes. $W$ is shared by all nodes and $\\sigma$ is ReLU." },
      { t: "qa", level: "intermediate", q: "How does GAT differ from GCN?", a: "GCN's aggregation weights $1/\\sqrt{\\tilde d_i\\tilde d_j}$ are fixed by the graph structure. GAT computes them from features: $e_{ij} = \\mathrm{LeakyReLU}(a^\\top[Wh_i\\|Wh_j])$, softmax over the neighbourhood. Different neighbours can get different importance, the weights adapt to features, and multiple heads are used. Cost: per-edge attention computation and memory; it is not guaranteed to outperform GCN on homophilous graphs (0.964 vs 0.973 in our run)." },
      { t: "qa", level: "intermediate", q: "What is over-smoothing and why does it happen?", a: "As layers are stacked, node representations converge to (almost) the same vector, so nodes become indistinguishable and accuracy drops. Each layer is a low-pass filter: $\\hat A^k X \\to u_1 u_1^\\top X$ because all eigenvalues of $\\hat A$ except the top one (= 1, eigenvector $\\propto\\sqrt{\\tilde d}$) have magnitude below 1. The Dirichlet energy $\\mathrm{tr}(H^\\top(I - \\hat A)H)$ decays geometrically. Remedies: few layers, residual/skip connections, jumping knowledge, normalisation such as PairNorm, DropEdge, or decoupling propagation from transformation (APPNP's personalised PageRank keeps a fraction of the input at every step)." },
      { t: "qa", level: "intermediate", q: "Compute the GCN weight on an edge between a node of degree 3 and a node of degree 1.", a: "With self-loops the degrees are 4 and 2, so $\\hat A_{ij} = 1/\\sqrt{4\\cdot2} = 0.3536$. The self-loop weights are $1/4$ and $1/2$." },
      { t: "qa", level: "advanced", q: "How expressive are message-passing GNNs?", a: "At most as powerful as the 1-dimensional Weisfeiler–Lehman test (Xu et al. 2019): two nodes with identical unfolded neighbourhood trees get identical representations, so e.g. a 6-cycle and two disjoint triangles (all nodes of degree 2, identical features) cannot be distinguished. Sum aggregation with injective MLP updates (GIN) achieves 1-WL power; mean and max lose multiset information. Going beyond needs higher-order GNNs, subgraph or positional/structural encodings (random-walk or Laplacian eigenvectors), or random node features." },
      { t: "qa", level: "advanced", q: "How would you train a GNN on a graph with a billion edges?", a: "Full-batch $\\hat A H W$ does not fit in memory. Options: neighbour sampling (GraphSAGE / PinSage) to build mini-batches of target nodes with a sampled k-hop computation graph; cluster-based batching (Cluster-GCN) that trains on dense subgraphs; precomputing $\\hat A^k X$ once (SGC, SIGN) and training an MLP on it; distributed sparse matrix multiplication with graph partitioning. Watch for exploding neighbourhood sizes with depth and for hubs; cap fan-out per layer." },
      { t: "qa", level: "advanced", q: "Show that $u \\propto \\tilde D^{1/2}\\mathbf 1$ is an eigenvector of $\\hat A$ with eigenvalue 1 and explain its role.", a: "$\\hat A\\tilde D^{1/2}\\mathbf 1 = \\tilde D^{-1/2}\\tilde A\\tilde D^{-1/2}\\tilde D^{1/2}\\mathbf 1 = \\tilde D^{-1/2}\\tilde A\\mathbf 1 = \\tilde D^{-1/2}\\tilde d = \\tilde D^{1/2}\\mathbf 1$. By Perron–Frobenius (connected graph, non-negative matrix) this is the top eigenvector; self-loops make $\\hat A$ aperiodic so all other eigenvalues are strictly inside the unit circle. Repeated propagation therefore projects features onto $u$: the over-smoothing limit $h_i \\propto \\sqrt{\\tilde d_i}$, carrying no information except degree." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which operation must a GNN's neighbour aggregation have?", options: ["Differentiability with respect to node ids", "Permutation invariance over the neighbours", "A fixed number of neighbours", "Sorting neighbours by degree"], answer: 1, explain: "Neighbours have no canonical order; sum, mean, max and attention-weighted sums give the same result for any ordering." },
          { kind: "numeric", q: "In a GCN, node $i$ has 5 neighbours and node $j$ has 2. What is $\\hat A_{ij}$ (3 decimals)?", answer: 0.236, tolerance: 0.001, explain: "$\\tilde d_i = 6$, $\\tilde d_j = 3$: $1/\\sqrt{18} = 0.2357$." },
          { kind: "numeric", q: "A node has degree 3 (without self-loop). What is the GCN weight on its own self-loop, $\\hat A_{ii}$?", answer: 0.25, tolerance: 0.001, explain: "$\\tilde d_i = 4$, so $1/\\sqrt{4\\cdot4} = 1/4$." },
          { q: "After $k$ message-passing layers, a node's representation depends on…", options: ["All nodes in the graph", "Its $k$-hop neighbourhood", "Only its immediate neighbours", "Nodes with the same degree"], answer: 1, explain: "Each layer extends the receptive field by one hop (see the path graph in the simulator)." },
          { kind: "numeric", q: "GAT scores of node $i$ over its neighbourhood (self, j, k) are $e = (0, \\ln 3, 0)$. What is $\\alpha_{ij}$?", answer: 0.6, tolerance: 0.001, explain: "$\\exp(e) = (1, 3, 1)$; $3/5 = 0.6$." },
          { q: "Repeated GCN propagation $\\hat A^k X$ on a connected graph converges to node vectors proportional to…", options: ["The input features", "$\\sqrt{\\tilde d_i}$ times one common vector", "Zero for every node", "One-hot community indicators"], answer: 1, explain: "The top eigenvector of $\\hat A$ is $\\propto\\tilde D^{1/2}\\mathbf 1$; everything else decays as $\\lambda_m^k$." },
          { q: "In semi-supervised node classification with a GCN, which nodes take part in the forward pass?", options: ["Only labelled nodes", "All nodes; the loss uses only labelled nodes", "Only test nodes", "A random half each epoch"], answer: 1, explain: "Unlabelled nodes still pass messages; this is how a few labels propagate information through the graph." },
          { q: "Which readout lets a graph-level model distinguish a graph from two disjoint copies of itself?", options: ["Mean pooling", "Max pooling", "Sum pooling", "None of them"], answer: 2, explain: "Mean and max are unchanged by duplication; the sum doubles." },
        ],
      },
      {
        t: "exercise",
        title: "A GCN layer by hand",
        prompt: "A path graph 0–1–2 has scalar features $x = (3, 0, -3)$ and $W = 1$, no non-linearity. (a) Write $\\tilde d$ and $\\hat A$. (b) Compute $\\hat A x$. (c) Compute the normalised Dirichlet energy before and after the layer.",
        hint: "$\\tilde d = (2, 3, 2)$. The energy is $\\sum_{(i,j)}(h_i/\\sqrt{\\tilde d_i} - h_j/\\sqrt{\\tilde d_j})^2$ over the two edges.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) normalised adjacency", tex: String.raw`\hat A = \begin{bmatrix}1/2 & 1/\sqrt6 & 0\\ 1/\sqrt6 & 1/3 & 1/\sqrt6\\ 0 & 1/\sqrt6 & 1/2\end{bmatrix} = \begin{bmatrix}0.5 & 0.408248 & 0\\ 0.408248 & 0.333333 & 0.408248\\ 0 & 0.408248 & 0.5\end{bmatrix}` },
              { title: "(b) one layer", tex: String.raw`\hat A x = (0.5\cdot3,\ 0.408248\cdot3 - 0.408248\cdot3,\ -0.5\cdot3) = (1.5,\ 0,\ -1.5)` },
              { title: "(c) energy before", tex: String.raw`\big(3/\sqrt2 - 0\big)^2 + \big(0 + 3/\sqrt2\big)^2 = 4.5 + 4.5 = 9` },
              { title: "(c) energy after", tex: String.raw`\big(1.5/\sqrt2\big)^2 + \big(1.5/\sqrt2\big)^2 = 1.125 + 1.125 = 2.25` },
            ],
          },
          { t: "p", text: "The layer exactly halves the features here, so the energy drops by a factor of 4. ($x$ happens to be an eigenvector of $\\hat A$ with eigenvalue 1/2; the eigenvector with eigenvalue 1 is $\\propto(\\sqrt2, \\sqrt3, \\sqrt2)$ and $x$ is orthogonal to it, so repeated layers send it to 0.)" },
        ],
      },
      {
        t: "exercise",
        title: "Coding: graph classification with a readout",
        prompt: "Using the `GCNLayer` from the PyTorch program, build a graph classifier: generate 200 random graphs of 20 nodes, half from a two-community block model ($p_{in} = 0.3$, $p_{out} = 0.02$) and half Erdős–Rényi graphs with the same expected number of edges, with constant node features (all ones). Apply two GCN layers, a sum readout and a linear layer. Train with cross-entropy and report test accuracy. Then replace the sum by a mean readout and compare.",
        hint: "Batch several graphs by placing them in one big disconnected graph (offset each graph's node indices) and keep a `batch` vector of graph ids; the readout is `torch.zeros(B, F).index_add_(0, batch, h)`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def make_graph(two_communities, rng, n=20):\n    y = np.repeat([0, 1], n // 2)\n    if two_communities:\n        p = np.where(y[:, None] == y[None, :], 0.3, 0.02)\n    else:\n        p = np.full((n, n), (0.3 * 90 + 0.02 * 100) / 190)   # same expected edge count\n    up = np.triu(rng.random((n, n)) < p, 1)\n    s, d = np.nonzero(up | up.T)\n    return torch.tensor(np.stack([s, d]))\n\n# batching: concatenate edge lists with node offsets g*20, batch = repeat(arange(B), 20)\nclass GraphNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.l1, self.l2, self.out = GCNLayer(1, 32), GCNLayer(32, 32), nn.Linear(32, 2)\n    def forward(self, x, ei, batch, B):\n        h = F.relu(self.l2(F.relu(self.l1(x, ei)), ei))\n        g = torch.zeros(B, h.shape[1]).index_add_(0, batch, h)   # sum readout\n        return self.out(g)",
          },
          { t: "p", text: "With constant input features the model can only use structure. Each GCN layer produces degree-dependent values, so after two layers the node vectors encode local degree patterns; the readout turns them into graph statistics. The two classes have the same expected number of edges, so the network must pick up structural differences (the block-model graphs have more triangles inside communities). Expect clearly better than chance but imperfect accuracy; compare sum and mean on the same seed and report both rather than assuming which wins." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Graph = nodes + edges (+ features). Represent with adjacency $A$ (or an edge list), degrees $d_i$, feature matrix $X$.",
          "Message passing: message → permutation-invariant aggregate → update; $k$ layers = $k$-hop receptive field; weights shared over nodes.",
          "GCN: $H' = \\sigma(\\hat A H W)$, $\\hat A = \\tilde D^{-1/2}(A + I)\\tilde D^{-1/2}$, $\\hat A_{ij} = 1/\\sqrt{\\tilde d_i\\tilde d_j}$.",
          "GAT: $\\alpha_{ij} = \\mathrm{softmax}_j\\,\\mathrm{LeakyReLU}(a_l^\\top Wh_i + a_r^\\top Wh_j)$ over $\\mathcal N(i)\\cup\\{i\\}$; multi-head; GATv2 fixes static attention.",
          "Tasks: node (per-node head), edge (pair scoring), graph (readout: sum/mean/max + MLP).",
          "Over-smoothing: $\\hat A^k X \\to u_1u_1^\\top X$, $u_1 \\propto \\sqrt{\\tilde d}$; Dirichlet energy $\\mathrm{tr}(H^\\top(I - \\hat A)H) \\to 0$. Keep GNNs shallow or add residuals.",
          "Results here: GCN 0.946 vs MLP 0.396 (NumPy, 9 labels); GCN 0.973 / GAT 0.964 / MLP 0.500 (PyTorch); 16-layer plain GCN 0.232.",
          "Expressiveness is bounded by the 1-WL test; sum aggregation (GIN) reaches it.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Deep Reinforcement Learning](/learn/m15/deep-rl) — another setting beyond supervised learning; GNN policies are used when states are graphs (molecules, chips, combinatorial problems).",
          "[Multimodal Models, CLIP and RAG](/learn/m15/multimodal) — joint embedding spaces; knowledge graphs and GNN retrievers appear in some retrieval systems.",
          "[The Transformer](/learn/m11/transformer-architecture) — self-attention is message passing on a complete graph; graph Transformers combine the two views.",
        ],
      },
    ],
  },
};

export default lesson;
