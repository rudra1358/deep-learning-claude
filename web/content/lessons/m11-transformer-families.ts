import type { Lesson } from "@/lib/types";

const MASKS_NUMPY = String.raw`import numpy as np

rng = np.random.default_rng(0)
PAD, CLS, SEP, MASK = 0, 1, 2, 3            # special token ids
N_SPECIAL, VOCAB = 4, 1000


def mlm_mask(ids, rng, vocab_size=VOCAB, p_select=0.15, p_mask=0.8, p_random=0.1):
    """BERT masking. Returns (corrupted input ids, labels); label -100 = not predicted."""
    ids = ids.copy()
    labels = np.full_like(ids, -100)
    candidates = ids >= N_SPECIAL                          # never select [PAD]/[CLS]/[SEP]
    selected = candidates & (rng.random(ids.shape) < p_select)
    labels[selected] = ids[selected]                       # the loss is computed ONLY here
    r = rng.random(ids.shape)
    to_mask = selected & (r < p_mask)                                   # 80% -> [MASK]
    to_random = selected & (r >= p_mask) & (r < p_mask + p_random)      # 10% -> random token
    ids[to_mask] = MASK
    ids[to_random] = rng.integers(N_SPECIAL, vocab_size, to_random.sum())    # (10% left unchanged)
    return ids, labels


# --- 1) one readable example with a tiny word vocabulary ---
words = "[PAD] [CLS] [SEP] [MASK] the rabbit took a watch out of its waistcoat pocket and hurried on".split()
sent = np.array([CLS] + list(range(4, len(words))) + [SEP])
ex_rng = np.random.default_rng(0)
inp, lab = mlm_mask(sent, ex_rng, len(words), p_select=0.3)       # higher rate only so the short example shows all cases
print("original :", " ".join(words[i] for i in sent))
print("input    :", " ".join(words[i] for i in inp))
print("labels   :", " ".join(words[l] if l >= 0 else "." for l in lab))

# --- 2) statistics over a large batch: is it really 15% and 80/10/10? ---
batch = rng.integers(N_SPECIAL, VOCAB, (4096, 128))
batch[:, 0], batch[:, -1] = CLS, SEP
inp, lab = mlm_mask(batch, rng)
sel = lab != -100
print(f"\nselected for prediction: {sel[:, 1:-1].mean():.4f} of ordinary tokens (target 0.15)")
print(f"  replaced by [MASK]   : {(inp[sel] == MASK).mean():.4f}  (target 0.80)")
print(f"  replaced by random   : {((inp[sel] != MASK) & (inp[sel] != lab[sel])).mean():.4f}  (target ~0.10)")
print(f"  left unchanged       : {(inp[sel] == lab[sel]).mean():.4f}  (target ~0.10 + random hits)")
print(f"special tokens ever selected: {sel[:, [0, -1]].any()}")


# --- 3) the four attention-mask patterns (1 = query row may attend to key column) ---
def show(name, M):
    print(f"\n{name}  (allowed pairs: {int(M.sum())})")
    for row in M.astype(int):
        print("  " + " ".join(map(str, row)))


n, prefix = 6, 3
bidirectional = np.ones((n, n), bool)                         # BERT / ViT encoder
causal = np.tril(np.ones((n, n), bool))                       # GPT decoder
prefix_lm = causal.copy()
prefix_lm[:, :prefix] = True                                  # every position sees the whole prefix
n_dec, n_enc, enc_pad = 4, 6, 1                               # T5/BART decoder -> encoder
cross = np.ones((n_dec, n_enc), bool)
cross[:, n_enc - enc_pad:] = False                            # only padding is hidden
show("bidirectional (encoder-only)", bidirectional)
show("causal (decoder-only)", causal)
show(f"prefix-LM (prefix of {prefix})", prefix_lm)
show(f"cross-attention: {n_dec} decoder queries x {n_enc} encoder keys, last key = [PAD]", cross)`;

const MASKS_NUMPY_OUT = String.raw`original : [CLS] the rabbit took a watch out of its waistcoat pocket and hurried on [SEP]
input    : [CLS] and [MASK] [MASK] a watch out of its waistcoat pocket and hurried [MASK] [SEP]
labels   : . the rabbit took . . . . . . . and . on .

selected for prediction: 0.1494 of ordinary tokens (target 0.15)
  replaced by [MASK]   : 0.7983  (target 0.80)
  replaced by random   : 0.1028  (target ~0.10)
  left unchanged       : 0.0990  (target ~0.10 + random hits)
special tokens ever selected: False

bidirectional (encoder-only)  (allowed pairs: 36)
  1 1 1 1 1 1
  1 1 1 1 1 1
  1 1 1 1 1 1
  1 1 1 1 1 1
  1 1 1 1 1 1
  1 1 1 1 1 1

causal (decoder-only)  (allowed pairs: 21)
  1 0 0 0 0 0
  1 1 0 0 0 0
  1 1 1 0 0 0
  1 1 1 1 0 0
  1 1 1 1 1 0
  1 1 1 1 1 1

prefix-LM (prefix of 3)  (allowed pairs: 24)
  1 1 1 0 0 0
  1 1 1 0 0 0
  1 1 1 0 0 0
  1 1 1 1 0 0
  1 1 1 1 1 0
  1 1 1 1 1 1

cross-attention: 4 decoder queries x 6 encoder keys, last key = [PAD]  (allowed pairs: 20)
  1 1 1 1 1 0
  1 1 1 1 1 0
  1 1 1 1 1 0
  1 1 1 1 1 0`;

const VISION_NUMPY = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---------------- 1) ViT patch embedding, two ways ----------------
C, H, W, P, D = 3, 32, 32, 8, 16                  # channels, height, width, patch size, embedding dim
img = rng.normal(size=(C, H, W))
W_e = rng.normal(0, 0.1, (C * P * P, D))         # one shared linear map for every patch
b_e = rng.normal(0, 0.1, D)

# (a) "cut into patches, flatten, multiply": (C,H,W) -> (H/P * W/P, C*P*P) -> (N, D)
patches = img.reshape(C, H // P, P, W // P, P).transpose(1, 3, 0, 2, 4).reshape(-1, C * P * P)
tokens_a = patches @ W_e + b_e

# (b) a convolution with D filters of size C x P x P and stride P (explicit loops)
kernels = W_e.T.reshape(D, C, P, P)              # filter d = column d of W_e, reshaped
tokens_b = np.zeros((H // P, W // P, D))
for i in range(H // P):
    for j in range(W // P):
        window = img[:, i * P:(i + 1) * P, j * P:(j + 1) * P]                   # C x P x P
        tokens_b[i, j] = np.tensordot(kernels, window, axes=([1, 2, 3], [0, 1, 2])) + b_e
tokens_b = tokens_b.reshape(-1, D)                # raster order: row of patches by row
print(f"patches {patches.shape} -> tokens {tokens_a.shape}; max |linear - strided conv| = {np.abs(tokens_a - tokens_b).max():.1e}")

# prepend a learnable [CLS] token and add position embeddings
cls = rng.normal(0, 0.02, (1, D))
pos = rng.normal(0, 0.02, (1 + tokens_a.shape[0], D))
seq = np.concatenate([cls, tokens_a]) + pos
print("sequence entering the encoder:", seq.shape, "(1 [CLS] + 16 patches)")
for res in [224, 384]:
    print(f"ViT-B/16 at {res}x{res}: {(res // 16) ** 2} patches + [CLS] = {(res // 16) ** 2 + 1} tokens, "
          f"patch-embedding weights {3 * 16 * 16 * 768 + 768:,}")

# ---------------- 2) Swin: window partition and the cyclic shift ----------------
Hp = Wp = 8; M = 4                                # an 8x8 grid of patch tokens, window size 4
grid = np.arange(Hp * Wp).reshape(Hp, Wp)
win_id = (np.arange(Hp)[:, None] // M) * (Wp // M) + (np.arange(Wp)[None, :] // M)
print("\nregular windows (window id of each token):\n", win_id)
shifted = np.roll(grid, shift=(-(M // 2), -(M // 2)), axis=(0, 1))   # cyclic shift by M/2
print("tokens now grouped in window 0 after the shift:", np.sort(shifted[:M, :M].ravel()))
print("their original windows:", np.unique(win_id.ravel()[shifted[:M, :M].ravel()]))


# ---------------- 3) attention cost: global MSA vs window MSA (Swin paper, eq. 1-2) ----------------
def flops_msa(h, w, C):
    return 4 * h * w * C ** 2 + 2 * (h * w) ** 2 * C          # projections + QK^T and AV


def flops_wmsa(h, w, C, M=7):
    return 4 * h * w * C ** 2 + 2 * M ** 2 * h * w * C         # attention only inside M x M windows


print("\nstage-1 attention cost in multiply-adds (patch 4, C = 96, M = 7), one layer:")
for img_size in [224, 448, 896]:
    h = w = img_size // 4
    g, wn = flops_msa(h, w, 96), flops_wmsa(h, w, 96)
    print(f"{img_size:4d}px: {h * w:6d} tokens  global {g / 1e9:8.2f} G  window {wn / 1e9:6.2f} G  ratio {g / wn:6.1f}x")`;

const VISION_NUMPY_OUT = String.raw`patches (16, 192) -> tokens (16, 16); max |linear - strided conv| = 1.7e-15
sequence entering the encoder: (17, 16) (1 [CLS] + 16 patches)
ViT-B/16 at 224x224: 196 patches + [CLS] = 197 tokens, patch-embedding weights 590,592
ViT-B/16 at 384x384: 576 patches + [CLS] = 577 tokens, patch-embedding weights 590,592

regular windows (window id of each token):
 [[0 0 0 0 1 1 1 1]
 [0 0 0 0 1 1 1 1]
 [0 0 0 0 1 1 1 1]
 [0 0 0 0 1 1 1 1]
 [2 2 2 2 3 3 3 3]
 [2 2 2 2 3 3 3 3]
 [2 2 2 2 3 3 3 3]
 [2 2 2 2 3 3 3 3]]
tokens now grouped in window 0 after the shift: [18 19 20 21 26 27 28 29 34 35 36 37 42 43 44 45]
their original windows: [0 1 2 3]

stage-1 attention cost in multiply-adds (patch 4, C = 96, M = 7), one layer:
 224px:   3136 tokens  global     2.00 G  window   0.15 G  ratio   13.8x
 448px:  12544 tokens  global    30.67 G  window   0.58 G  ratio   52.8x
 896px:  50176 tokens  global   485.23 G  window   2.32 G  ratio  209.0x`;

const PARAMS_TORCH = String.raw`import torch
import torchvision
from transformers import (BertConfig, BertModel, GPT2Config, GPT2Model, T5Config, T5Model,
                          BartConfig, BartModel)
from transformers.utils import logging

logging.set_verbosity_error()          # silence informational messages
torch.manual_seed(0)


def count(model):
    return sum(p.numel() for p in model.parameters())


rows = []
# Defaults of BertConfig() / GPT2Config() are the BERT-base and GPT-2 (small) shapes.
bert = BertModel(BertConfig())
rows.append(("BERT-base (encoder)", count(bert), bert.embeddings.word_embeddings.weight.numel()))
del bert
gpt2 = GPT2Model(GPT2Config())
rows.append(("GPT-2 small (decoder)", count(gpt2), gpt2.wte.weight.numel()))
del gpt2
t5 = T5Model(T5Config(d_model=512, d_ff=2048, d_kv=64, num_layers=6, num_heads=8, vocab_size=32128))
rows.append(("T5-small shape (enc-dec)", count(t5), t5.shared.weight.numel()))
del t5
bart = BartModel(BartConfig(d_model=768, encoder_layers=6, decoder_layers=6, encoder_attention_heads=12,
                            decoder_attention_heads=12, encoder_ffn_dim=3072, decoder_ffn_dim=3072))
rows.append(("BART-base shape (enc-dec)", count(bart), bart.shared.weight.numel()))
del bart
vit = torchvision.models.vit_b_16(weights=None)
rows.append(("ViT-B/16 (torchvision)", count(vit), count(vit.conv_proj)))
swin = torchvision.models.swin_t(weights=None)
rows.append(("Swin-T (torchvision)", count(swin), count(swin.features[0])))

print(f"{'model':28s} {'parameters':>12s} {'embedding':>12s}")
for name, n, emb in rows:
    print(f"{name:28s} {n:12,d} {emb:12,d}")

# the ViT patch embedding IS a 16x16 convolution with stride 16 ...
conv = vit.conv_proj
print("\nvit.conv_proj:", conv)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y_conv = conv(x).flatten(2).transpose(1, 2)                     # (1, 196, 768)
    # ... which equals: cut 16x16 patches, flatten each, multiply by the reshaped kernel
    patches = x.unfold(2, 16, 16).unfold(3, 16, 16)                 # (1, 3, 14, 14, 16, 16)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 196, 3 * 16 * 16)
    y_lin = patches @ conv.weight.reshape(768, -1).T + conv.bias
print("tokens:", tuple(y_conv.shape), " max |conv - linear| =", f"{(y_conv - y_lin).abs().max().item():.1e}")
with torch.no_grad():
    print("ViT-B/16 logits for one random image:", tuple(vit.eval()(x).shape))`;

const PARAMS_TORCH_OUT = String.raw`model                          parameters    embedding
BERT-base (encoder)           109,482,240   23,440,896
GPT-2 small (decoder)         124,439,808   38,597,376
T5-small shape (enc-dec)       60,506,624   16,449,536
BART-base shape (enc-dec)     139,420,416   38,603,520
ViT-B/16 (torchvision)         86,567,656      590,592
Swin-T (torchvision)           28,288,354        4,896

vit.conv_proj: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
tokens: (1, 196, 768)  max |conv - linear| = 6.4e-06
ViT-B/16 logits for one random image: (1, 1000)`;

const TINY_TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
torch.set_num_threads(1)

# =============== Part 1: a tiny BERT-style masked-LM ===============
TEXT = ("alice was beginning to get very tired of sitting by her sister on the bank . "
        "so she was considering in her own mind whether the pleasure of making a daisy chain "
        "would be worth the trouble of getting up and picking the daisies . "
        "suddenly a white rabbit with pink eyes ran close by her . "
        "the rabbit took a watch out of its waistcoat pocket and looked at it and then hurried on . "
        "alice started to her feet and ran across the field after it .")
specials = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
vocab = specials + sorted(set(TEXT.split()))
stoi = {w: i for i, w in enumerate(vocab)}
words = torch.tensor([stoi[w] for w in TEXT.split()])
V, D, L = len(vocab), 64, 12                     # vocab, width, sequence length (incl. CLS/SEP)
print(f"MLM: {len(words)} words, vocabulary {V}")


class TinyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(L, D)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)            # NO causal mask: bidirectional
        self.mlm_head = nn.Linear(D, V)                       # predicts the original token

    def forward(self, ids):
        h = self.enc(self.tok(ids) + self.pos(torch.arange(ids.shape[1])))
        return self.mlm_head(h), h[:, 0]                      # token logits, [CLS] vector


def make_batch(B=32):
    s = torch.randint(0, len(words) - (L - 2), (B,))
    ids = torch.stack([torch.cat([torch.tensor([1]), words[i:i + L - 2], torch.tensor([2])]) for i in s])
    labels = torch.full_like(ids, -100)
    sel = torch.rand(ids.shape) < 0.15
    sel[:, 0] = sel[:, -1] = False                            # never mask [CLS] / [SEP]
    labels[sel] = ids[sel]
    r = torch.rand(ids.shape)
    inp = ids.clone()
    inp[sel & (r < 0.8)] = 3                                  # 80% [MASK]
    rand = sel & (r >= 0.8) & (r < 0.9)                       # 10% random word, 10% unchanged
    inp[rand] = torch.randint(4, V, (int(rand.sum()),))
    return inp, labels


bert = TinyBERT()
opt = torch.optim.AdamW(bert.parameters(), lr=2e-3, weight_decay=0.01)
for step in range(301):
    inp, labels = make_batch()
    logits, _ = bert(inp)
    loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:3d}  MLM loss {loss.item():.3f}  (on {int((labels != -100).sum())} masked tokens)")

bert.eval()
probe = "[CLS] the rabbit took a [MASK] out of its waistcoat pocket [SEP]".split()
with torch.no_grad():
    logits, cls_vec = bert(torch.tensor([[stoi[w] for w in probe]]))
p = F.softmax(logits[0, probe.index("[MASK]")], -1)
top = p.topk(3)
print("  fill-in '" + " ".join(probe[1:-1]) + "':",
      ", ".join(f"{vocab[i]} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))
print("  [CLS] vector for a classification head:", tuple(cls_vec.shape))

# =============== Part 2: a tiny Vision Transformer on 8x8 digits ===============
X, y = load_digits(return_X_y=True)                            # 1797 grey 8x8 images, 10 classes
X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16.0
y = torch.tensor(y)
perm = torch.randperm(len(X))
tr, te = perm[:1400], perm[1400:]
P, DV = 2, 48                                                  # 2x2 patches -> 16 tokens


class TinyViT(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.patch = nn.Conv2d(1, DV, kernel_size=P, stride=P)     # patch embedding = strided conv
        self.cls = nn.Parameter(torch.zeros(1, 1, DV))
        self.pos = nn.Parameter(torch.randn(1, 1 + (8 // P) ** 2, DV) * 0.02)
        layer = nn.TransformerEncoderLayer(DV, 4, 2 * DV, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm, self.head = nn.LayerNorm(DV), nn.Linear(DV, n_classes)

    def forward(self, img, verbose=False):
        t = self.patch(img).flatten(2).transpose(1, 2)            # (B, 16, DV)
        t = torch.cat([self.cls.expand(len(img), -1, -1), t], 1) + self.pos
        h = self.enc(t)
        if verbose:
            print("  shapes: image", tuple(img.shape), "-> tokens", tuple(t.shape), "-> encoder", tuple(h.shape))
        return self.head(self.norm(h[:, 0]))                      # classify from [CLS]


vit = TinyViT()
print(f"\nViT: {sum(p.numel() for p in vit.parameters()):,} parameters")
with torch.no_grad():
    out = vit(X[:2], verbose=True)
print("  untrained logits for image 0:", [round(v, 2) for v in out[0].tolist()])
opt = torch.optim.AdamW(vit.parameters(), lr=3e-3, weight_decay=0.05)
for epoch in range(1, 13):
    vit.train()
    for idx in tr[torch.randperm(len(tr))].split(64):
        loss = F.cross_entropy(vit(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 4 == 0:
        vit.eval()
        with torch.no_grad():
            acc = (vit(X[te]).argmax(1) == y[te]).float().mean().item()
        print(f"  epoch {epoch:2d}  train loss {loss.item():.3f}  test accuracy {acc:.3f} ({len(te)} images)")`;

const TINY_TORCH_OUT = String.raw`MLM: 89 words, vocabulary 65
  step   0  MLM loss 4.578  (on 48 masked tokens)
  step 100  MLM loss 2.517  (on 49 masked tokens)
  step 200  MLM loss 0.663  (on 44 masked tokens)
  step 300  MLM loss 0.518  (on 64 masked tokens)
  fill-in 'the rabbit took a [MASK] out of its waistcoat pocket': watch 0.89, of 0.06, its 0.02
  [CLS] vector for a classification head: (1, 64)

ViT: 39,610 parameters
  shapes: image (2, 1, 8, 8) -> tokens (2, 17, 48) -> encoder (2, 17, 48)
  untrained logits for image 0: [0.21, -1.39, -0.81, 0.35, -0.49, 0.1, 0.72, -0.78, -0.6, -0.02]
  epoch  4  train loss 0.826  test accuracy 0.675 (397 images)
  epoch  8  train loss 0.722  test accuracy 0.907 (397 images)
  epoch 12  train loss 0.085  test accuracy 0.952 (397 images)`;

const KERAS_VIT = String.raw`import tensorflow as tf
from tensorflow import keras

IMG, P, D, HEADS, LAYERS, CLASSES = 32, 4, 64, 4, 4, 10


class AddClsAndPosition(keras.layers.Layer):
    def build(self, shape):
        n = shape[1]
        self.cls = self.add_weight(shape=(1, 1, D), initializer="zeros", name="cls")
        self.pos = self.add_weight(shape=(1, n + 1, D), initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")

    def call(self, t):
        cls = tf.repeat(self.cls, tf.shape(t)[0], axis=0)
        return tf.concat([cls, t], axis=1) + self.pos


inp = keras.Input((IMG, IMG, 3))
t = keras.layers.Conv2D(D, kernel_size=P, strides=P)(inp)        # patch embedding = strided conv
t = keras.layers.Reshape(((IMG // P) ** 2, D))(t)                # (batch, 64 patches, D)
t = AddClsAndPosition()(t)
for _ in range(LAYERS):                                          # bidirectional encoder blocks
    h = keras.layers.LayerNormalization()(t)
    t = t + keras.layers.MultiHeadAttention(HEADS, D // HEADS)(h, h)
    h = keras.layers.LayerNormalization()(t)
    t = t + keras.layers.Dense(D)(keras.layers.Dense(2 * D, activation="gelu")(h))
logits = keras.layers.Dense(CLASSES)(keras.layers.LayerNormalization()(t[:, 0]))   # [CLS] token
vit = keras.Model(inp, logits)
vit.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
vit.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=0.05),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
vit.fit(x_train / 255.0, y_train, batch_size=128, epochs=3, validation_data=(x_test / 255.0, y_test))`;

const lesson: Lesson = {
  id: "m11/transformer-families",
  title: "BERT, GPT, T5, ViT and Swin",
  summary:
    "The three Transformer families — encoder-only (BERT, RoBERTa), decoder-only (GPT-1/2/3) and encoder–decoder (T5, BART) — defined by their attention masks and pretraining objectives, and the move to vision with the Vision Transformer (patch embedding = strided convolution) and the Swin Transformer (shifted windows, linear cost). Masking procedures, mask patterns, parameter counts of real configurations and tiny working models, all computed.",
  minutes: 95,
  level: "advanced",
  prerequisites: ["m11/transformer-architecture", "m11/training-inference", "m10/scaled-dot-product-attention", "m08/convolution"],
  next: ["m12/transfer-learning", "m13/nlp-tasks", "m20/llm-pretraining", "m15/multimodal"],
  terms: ["encoder-only", "decoder-only", "encoder-decoder", "masked-language-modeling", "next-sentence-prediction", "cls-token", "span-corruption", "in-context-learning", "prefix-lm", "vision-transformer", "patch-embedding", "swin-transformer", "shifted-window-attention", "patch-merging"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "The original Transformer (Vaswani et al., 2017) was an encoder–decoder for machine translation. Within two years it had split into three families, distinguished by which half of the architecture they keep and — more fundamentally — by **which positions each token may attend to** and **what the model is trained to predict**:" },
      { t: "list", items: [
        "**Encoder-only** (BERT, Devlin et al., 2018; RoBERTa, Liu et al., 2019): bidirectional attention, trained to reconstruct masked tokens. Produces contextual representations of an input; used for classification, tagging, retrieval and extractive question answering.",
        "**Decoder-only** (GPT, Radford et al., 2018; GPT-2, 2019; GPT-3, Brown et al., 2020): causal attention, trained to predict the next token. Generates text; at scale it performs new tasks from instructions and examples in the prompt (**in-context learning**). Nearly all large language models today belong to this family.",
        "**Encoder–decoder** (T5, Raffel et al., 2019/2020; BART, Lewis et al., 2019): a bidirectional encoder reads the input, a causal decoder generates the output while cross-attending to it. Natural for input→output tasks: translation, summarization, and T5's “everything is text-to-text”.",
      ] },
      { t: "p", text: "Then the same encoder was pointed at images. The **Vision Transformer** (ViT, Dosovitskiy et al., 2020, “An Image is Worth 16x16 Words”) cuts an image into patches, treats each patch as a token and runs an unmodified BERT-style encoder — showing that with enough pretraining data, convolutions are not required for state-of-the-art image recognition. The **Swin Transformer** (Liu et al., 2021) reintroduced locality and hierarchy: attention inside small windows that shift between layers, and patch merging that builds a CNN-like feature pyramid, making the cost linear in image size so that Transformers could serve as backbones for detection and segmentation." },
      { t: "p", text: "This lesson treats each family through the three things that define it — architecture, attention mask, pretraining objective — and backs every claim with computation: the BERT masking procedure, the four mask patterns, parameter counts of real configurations built with random weights, the proof that patch embedding is a strided convolution, Swin's complexity, and tiny working BERT and ViT models." },
      { t: "callout", kind: "key", text: "Same building block, different mask and objective. Bidirectional mask + masked-token prediction → understanding (BERT). Causal mask + next-token prediction → generation (GPT). Bidirectional encoder + causal decoder + cross-attention → transformation of one sequence into another (T5, BART). Image patches as tokens → ViT; windows + hierarchy → Swin." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[The Transformer Architecture](/learn/m11/transformer-architecture) — encoder and decoder blocks, cross-attention, residuals and LayerNorm. Each family keeps a subset of these pieces.",
        "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — causal-LM training with shifted targets, teacher forcing, the KV cache and the $O(n^2 d)$ cost of attention. We build on its objectives here.",
        "[Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — attention masks as matrices of allowed (query, key) pairs; padding and causal masks.",
        "[Positional Encoding](/learn/m10/positional-encoding) — BERT, GPT-2 and ViT use **learned** absolute position embeddings; T5 uses relative position biases; Swin uses a relative position bias inside each window.",
        "[Convolution](/learn/m08/convolution) — kernel size, stride, output size. ViT's patch embedding is shown to be exactly a convolution with kernel = stride = patch size.",
        "[Tokenization](/learn/m13/text-preprocessing) — BERT uses WordPiece, GPT-2 byte-level BPE, T5 SentencePiece; the special tokens `[CLS]`, `[SEP]`, `[MASK]` live in the vocabulary.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "A reader, a writer and a translator", text: "**BERT is a reader** doing a cloze test: a sentence with some words blanked out, and it may look both left and right to fill them in. Good readers build a rich understanding of every word in context — exactly what classification and search need — but a cloze test never teaches you to write a story from left to right. **GPT is a writer**: it only ever sees what it has written so far and must continue it. Writers trained on enough text learn to continue anything, including “Question: … Answer:”. **T5 is a translator** with two roles: one part reads the whole source text freely, the other writes the output left to right while glancing back at the source." },
      { t: "p", text: "The attention mask is the formal version of these roles. A bidirectional mask lets every token see every other token (the reader). A causal mask hides the future (the writer). An encoder–decoder gives the writer a second, unmasked view of the source through cross-attention (the translator). A **prefix-LM** mask is a hybrid inside one stack: the prompt is read bidirectionally, the continuation is written causally." },
      { t: "callout", kind: "analogy", title: "An image as a sentence of puzzle pieces", text: "ViT cuts a photo into a grid of 16×16-pixel puzzle pieces, flattens each piece into a vector, and reads the grid row by row as if it were a 196-word sentence. Nothing tells the model that piece 15 sits right below piece 1 except the learned position embeddings — it must discover image geometry from data, which is why ViT needs a lot of it. Swin instead looks at pieces through small windows (like reading a page paragraph by paragraph), shifts the window grid every other layer so that neighbouring windows exchange information, and merges pieces into bigger ones as it goes deeper — the same coarse-to-fine pyramid a CNN builds." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "BERT: encoder-only, masked language modelling" },
      { t: "steps", items: [
        { title: "Input format", text: "`[CLS] sentence A [SEP] sentence B [SEP]`, WordPiece tokens (vocabulary 30,522 in the released English models). Each token's input vector = token embedding + learned position embedding (up to 512 positions) + **segment embedding** (A or B)." },
        { title: "Encoder", text: "A stack of bidirectional Transformer encoder layers (BERT-base: 12 layers, $d=768$, 12 heads, FFN 3072; BERT-large: 24 layers, $d=1024$, 16 heads). No causal mask; only padding is masked." },
        { title: "Masked LM (MLM)", text: "Select 15% of token positions at random. Of those: 80% are replaced by `[MASK]`, 10% by a random token, 10% left unchanged. The model predicts the original token at the selected positions only. The 10%/10% exist because `[MASK]` never appears at fine-tuning time: the model must keep a good representation of every token, not just of `[MASK]`." },
        { title: "Next-sentence prediction (NSP)", text: "Half the time sentence B really follows A, half the time it is a random sentence. A binary classifier on the final `[CLS]` vector predicts which. RoBERTa (Liu et al., 2019) found NSP unnecessary: removing it, masking dynamically (a new mask each time a sequence is seen), and training longer on more data with bigger batches gave better results." },
        { title: "Fine-tuning heads", text: "Sentence classification: a linear layer on the final `[CLS]` vector (BERT passes it through a tanh *pooler* layer first). Token tagging (NER): a linear layer on every token's vector. Extractive QA: two vectors score each token as answer start and answer end. All parameters are fine-tuned with a small learning rate for a few epochs." },
      ] },
      { t: "h", text: "GPT: decoder-only, causal language modelling" },
      { t: "steps", items: [
        { title: "Architecture", text: "A stack of decoder blocks **without** cross-attention: masked self-attention + FFN. Token + learned position embeddings in; logits over the vocabulary out (GPT-2 ties the output matrix to the token embedding)." },
        { title: "Objective", text: "Next-token prediction at every position, exactly as in [Training and Inference](/learn/m11/training-inference)." },
        { title: "GPT-1 (2018)", text: "12 layers, $d=768$; pretrained on a books corpus, then **fine-tuned** per task with task-specific input formatting and a small head." },
        { title: "GPT-2 (2019)", text: "Up to 1.5 billion parameters, trained on WebText, context 1024, LayerNorm moved to the input of each sub-block (pre-LN). Showed that a language model can perform tasks **zero-shot** when the task is phrased as text continuation." },
        { title: "GPT-3 (2020)", text: "175 billion parameters (96 layers, $d=12288$, 96 heads, context 2048). Introduced the framing of **in-context learning**: zero-, one- and few-shot prompting, where demonstrations are placed in the prompt and no weights are updated." },
      ] },
      { t: "h", text: "T5 and BART: encoder–decoder" },
      { t: "list", items: [
        "**T5** casts every task as text-to-text: input `translate English to German: That is good.` → output `Das ist gut.`; classification outputs the label as a word. One model, one loss (cross-entropy on output tokens with teacher forcing), one decoding procedure for all tasks.",
        "**Span corruption** (T5's pretraining objective): drop random spans totalling about 15% of the tokens (mean span length 3) and replace each span with a unique sentinel (`<extra_id_0>`, `<extra_id_1>`, …). The decoder outputs only the dropped spans, each preceded by its sentinel. Targets are short, so pretraining is cheaper than reconstructing the whole input.",
        "**Relative position bias**: T5 adds a learned scalar per (head, bucketed relative distance) to the attention logits instead of adding position embeddings to the inputs.",
        "**BART** is a denoising autoencoder: corrupt the text (token masking, deletion, **text infilling** of spans with a single mask, sentence permutation, rotation) and train the decoder to regenerate the full original. Well suited to summarization and generation that rewrites its input.",
      ] },
      { t: "h", text: "Vision Transformer (ViT)" },
      { t: "steps", items: [
        { title: "Patchify", text: "Split an $H\\times W\\times C$ image into non-overlapping $P\\times P$ patches: $N = HW/P^2$ patches (224×224 with $P=16$ gives $14\\times14 = 196$)." },
        { title: "Patch embedding", text: "Flatten each patch to a $P^2C$ vector (768 for 16×16×3) and multiply by a shared matrix $E\\in\\mathbb R^{P^2C\\times D}$. This is identical to a convolution with $D$ filters of size $P\\times P$ and stride $P$ — which is how it is implemented." },
        { title: "Add [CLS] and positions", text: "Prepend a learnable `[CLS]` vector (sequence length $N+1 = 197$) and add learned 1-D position embeddings." },
        { title: "Encoder", text: "A standard pre-LN Transformer encoder, bidirectional (ViT-B: 12 layers, $D=768$, 12 heads, MLP 3072)." },
        { title: "Head", text: "LayerNorm, then a linear classifier on the final `[CLS]` vector (or on the mean of the patch tokens)." },
        { title: "Data hunger", text: "With no built-in locality or translation equivariance, ViT must learn them. The paper reports that ViT trained on ImageNet-1k alone underperforms comparable ResNets, but overtakes them when pretrained on much larger datasets (ImageNet-21k, JFT-300M). Later work (DeiT, Touvron et al., 2020) showed strong augmentation, regularization and distillation make ImageNet-1k training work." },
      ] },
      { t: "h", text: "Swin Transformer" },
      { t: "steps", items: [
        { title: "Small patches", text: "Patch size 4 (a 4×4 stride-4 convolution): a 224×224 image becomes a $56\\times56$ grid of tokens with $C$ channels (Swin-T: $C = 96$)." },
        { title: "Window attention (W-MSA)", text: "Partition the grid into non-overlapping $M\\times M$ windows ($M=7$) and run self-attention **inside each window only**: each token attends to 49 others, not 3136." },
        { title: "Shifted windows (SW-MSA)", text: "In every second block, shift the window grid by $\\lfloor M/2\\rfloor$ tokens, so new windows straddle the old boundaries and information flows between neighbouring windows. Implemented as a cyclic shift (`roll`) of the feature map plus an attention mask that stops tokens which were not adjacent before the roll from attending to each other." },
        { title: "Patch merging", text: "Between stages, concatenate each 2×2 group of neighbouring tokens ($4C$ channels) and project to $2C$: resolution halves, channels double. Four stages give feature maps at strides 4, 8, 16, 32 — a hierarchy like a CNN's, usable by detection and segmentation heads." },
        { title: "Relative position bias", text: "A learned bias $B$ indexed by the relative offset between two tokens in a window is added to the attention logits: $\\mathrm{softmax}(QK^\\top/\\sqrt d + B)V$." },
      ] },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Pretraining objectives" },
      { t: "p", text: String.raw`Let $\mathcal M$ be the set of selected positions and $\tilde x$ the corrupted input. The **MLM** loss is the cross-entropy at selected positions only; **NSP** is a binary cross-entropy on the [CLS] vector $h_{\text{CLS}}$; the **causal LM** loss covers every position; the **encoder–decoder** loss covers every output token given the full input $x$:` },
      { t: "math", tex: String.raw`\mathcal L_{\text{MLM}} = -\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x),\qquad \mathcal L_{\text{NSP}} = -\big[s\log\sigma(w^\top h_{\text{CLS}}) + (1-s)\log(1-\sigma(w^\top h_{\text{CLS}}))\big]` },
      { t: "math", tex: String.raw`\mathcal L_{\text{CLM}} = -\frac1T\sum_{t}\log p_\theta(x_{t+1}\mid x_{\le t}),\qquad \mathcal L_{\text{seq2seq}} = -\frac1m\sum_{j=1}^{m}\log p_\theta(y_j\mid y_{<j},\,x)`, caption: "$s\\in\\{0,1\\}$: whether sentence B really follows A; $\\sigma$: sigmoid; $y$: target sequence of length $m$." },
      { t: "p", text: "Per input token, BERT's procedure gives: probability $0.15\\times0.8 = 0.12$ of becoming `[MASK]`, $0.015$ of becoming a random token, $0.015$ of staying unchanged but still being predicted, and $0.85$ of being ignored by the loss. Only 15% of positions produce a learning signal — one reason causal LM (100% of positions) and ELECTRA's replaced-token detection were later found to be more sample-efficient." },
      { t: "p", text: String.raw`For **T5 span corruption** with corruption rate 15% and mean span length 3, a 512-token input loses $0.15\times512 = 76.8$ tokens in about $76.8/3 = 25.6$ spans. The target is those $\approx 77$ tokens plus $\approx 26$ sentinels — about a fifth of the input length.` },
      { t: "h", text: "The four attention masks as matrices" },
      { t: "p", text: String.raw`Write the mask as $M_{ij}\in\{0,-\infty\}$ added to the scores (0 = query $i$ may see key $j$). For $n$ tokens and a prefix of length $p$:` },
      { t: "math", tex: String.raw`M^{\text{bi}}_{ij} = 0,\qquad M^{\text{causal}}_{ij} = \begin{cases}0 & j\le i\\ -\infty & j>i\end{cases},\qquad M^{\text{prefix}}_{ij} = \begin{cases}0 & j\le i \text{ or } j < p\\ -\infty & \text{otherwise}\end{cases}` },
      { t: "p", text: String.raw`Allowed pairs: bidirectional $n^2$ (36 for $n=6$); causal $n(n+1)/2$ (21); prefix-LM $n(n+1)/2 + p(p-1)/2$ (21 + 3 = 24 for $p=3$). Cross-attention is a rectangular $m\times n_{\text{enc}}$ matrix in which every decoder query sees every (non-padding) encoder key. The NumPy code prints all four.` },
      { t: "h", text: "Parameter counts from the configuration" },
      { t: "p", text: String.raw`One encoder layer of width $d$ with FFN width $4d$ has $4(d^2+d)$ attention parameters (Q, K, V, O with biases), $8d^2 + 5d$ FFN parameters and $4d$ LayerNorm parameters, i.e. $12d^2 + 13d$. For $d=768$: $7{,}087{,}872$ per layer and $85{,}054{,}464$ for 12 layers. Adding embeddings gives the full model:` },
      { t: "steps", items: [
        { title: "BERT-base", tex: String.raw`\underbrace{(30522+512+2)\cdot768 + 2\cdot768}_{\text{token+position+segment+LN}\;=\;23{,}837{,}184} + 85{,}054{,}464 + \underbrace{768^2+768}_{\text{pooler}\;=\;590{,}592} = 109{,}482{,}240` },
        { title: "GPT-2 small", tex: String.raw`\underbrace{50257\cdot768}_{38{,}597{,}376} + \underbrace{1024\cdot768}_{786{,}432} + 85{,}054{,}464 + \underbrace{2\cdot768}_{\text{final LN}} = 124{,}439{,}808` },
        { title: "ViT-B/16", tex: String.raw`\underbrace{3\cdot16^2\cdot768+768}_{\text{patch conv}\;=\;590{,}592} + \underbrace{768}_{\text{CLS}} + \underbrace{197\cdot768}_{151{,}296} + 85{,}054{,}464 + \underbrace{1536}_{\text{LN}} + \underbrace{768\cdot1000+1000}_{\text{head}} = 86{,}567{,}656` },
        { title: "GPT-3 estimate", text: "The dominant term $12Ld^2$ ignores biases and LayerNorms:", tex: String.raw`12\cdot96\cdot12288^2 + (50257+2048)\cdot12288 = 173{,}946{,}175{,}488 + 642{,}723{,}840 \approx 1.746\times10^{11}` },
      ] },
      { t: "p", text: "All three exact totals are reproduced by constructing the models from their configurations in the PyTorch section; the GPT-3 estimate lands within 0.3% of the published 175 billion. Note how much of a small model is embedding: 21% of BERT-base and 31% of GPT-2 small, but under 1% of ViT-B/16, whose “vocabulary” is a 768×768 linear map." },
      { t: "h", text: "Patch embedding is a strided convolution" },
      { t: "p", text: String.raw`A convolution with $D$ filters $K_d\in\mathbb R^{C\times P\times P}$, stride $P$ and no padding produces at output location $(i,j)$:` },
      { t: "math", tex: String.raw`y_{d,i,j} = b_d + \sum_{c=1}^{C}\sum_{u=0}^{P-1}\sum_{v=0}^{P-1} K_{d,c,u,v}\, x_{c,\,iP+u,\,jP+v} = b_d + \big\langle \operatorname{vec}(K_d),\ \operatorname{vec}(x_{:,\,iP:iP+P,\,jP:jP+P})\big\rangle` },
      { t: "p", text: String.raw`Because the stride equals the kernel size, the receptive fields tile the image without overlap, so each output location sees exactly one patch, and the sum is the dot product of the flattened patch with the flattened filter. Stacking the $D$ flattened filters as columns gives $E$ with $E_{:,d} = \operatorname{vec}(K_d)$, so $y_{:,i,j} = E^\top\operatorname{vec}(\text{patch}_{ij}) + b$ — the linear patch embedding. Output size: $\lfloor (H-P)/P\rfloor + 1 = H/P$ per side. Numerically the two agree to $1.7\times10^{-15}$ (NumPy, float64) and $6.4\times10^{-6}$ (PyTorch's own ViT-B/16 layer, float32).` },
      { t: "h", text: "Cost of global vs window attention (Swin)" },
      { t: "p", text: String.raw`For an $h\times w$ grid of tokens with $C$ channels, the Swin paper counts multiply-adds of one multi-head self-attention layer as` },
      { t: "math", tex: String.raw`\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C, \qquad \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC`, caption: "$4hwC^2$: Q, K, V and output projections. $2(hw)^2C$: $QK^\\top$ and $AV$ over all $hw$ tokens. With windows, each token attends to only $M^2$ others." },
      { t: "p", text: String.raw`Global attention is quadratic in the number of tokens $hw$ (hence quartic in image side); window attention is **linear** in $hw$ for fixed $M$. The ratio is $(4C + 2hw)/(4C + 2M^2)$. Stage 1 of Swin-T on a 224×224 image ($h=w=56$, $C=96$, $M=7$):` },
      { t: "steps", items: [
        { title: "Global", tex: String.raw`4\cdot3136\cdot96^2 + 2\cdot3136^2\cdot96 = 115{,}605{,}504 + 1{,}888{,}223{,}232 \approx 2.00\times10^9` },
        { title: "Windowed", tex: String.raw`4\cdot3136\cdot96^2 + 2\cdot49\cdot3136\cdot96 = 115{,}605{,}504 + 29{,}503{,}488 \approx 0.145\times10^9` },
        { title: "Ratio", tex: String.raw`\frac{4\cdot96 + 2\cdot3136}{4\cdot96 + 2\cdot49} = \frac{6656}{482} \approx 13.8` },
      ] },
      { t: "p", text: "Doubling the image side multiplies the token count by 4: windowed cost grows 4× (0.15 → 0.58 → 2.32 G multiply-adds at 224, 448, 896 pixels), global cost about 16× (2.00 → 30.67 → 485.23 G). Patch merging further reduces tokens by 4× per stage while doubling $C$." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "The attention explorer computes real multi-head self-attention on any sentence you type. Use it to see the difference between the **encoder-only** and **decoder-only** families, which is nothing but the mask." },
      { t: "sim", id: "attention", caption: "Toggle the causal mask to switch between a BERT-style (bidirectional) and a GPT-style (causal) layer." },
      { t: "list", ordered: true, items: [
        "Type `the bank of the river` with **Causal mask** off and go to the softmax stage. Click the row for `bank`: it puts weight on `river` (a later word) — a bidirectional encoder can use right context to disambiguate. (The weights are random, so the *amount* is arbitrary; the *possibility* is the point.)",
        "Turn **Causal mask** on: the `bank` row now has exact zeros for `of the river`. A decoder-only model must commit to a representation of `bank` before seeing `river`.",
        "Compare the last row with the mask on and off: identical. In a causal model the final token is the only one that sees the whole sequence, which is why GPT-style classifiers read the **last** token while BERT reads `[CLS]` (the first).",
        "Count the non-zero weights with the mask on: $n(n+1)/2$, matching the causal pattern printed by the NumPy code below. A prefix-LM would additionally fill the top-left $p\\times p$ block.",
      ] },
      { t: "p", text: "The second explorer is a small **decoder-only** (GPT-style, pre-LN, weight-tied) character model that was actually trained on Chapter I of *Alice's Adventures in Wonderland* and runs in your browser: the GPT family in miniature." },
      { t: "sim", id: "transformer", caption: "A trained decoder-only Transformer: follow a prompt through attention and feed-forward layers to its next-character prediction, then generate." },
      { t: "list", items: [
        "Open an **Attn** stage and compare the per-head mini heatmaps: every one is lower-triangular — the causal pattern printed by the NumPy code below. A BERT-style encoder's heatmaps would be full squares.",
        "Type `a White Rabbit ran` and, at the **Logits** stage, click the space after `White`: the prediction for the next character there is the same as if the prompt ended at `White ` — the later words `Rabbit ran` cannot influence it. A decoder can only continue; filling a blank using words on both sides is an encoder's job.",
        "Use the Generate panel (greedy or sample with the **Temperature T** slider): this is the decoder-only family's native mode of use — every task must be phrased as a continuation, which is exactly the framing GPT-2 and GPT-3 exploited.",
      ] },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "**1. BERT's masking procedure and the four attention-mask patterns.** `mlm_mask` implements the 15% / 80-10-10 rule, never touching special tokens, and returns labels with $-100$ (the conventional ignore index) at unselected positions. A short sentence shows each case; 4096 random sequences confirm the proportions." },
      { t: "code", lang: "python", title: "mlm_masking_and_masks.py", code: MASKS_NUMPY, output: MASKS_NUMPY_OUT },
      { t: "list", items: [
        "In the example (selection rate raised to 30% only for display), `rabbit`, `took` and `on` became `[MASK]`, `the` was replaced by the random word `and`, and the second `and` was selected but left unchanged — yet all five are prediction targets.",
        "Over $4096\\times126$ ordinary tokens: 14.94% selected; of those 79.83% `[MASK]`, 10.28% random, 9.90% unchanged — matching 15% and 80/10/10 up to sampling noise.",
        "The mask matrices have 36, 21, 24 and 20 allowed pairs, matching $n^2$, $n(n+1)/2$, $n(n+1)/2+p(p-1)/2$ and $m(n_{\\text{enc}}-1)$.",
      ] },
      { t: "p", text: "**2. ViT patch embedding from scratch, Swin windows and costs.** Patch embedding is computed by reshaping (flatten each patch, multiply by $E$) and by an explicit stride-$P$ convolution loop; then the `[CLS]` token and position embeddings are added. The Swin part partitions an 8×8 token grid into 4×4 windows, applies the cyclic shift, and evaluates the complexity formulas." },
      { t: "code", lang: "python", title: "patch_embedding_and_swin.py", code: VISION_NUMPY, output: VISION_NUMPY_OUT },
      { t: "list", items: [
        "`img.reshape(C, H//P, P, W//P, P).transpose(1, 3, 0, 2, 4)` puts the patch-grid indices first and the within-patch (channel, row, column) indices last, so the final reshape flattens each patch in the same $(c,u,v)$ order as the conv filter `W_e.T.reshape(D, C, P, P)` — that ordering is the whole trick, and the $1.7\\times10^{-15}$ difference confirms it.",
        "The patch-embedding weight count of ViT-B/16 (590,592) does not depend on the resolution; only the number of tokens (and the position-embedding table) does — 197 tokens at 224 px, 577 at 384 px. Fine-tuning at higher resolution therefore requires interpolating the position embeddings.",
        "After `np.roll` by $-M/2$ in both axes, the first window contains tokens 18–21, 26–29, 34–37, 42–45, which came from **all four** original windows: the shifted layer mixes information across the old window boundaries. Tokens that wrap around the image edge would land in the same window as distant tokens; Swin's attention mask blocks those pairs.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**Parameter counts measured from real configurations.** Hugging Face `transformers` model classes and `torchvision` models are built from their configurations with **random weights** (nothing is downloaded): `BertConfig()` and `GPT2Config()` default to the BERT-base and GPT-2-small shapes; T5 and BART are given the t5-small and bart-base shapes explicitly. The script then shows that `vit_b_16`'s patch embedding is a `Conv2d(3, 768, 16, stride=16)` and equals unfold + matrix multiply." },
          { t: "code", lang: "python", title: "family_parameter_counts.py", code: PARAMS_TORCH, output: PARAMS_TORCH_OUT },
          { t: "list", items: [
            "BERT-base 109,482,240, GPT-2 small 124,439,808 and ViT-B/16 86,567,656 match the hand derivations in the math section exactly. `BertModel` includes the pooler; `GPT2Model` excludes the (tied) output head.",
            "The T5-small-shaped model has about 60.5M parameters with a shared 32,128-token embedding; the BART-base-shaped model 139.4M. Swin-T has 28.3M, of which its stride-4 patch embedding plus LayerNorm is only 4,896.",
            "The float32 conv-vs-linear difference ($6.4\\times10^{-6}$) is round-off from summing 768 products in a different order.",
          ] },
          { t: "p", text: "**A tiny BERT-style MLM and a tiny ViT, trained.** Part 1 builds a 2-layer bidirectional encoder over a word vocabulary from the opening of *Alice's Adventures in Wonderland* (public domain), applies the 80/10/10 masking to each batch, and computes the loss only at masked positions (`ignore_index=-100`). Part 2 is a ViT on scikit-learn's 8×8 digit images: 2×2 patches via a strided `Conv2d`, a `[CLS]` token, learned position embeddings, a 2-layer encoder and a linear head on `[CLS]`." },
          { t: "code", lang: "python", title: "tiny_bert_and_vit.py", code: TINY_TORCH, output: TINY_TORCH_OUT },
          { t: "list", items: [
            "MLM loss falls from 4.578 (≈ $\\ln 65 = 4.17$ plus random-initialization over-confidence) to about 0.5 on a corpus of 89 words — it is memorizing, which is fine for a demonstration. Asked to fill `the rabbit took a [MASK] out of its waistcoat pocket`, it puts 0.89 on `watch` using context on **both** sides of the blank.",
            "The ViT turns each 8×8 image into 16 patch tokens + `[CLS]` = 17 tokens of width 48, and reaches 95.2% test accuracy on 397 held-out digits after 12 epochs, with 39,610 parameters. This is a toy: on such small images a small CNN is a natural baseline, and ViT's advantages appear only with large data.",
          ] },
          { t: "callout", kind: "tip", title: "Using pretrained models in practice", text: "With network access, `AutoModel.from_pretrained(\"bert-base-uncased\")`, `AutoModelForCausalLM.from_pretrained(\"gpt2\")`, `AutoModelForSeq2SeqLM.from_pretrained(\"t5-small\")` and `torchvision.models.vit_b_16(weights=\"IMAGENET1K_V1\")` load trained weights into the same architectures counted above. For a classification task, `AutoModelForSequenceClassification` adds the `[CLS]` head for you." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "A small ViT in Keras for 32×32 CIFAR-10 images: `Conv2D(D, kernel_size=P, strides=P)` is the patch embedding, a custom layer adds the `[CLS]` token and learned position embeddings, and `MultiHeadAttention` without a mask gives the bidirectional encoder." },
          { t: "code", lang: "python", title: "vit_keras.py (not executed in CI)", code: KERAS_VIT, output: "Prints the model summary (64 patch tokens + [CLS] of width 64), then the training loss and accuracy and the validation accuracy for each of 3 epochs." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "h", text: "The families compared" },
      { t: "table", head: ["Family", "Examples", "Architecture", "Pretraining objective", "Attention mask", "Typical tasks"], rows: [
        ["Encoder-only", "BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT", "Encoder stack", "Masked LM (+ NSP in BERT); replaced-token detection (ELECTRA)", "Bidirectional (padding only)", "Classification, NER, extractive QA, embeddings for search"],
        ["Decoder-only", "GPT-1/2/3 and most modern LLMs", "Decoder stack without cross-attention", "Causal LM (next token)", "Causal", "Open-ended generation, chat, code, few-shot tasks via prompting"],
        ["Encoder–decoder", "T5, BART, original Transformer", "Encoder + decoder with cross-attention", "Span corruption (T5), denoising (BART), seq2seq", "Encoder bidirectional; decoder causal + full cross-attention", "Translation, summarization, text-to-text tasks"],
        ["Prefix LM", "UniLM, T5 variants", "One stack", "LM on the continuation given a bidirectional prefix", "Prefix-LM", "Conditional generation in a single stack"],
        ["Vision Transformer", "ViT, DeiT", "Encoder over patch tokens", "Supervised classification (large data); self-supervised variants", "Bidirectional", "Image classification, backbone"],
        ["Hierarchical vision Transformer", "Swin", "Windowed encoder stages with patch merging", "Supervised classification (ImageNet)", "Within-window, shifted every other layer", "Backbone for detection and segmentation"],
      ] },
      { t: "h", text: "Notable variants" },
      { t: "table", head: ["Variant", "Change", "Advantage", "Disadvantage"], rows: [
        ["RoBERTa (Liu et al., 2019)", "No NSP, dynamic masking, more data, larger batches, longer training", "Better than BERT with the same architecture", "More pretraining compute"],
        ["ALBERT (Lan et al., 2019)", "Factorized embedding, cross-layer parameter sharing, sentence-order prediction", "Far fewer parameters", "Compute per forward pass is not reduced"],
        ["DistilBERT (Sanh et al., 2019)", "6-layer student distilled from BERT-base", "Smaller and faster at inference", "Some accuracy loss"],
        ["ELECTRA (Clark et al., 2020)", "Discriminator detects tokens replaced by a small generator", "Loss on every token, more sample-efficient", "Two networks during pretraining"],
        ["BART (Lewis et al., 2019)", "Encoder–decoder denoising autoencoder", "Strong for summarization and rewriting", "Two stacks to run"],
        ["DeiT (Touvron et al., 2020)", "Data-efficient ViT training with strong augmentation and a distillation token", "Trains well on ImageNet-1k alone", "Relies on a teacher for the distillation variant"],
        ["MAE (He et al., 2021)", "Mask 75% of patches; encoder sees visible ones, light decoder reconstructs pixels", "Self-supervised ViT pretraining, cheap encoder pass", "Pixel reconstruction targets need fine-tuning"],
        ["Swin (Liu et al., 2021)", "Shifted-window attention, patch merging", "Linear cost, multi-scale features", "More complex implementation (shift + masks)"],
      ] },
      { t: "h", text: "How to choose a family for a task" },
      { t: "list", items: [
        "**Output is a label or a span of the input** (sentiment, topic, NER, extractive QA, retrieval embeddings) and latency/cost matters → an encoder-only model, fine-tuned. One bidirectional pass, no generation loop.",
        "**Output is free text, or tasks are many and change often** (chat, code, reasoning, few-shot tasks without labelled data) → decoder-only. Prompting and in-context learning avoid task-specific training; fine-tune or instruction-tune for reliability.",
        "**Output is a transformation of a well-defined input** (translation, summarization, speech-to-text) with labelled pairs → encoder–decoder; the encoder reads the input bidirectionally once and the decoder's cross-attention keeps it in view. Large decoder-only models also do these tasks well via prompting, at a higher inference cost.",
        "**Images, large labelled or pretraining data, classification** → ViT (or a pretrained ViT fine-tuned). **Small data** → a CNN or a ViT pretrained elsewhere (DeiT-style training, MAE pretraining).",
        "**Dense prediction (detection, segmentation) or high resolution** → a hierarchical backbone such as Swin (or a CNN), because global attention over thousands of small patches is too expensive and heads need multi-scale features.",
      ] },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Search and retrieval**: BERT-style encoders produce embeddings for query–document matching and re-ranking; Google announced in 2019 that it uses BERT in Search ranking.",
        "**Text classification and information extraction**: fine-tuned encoder-only models for sentiment, intent detection, content moderation and named-entity recognition — see [NLP tasks](/learn/m13/nlp-tasks).",
        "**Generative assistants and code completion**: decoder-only LLMs (the GPT line and its successors) trained with the causal-LM objective, then instruction-tuned — see [Pretraining LLMs](/learn/m20/llm-pretraining) and [Alignment](/learn/m20/alignment).",
        "**Translation, summarization and speech recognition**: encoder–decoder models (T5 and BART derivatives; Whisper for speech).",
        "**Vision backbones**: ViT and Swin pretrained on large datasets are fine-tuned for classification, detection and segmentation — see [Transfer learning](/learn/m12/transfer-learning) and [Object detection](/learn/m12/object-detection). ViT-style image encoders are also the vision half of image–text models such as CLIP — see [Multimodal models](/learn/m15/multimodal).",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "table", head: ["Family", "Strengths", "Limitations"], rows: [
        ["Encoder-only", "Bidirectional context gives strong representations per parameter; one pass per input; cheap to fine-tune and serve", "Cannot generate fluently; MLM supervises only ~15% of tokens; `[MASK]` pretrain/fine-tune mismatch; fixed maximum length (512 in BERT)"],
        ["Decoder-only", "One objective that uses every token; generation and in-context learning; one model for many tasks; scales well", "Left-to-right only (no right context for a token's representation); sequential, memory-bound decoding with a growing KV cache"],
        ["Encoder–decoder", "Input read bidirectionally once; output generated with cross-attention; strong for seq2seq tasks with paired data", "Two stacks and three attention types to implement; less common at the largest scales"],
        ["ViT", "Global receptive field from layer 1; same architecture as NLP models; scales with data and model size", "No locality prior: needs large data or heavy augmentation/distillation; tokens quadratic in resolution; single-scale features"],
        ["Swin", "Linear cost in image size; multi-scale feature maps; good dense-prediction backbone", "Windowing, shifting and masking complicate implementation; attention is local within a layer"],
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Computing the MLM loss on all positions.** Unselected positions must have label $-100$ (ignored); otherwise the model is mostly trained to copy its input, which is trivial, and the gradient signal from the masked positions is diluted.",
        "**Selecting special tokens for masking** (`[CLS]`, `[SEP]`, `[PAD]`). They carry no content to predict and padding must never contribute to the loss. Exclude them from the candidate set, as `mlm_mask` does.",
        "**Using a causal mask in an encoder (or forgetting it in a decoder).** An encoder with a causal mask loses right context; a decoder without one sees its targets. The attention-mask pattern is the definition of the family — check it.",
        "**Reading the wrong token for a sentence-level head.** BERT-style: the first token (`[CLS]`). GPT-style: the last non-padding token (the only one that saw the whole input). Using GPT's first token gives a representation of one word.",
        "**Believing GPT-2 or GPT-3 are fine-tuned per task.** GPT-1 was; GPT-2 emphasized zero-shot transfer and GPT-3 few-shot in-context learning, where demonstrations go into the prompt and no gradients are computed.",
        "**Flattening patches in a different order from the conv kernel** when implementing patch embedding by hand (e.g. channels-last patches against a channels-first kernel). The result is still a linear map, so training \"works\" but pretrained weights are scrambled. Verify against the convolution as in this lesson.",
        "**Feeding a ViT an image at a new resolution without interpolating position embeddings.** The number of patches changes, the position table no longer matches; resample it in 2-D (as fine-tuning code does) instead of truncating.",
        "**Training a ViT from scratch on a small dataset and concluding Transformers do not work for vision.** The architecture lacks convolutional priors; use pretrained weights, strong augmentation, or a CNN/hybrid.",
        "**Quoting benchmark numbers from memory.** Parameter counts can be computed from a configuration — as here — but accuracies depend on data, recipe and evaluation protocol; cite the paper and the exact setting.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the three Transformer families and what distinguishes them?", a: "Encoder-only (BERT): bidirectional self-attention, trained with masked LM; produces representations. Decoder-only (GPT): causal self-attention, trained with next-token prediction; generates text. Encoder–decoder (T5, BART): bidirectional encoder plus causal decoder with cross-attention; maps an input sequence to an output sequence. The attention mask and the pretraining objective are the defining differences." },
      { t: "qa", level: "beginner", q: "What is the [CLS] token used for?", a: "A special token prepended to the input whose final hidden vector is used as a summary of the whole sequence: BERT feeds it to the NSP classifier during pretraining and to classification heads during fine-tuning; ViT classifies the image from it. Because attention is bidirectional, it can aggregate information from every position." },
      { t: "qa", level: "beginner", q: "How does a Vision Transformer turn an image into tokens?", a: "It cuts the image into non-overlapping $P\\times P$ patches (16×16 in ViT-B/16: 196 patches for 224×224), flattens each patch and maps it with a shared linear layer to a $D$-dimensional vector — implemented as a $P\\times P$ convolution with stride $P$ — then prepends a [CLS] token and adds learned position embeddings." },
      { t: "qa", level: "intermediate", q: "Explain BERT's 80/10/10 masking rule.", a: "15% of tokens are selected for prediction. Of those, 80% become [MASK], 10% a random token and 10% stay unchanged. If every selected token became [MASK], the model would learn that only [MASK] positions need good predictions, and [MASK] never appears at fine-tuning time. Random and unchanged tokens force it to model every token's identity from context, since any token might be a target." },
      { t: "qa", level: "intermediate", q: "Why did RoBERTa drop next-sentence prediction, and what else did it change?", a: "Its ablations found NSP did not help (and removing it could improve downstream results) when training on contiguous full-length text. It also used dynamic masking (a fresh mask each time a sequence is seen), much more data, larger batches and longer training, with byte-level BPE — showing BERT was significantly undertrained." },
      { t: "qa", level: "intermediate", q: "What is T5's span-corruption objective?", a: "Random spans covering about 15% of the input tokens (mean length 3) are each replaced by a unique sentinel token in the encoder input; the decoder must output each sentinel followed by the tokens it replaced. Targets are short, which makes pretraining efficient, and the format matches T5's text-to-text framing for all tasks." },
      { t: "qa", level: "intermediate", q: "What is in-context learning?", a: "The ability of a (large, decoder-only) language model to perform a task specified in its prompt — by an instruction and possibly a few input–output demonstrations — without any weight update. GPT-3 (Brown et al., 2020) studied it systematically as zero-, one- and few-shot evaluation. It emerges from next-token pretraining on diverse text." },
      { t: "qa", level: "advanced", q: "Show that ViT's patch embedding equals a convolution, and state the output size.", a: "A conv with $D$ filters of size $C\\times P\\times P$, stride $P$, no padding computes at $(i,j)$ the dot product of filter $d$ with the patch starting at $(iP, jP)$. Stride = kernel means patches do not overlap and tile the image, so the output at $(i,j)$ is $E^\\top\\mathrm{vec}(\\text{patch}_{ij}) + b$ with $E$'s columns = flattened filters. Output spatial size is $H/P\\times W/P$, i.e. $N = HW/P^2$ tokens. Numerically they agree to float round-off if the flattening order $(c,u,v)$ matches." },
      { t: "qa", level: "advanced", q: "Why is Swin's cost linear in image size, and how do shifted windows restore cross-window communication?", a: "Attention is restricted to $M\\times M$ windows, so each of the $hw$ tokens attends to $M^2$ tokens: $\\Omega = 4hwC^2 + 2M^2hwC$, linear in $hw$ versus $2(hw)^2C$ for global attention (13.8× cheaper at stage 1 for 224×224, 209× at 896×896). Alternate blocks shift the window partition by $\\lfloor M/2\\rfloor$ via a cyclic roll, so new windows contain tokens from up to four old windows; an attention mask prevents wrapped-around, non-adjacent tokens from interacting. Stacking regular and shifted blocks plus patch merging grows the receptive field to the whole image." },
      { t: "qa", level: "advanced", q: "When would you choose an encoder–decoder over a decoder-only model?", a: "When the task is a well-defined transformation with paired data and the input is long relative to the output (summarization, translation, speech-to-text): the encoder reads the input once bidirectionally, and decoding cost/KV cache involve only the output, with cross-attention to a fixed encoder memory. Decoder-only models are preferred when one general model must do many tasks via prompting, or for open-ended generation, and benefit from the largest-scale pretraining ecosystems." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { q: "Which attention mask does BERT use during pretraining?", options: ["Causal", "Bidirectional (only padding masked)", "Prefix-LM", "Sliding window"], answer: 1, explain: "BERT is encoder-only: every token attends to every non-padding token, so masked words are predicted from both sides." },
        { kind: "numeric", q: "Under BERT's masking rule, what fraction of all ordinary input tokens is replaced by a random token?", answer: 0.015, tolerance: 0.0001, explain: "$0.15\\times0.10 = 0.015$." },
        { kind: "numeric", q: "How many tokens (including [CLS]) does ViT with patch size 16 process for a 384×384 image?", answer: 577, tolerance: 0, explain: "$(384/16)^2 = 24^2 = 576$ patches + 1 [CLS] = 577." },
        { kind: "numeric", q: "How many parameters does the patch-embedding convolution of a ViT with $P=14$, RGB input and $D=1024$ have (weights + biases)?", answer: 603136, tolerance: 0, explain: "$3\\cdot14^2\\cdot1024 + 1024 = 602{,}112 + 1{,}024 = 603{,}136$." },
        { q: "Which statement about GPT-3 is correct?", options: ["It is fine-tuned separately for each benchmark task", "It is an encoder-only model", "It performs tasks from prompts with demonstrations, without weight updates (in-context learning)", "It is pretrained with span corruption"], answer: 2, explain: "Brown et al. (2020) evaluated GPT-3 zero-, one- and few-shot via the prompt." },
        { kind: "numeric", q: "For Swin with $C=96$ and window $M=7$, by what factor is window attention cheaper than global attention on a $28\\times28$ token grid? Use $(4C+2hw)/(4C+2M^2)$ (1 decimal).", answer: 4.1, tolerance: 0.05, explain: "$(384 + 1568)/(384 + 98) = 1952/482 = 4.05$." },
        { q: "What does Swin's patch merging do between stages?", options: ["Doubles resolution, halves channels", "Concatenates 2×2 neighbouring tokens and projects 4C → 2C: half resolution, double channels", "Averages all tokens into one", "Removes the [CLS] token"], answer: 1, explain: "It builds a CNN-like pyramid at strides 4, 8, 16, 32." },
        { q: "A prefix-LM mask with prefix length p differs from a causal mask by…", options: ["Blocking the diagonal", "Allowing all positions to see the whole prefix (top-left p×p block fully visible)", "Allowing all positions to see the future", "Hiding the prefix from the continuation"], answer: 1, explain: "Prefix tokens attend bidirectionally among themselves; continuation tokens remain causal." },
      ] },
      { t: "exercise", title: "Count parameters of BERT-large from its configuration", prompt: "BERT-large has 24 layers, $d=1024$, FFN width 4096, vocabulary 30,522, 512 positions, 2 segment types and a pooler. Using the per-layer count $12d^2+13d$ and the embedding/pooler terms from the math section, compute the total. Then check it with `BertModel(BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096))`.", hint: "Embeddings: $(30522+512+2)\\cdot d + 2d$ (LayerNorm). Pooler: $d^2+d$.", solution: [
        { t: "steps", items: [
          { title: "Per layer", tex: String.raw`12\cdot1024^2 + 13\cdot1024 = 12{,}582{,}912 + 13{,}312 = 12{,}596{,}224` },
          { title: "24 layers", tex: String.raw`24\cdot12{,}596{,}224 = 302{,}309{,}376` },
          { title: "Embeddings", tex: String.raw`31{,}036\cdot1024 + 2048 = 31{,}780{,}864 + 2{,}048 = 31{,}782{,}912` },
          { title: "Pooler", tex: String.raw`1024^2 + 1024 = 1{,}049{,}600` },
          { title: "Total", tex: String.raw`302{,}309{,}376 + 31{,}782{,}912 + 1{,}049{,}600 = 335{,}141{,}888 \approx 335\text{M}` },
        ] },
        { t: "p", text: "The commonly quoted “340M” for BERT-large is this number rounded (and, depending on the source, including the pretraining heads). Constructing the configuration with `BertModel` reproduces 335,141,888 exactly, by the same per-component reasoning that reproduced BERT-base's 109,482,240 in the PyTorch section." },
      ] },
      { t: "exercise", title: "Turn the tiny ViT into a Swin-style windowed model", prompt: "In `tiny_bert_and_vit.py`, the 4×4 grid of patch tokens attends globally. (a) Add an attention mask so each token attends only within its 2×2 window (and [CLS] attends to everything). (b) Count allowed (query, key) pairs among the 16 patch tokens before and after. (c) Why does accuracy typically change little here, and when would windowing matter?", hint: "Window id of a token at grid position (r, c) is (r // 2) * 2 + (c // 2). PyTorch's `nn.TransformerEncoder` accepts `mask` of shape (L, L) where `True` means **not allowed**.", solution: [
        { t: "code", lang: "text", code: String.raw`g = torch.arange(16)
win = (g // 4 // 2) * 2 + (g % 4 // 2)                  # window id of each patch token
allowed = torch.ones(17, 17, dtype=torch.bool)          # index 0 = [CLS]
allowed[1:, 1:] = win[:, None] == win[None, :]
self.register_buffer("block", ~allowed)                 # True = blocked
...
h = self.enc(t, mask=self.block)` },
        { t: "p", text: "(b) Globally: $16^2 = 256$ pairs. Windowed: 4 windows × $4^2$ = 64 pairs, a 4× reduction ($=hw/M^2 = 16/4$). (c) With 16 tokens, global attention is cheap and two layers of windowed attention plus the [CLS] token already mix information, so accuracy is similar. Windowing matters when the grid is large (thousands of tokens), where the quadratic cost dominates — then shifted windows are needed so information crosses window borders." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Encoder-only (BERT, RoBERTa): bidirectional mask, masked LM (15%, 80/10/10), [CLS] for sequence tasks; NSP was dropped by RoBERTa. Best for understanding tasks.",
        "Decoder-only (GPT-1/2/3): causal mask, next-token prediction on every position; generation and in-context learning (GPT-3: 175B parameters, few-shot prompting).",
        "Encoder–decoder (T5, BART): bidirectional encoder, causal decoder, cross-attention; T5 text-to-text with span corruption (15%, mean span 3, sentinels); BART denoising.",
        "Masks: bidirectional $n^2$ pairs, causal $n(n+1)/2$, prefix-LM adds $p(p-1)/2$, cross-attention $m\\times n_{\\text{enc}}$.",
        "Parameters per layer $12d^2 + 13d$; BERT-base 109,482,240, GPT-2 small 124,439,808, ViT-B/16 86,567,656 (computed and verified); GPT-3 ≈ $12Ld^2$ ≈ 174.6B.",
        "ViT: $N = HW/P^2$ patch tokens via a $P\\times P$ stride-$P$ convolution (= linear map on flattened patches), [CLS], learned positions; data-hungry without convolutional priors.",
        "Swin: $\\Omega(\\text{W-MSA}) = 4hwC^2 + 2M^2hwC$ — linear in image size; shifted windows connect neighbouring windows; patch merging builds a 4-stage pyramid.",
        "Choose by output type and data: labels/spans → encoder; open-ended text/many tasks → decoder; paired transformations → encoder–decoder; images → ViT with large data, Swin/CNN for dense, high-resolution tasks.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Transfer learning and fine-tuning](/learn/m12/transfer-learning) — how pretrained ViT/Swin (and CNN) backbones are adapted to new tasks: frozen features vs full fine-tuning.",
        "[NLP tasks](/learn/m13/nlp-tasks) — classification, NER, QA, summarization and translation, and which family solves each.",
        "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — the decoder-only recipe at scale: data, objectives and compute-optimal training.",
        "[Multimodal models and CLIP](/learn/m15/multimodal) — pairing a ViT image encoder with a text Transformer in a joint embedding space.",
        "[Self-supervised learning](/learn/m15/self-supervised) — label-free pretraining for vision, the context for MAE-style masked image modelling.",
      ] },
    ],
  },
};

export default lesson;
