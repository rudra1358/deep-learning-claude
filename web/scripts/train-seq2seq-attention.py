"""
train-seq2seq-attention.py
==========================

Purpose
-------
Trains the model behind the "Seq2Seq Attention Alignment" simulator (lesson m10/attention-intro):
a small GRU encoder-decoder with additive (Bahdanau) attention on a synthetic task whose correct
alignment is known in advance -- **reversing a string of lowercase letters** ("attention" ->
"noitnetta"). Output position i must copy input position T+1-i, so a well-trained model's attention
matrix should be an anti-diagonal.

For the "fixed-length context vector bottleneck" comparison it also trains the *same* architecture
WITHOUT attention: the decoder receives one fixed context vector c = [forward h_T ; backward h_1]
at every step instead of a step-dependent c_i. Both models are evaluated by exact-match accuracy
(greedy decoding) as a function of input length.

Architecture (identical for both runs except the context):
  source embedding   29 x 16
  encoder            bidirectional GRU, 24 units per direction  -> annotations h_j in R^48
  decoder init       s_0 = tanh(W_init <-h_1 + b_init)                    (Bahdanau et al. 2015)
  attention          e_ij = v^T tanh(W_s s_{i-1} + W_h h_j),  alpha_i = softmax_j(e_i),
                     c_i = sum_j alpha_ij h_j                              (attention dim 32)
  decoder            GRUCell, input [emb(y_{i-1}) ; c_i] (16+48), 48 units  (PyTorch gate convention)
  output             logits_i = W_o [s_i ; c_i] + b_o  over 29 symbols

Vocabulary: 0 <pad>, 1 <sos>, 2 <eos>, 3..28 = 'a'..'z'.
Training: lengths 1..20 uniformly, fresh random strings every batch (no fixed training set, so no
overfitting), teacher forcing, cross-entropy, Adam, gradient clipping. Evaluation: 200 fresh strings
per length 1..25 (21..25 are longer than anything seen in training).

Output
------
  web/public/models/seq2seq-attention.json  (attention model weights as base64 little-endian float32,
  vocab, config, measured accuracy-by-length for both models, and verification vectors -- every
  intermediate of greedy decoding for a few inputs -- used to test the TypeScript engine
  web/lib/sim/seq2seq.ts).

How to run (from the repository root)
-------------------------------------
  .venv/Scripts/python.exe web/scripts/train-seq2seq-attention.py

Requires torch (CPU is fine) and numpy. Deterministic seeds (torch.manual_seed(0), numpy seed 0).
Runtime: about 12 minutes on a laptop CPU with 4 threads (attention model ~8-12 min, no-attention
model ~5 min; measured 735 s + 293 s while other jobs shared the CPU).

Measured results of this script (exact match, 200 strings per length):
  attention:     100% for every length 1..21, then 99.0% (22), 90.5% (23), 61.5% (24), 27.5% (25)
  no attention:  100% (1-3), 93.5% (6), 63.5% (8), 30.5% (10), 3.5% (13), 0% from length 16 on
Output file size: ~181 KB.
"""

import base64
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --------------------------------------------------------------------------- configuration
SEED = 0
CHARS = "abcdefghijklmnopqrstuvwxyz"
PAD, SOS, EOS = 0, 1, 2
ITOS = ["<pad>", "<sos>", "<eos>"] + list(CHARS)
STOI = {c: i for i, c in enumerate(ITOS)}
V = len(ITOS)                      # 29 symbols
EMB, ENC_H, DEC_H, ATT = 16, 24, 48, 32
MIN_LEN, MAX_LEN = 1, 20           # training lengths
EVAL_MAX_LEN, EVAL_N = 25, 200     # evaluation: lengths 1..25, 200 strings each
STEPS, BATCH, LR = 6000, 128, 3e-3
OUT = Path(__file__).resolve().parent.parent / "public" / "models" / "seq2seq-attention.json"


def encode(s):
    return [STOI[c] for c in s]


def make_batch(rng, n, lo=MIN_LEN, hi=MAX_LEN, length=None):
    """Random lowercase strings; returns padded src (B,T), lengths, padded tgt (B,T+1) ending in EOS."""
    lens = np.full(n, length) if length is not None else rng.integers(lo, hi + 1, size=n)
    T = int(lens.max())
    src = np.full((n, T), PAD, dtype=np.int64)
    tgt = np.full((n, T + 1), PAD, dtype=np.int64)
    strs = []
    for b, L in enumerate(lens):
        ids = rng.integers(3, V, size=L)
        src[b, :L] = ids
        tgt[b, :L] = ids[::-1]
        tgt[b, L] = EOS
        strs.append("".join(ITOS[i] for i in ids))
    return torch.tensor(src), torch.tensor(lens), torch.tensor(tgt), strs


class Seq2Seq(nn.Module):
    def __init__(self, attention=True):
        super().__init__()
        self.attention = attention
        self.src_emb = nn.Embedding(V, EMB, padding_idx=PAD)
        self.encoder = nn.GRU(EMB, ENC_H, batch_first=True, bidirectional=True)
        self.init = nn.Linear(ENC_H, DEC_H)
        if attention:
            self.W_s = nn.Linear(DEC_H, ATT, bias=False)
            self.W_h = nn.Linear(2 * ENC_H, ATT, bias=False)
            self.v = nn.Linear(ATT, 1, bias=False)
        self.tgt_emb = nn.Embedding(V, EMB)
        self.cell = nn.GRUCell(EMB + 2 * ENC_H, DEC_H)
        self.out = nn.Linear(DEC_H + 2 * ENC_H, V)

    def encode(self, src, lens):
        x = self.src_emb(src)
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        H, h_n = self.encoder(packed)
        H, _ = pad_packed_sequence(H, batch_first=True, total_length=src.shape[1])
        mask = torch.arange(src.shape[1])[None, :] < lens[:, None]       # (B, T) real positions
        s0 = torch.tanh(self.init(h_n[1]))                               # backward state at position 1
        c_fixed = torch.cat([h_n[0], h_n[1]], dim=-1)                    # [->h_T ; <-h_1]
        keys = self.W_h(H) if self.attention else None                   # W_h h_j, precomputed once
        return H, mask, s0, c_fixed, keys

    def step(self, y_prev, s, H, mask, c_fixed, keys):
        if self.attention:
            e = self.v(torch.tanh(self.W_s(s)[:, None, :] + keys)).squeeze(-1)   # (B, T)
            e = e.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(e, dim=-1)
            c = torch.bmm(alpha[:, None, :], H).squeeze(1)                       # (B, 2*ENC_H)
        else:
            e, alpha, c = None, None, c_fixed
        s_new = self.cell(torch.cat([self.tgt_emb(y_prev), c], dim=-1), s)
        logits = self.out(torch.cat([s_new, c], dim=-1))
        return logits, s_new, e, alpha, c

    def forward(self, src, lens, tgt):
        H, mask, s, c_fixed, keys = self.encode(src, lens)
        y = torch.full((src.shape[0],), SOS, dtype=torch.long)
        all_logits = []
        for i in range(tgt.shape[1]):
            logits, s, *_ = self.step(y, s, H, mask, c_fixed, keys)
            all_logits.append(logits)
            y = tgt[:, i]                                                      # teacher forcing
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def greedy(self, src, lens, max_steps):
        H, mask, s, c_fixed, keys = self.encode(src, lens)
        y = torch.full((src.shape[0],), SOS, dtype=torch.long)
        outs = []
        for _ in range(max_steps):
            logits, s, *_ = self.step(y, s, H, mask, c_fixed, keys)
            y = logits.argmax(-1)
            outs.append(y)
        return torch.stack(outs, dim=1)


def decode_ids(row):
    out = []
    for t in row.tolist():
        if t == EOS:
            return "".join(out), True
        out.append(ITOS[t] if t >= 3 else "?")
    return "".join(out), False


def accuracy_by_length(model, seed=123):
    rng = np.random.default_rng(seed)
    res = []
    model.eval()
    for L in range(1, EVAL_MAX_LEN + 1):
        src, lens, tgt, strs = make_batch(rng, EVAL_N, length=L)
        pred = model.greedy(src, lens, max_steps=EVAL_MAX_LEN + 5)
        ok = 0
        for b in range(EVAL_N):
            text, ended = decode_ids(pred[b])
            ok += ended and text == strs[b][::-1]
        res.append(ok / EVAL_N)
    model.train()
    return res


def train(attention):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Seq2Seq(attention)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / STEPS)))
    t0 = time.time()
    for k in range(1, STEPS + 1):
        src, lens, tgt, _ = make_batch(rng, BATCH)
        logits = model(src, lens, tgt)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if k % 1000 == 0:
            print(f"  [{'attn' if attention else 'no-attn'}] step {k:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, time.time() - t0


def b64(t):
    a = t.detach().cpu().numpy().astype("<f4")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


@torch.no_grad()
def trace(model, text, max_steps=EVAL_MAX_LEN + 5):
    """Every intermediate of greedy decoding for one input (batch of 1), as plain lists."""
    src = torch.tensor([encode(text)])
    lens = torch.tensor([len(text)])
    H, mask, s, c_fixed, keys = model.encode(src, lens)
    y = torch.tensor([SOS])
    s0, steps = s[0].tolist(), []
    for _ in range(max_steps):
        logits, s, e, alpha, c = model.step(y, s, H, mask, c_fixed, keys)
        probs = torch.softmax(logits, -1)
        y = logits.argmax(-1)
        steps.append({  # s_{i-1} is not stored: it is s0 for step 1 and the previous step's "s" afterwards
            "e": e[0].tolist(), "alpha": alpha[0].tolist(),
            "context": c[0].tolist(), "s": s[0].tolist(), "probs": probs[0].tolist(), "token": int(y),
        })
        if int(y) == EOS:
            break
    out, _ = decode_ids(torch.tensor([st["token"] for st in steps]))
    return {"input": text, "H": H[0].tolist(), "s0": s0, "output": out, "steps": steps}


def round_floats(x, nd=5):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [round_floats(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: round_floats(v, nd) for k, v in x.items()}
    return x


def main():
    torch.set_num_threads(4)
    results = {}
    for attention in (True, False):
        name = "attention" if attention else "no-attention"
        print(f"training {name} ...", flush=True)
        model, secs = train(attention)
        acc = accuracy_by_length(model)
        n_params = sum(p.numel() for p in model.parameters())
        results[name] = {"model": model, "acc": acc, "params": n_params, "seconds": secs}
        print(f"  {name}: {n_params} params, {secs:.0f}s")
        print("  exact match by length:", " ".join(f"{L}:{a:.2f}" for L, a in enumerate(acc, 1)), flush=True)

    model = results["attention"]["model"].eval()
    sd = model.state_dict()
    weights = {
        "src_emb": b64(sd["src_emb.weight"]),
        # nn.GRU, gate order (r, z, n); _reverse = backward direction
        "enc_f_Wih": b64(sd["encoder.weight_ih_l0"]), "enc_f_Whh": b64(sd["encoder.weight_hh_l0"]),
        "enc_f_bih": b64(sd["encoder.bias_ih_l0"]), "enc_f_bhh": b64(sd["encoder.bias_hh_l0"]),
        "enc_b_Wih": b64(sd["encoder.weight_ih_l0_reverse"]), "enc_b_Whh": b64(sd["encoder.weight_hh_l0_reverse"]),
        "enc_b_bih": b64(sd["encoder.bias_ih_l0_reverse"]), "enc_b_bhh": b64(sd["encoder.bias_hh_l0_reverse"]),
        "init_W": b64(sd["init.weight"]), "init_b": b64(sd["init.bias"]),
        "W_s": b64(sd["W_s.weight"]), "W_h": b64(sd["W_h.weight"]), "v": b64(sd["v.weight"]),
        "tgt_emb": b64(sd["tgt_emb.weight"]),
        "dec_Wih": b64(sd["cell.weight_ih"]), "dec_Whh": b64(sd["cell.weight_hh"]),
        "dec_bih": b64(sd["cell.bias_ih"]), "dec_bhh": b64(sd["cell.bias_hh"]),
        "out_W": b64(sd["out.weight"]), "out_b": b64(sd["out.bias"]),
    }
    verification = [round_floats(trace(model, t)) for t in ["attention", "zyx"]]   # 5 decimals keeps the file < 200 KB
    for v in verification:
        print(f"  verify: {v['input']!r} -> {v['output']!r}")
    data = {
        "task": "Reverse a string of lowercase letters (a-z), lengths 1-20 in training.",
        "config": {"emb": EMB, "encHidden": ENC_H, "decHidden": DEC_H, "att": ATT, "vocabSize": V,
                   "pad": PAD, "sos": SOS, "eos": EOS, "minLen": MIN_LEN, "maxLen": MAX_LEN,
                   "maxDecodeSteps": EVAL_MAX_LEN + 5},
        "vocab": ITOS,
        "training": {"steps": STEPS, "batch": BATCH, "lr": LR, "optimizer": "Adam + cosine decay, clip 1.0",
                     "seed": SEED, "teacherForcing": True},
        "params": {"attention": results["attention"]["params"], "noAttention": results["no-attention"]["params"]},
        "accuracyByLength": {"lengths": list(range(1, EVAL_MAX_LEN + 1)), "samplesPerLength": EVAL_N,
                             "attention": results["attention"]["acc"], "noAttention": results["no-attention"]["acc"]},
        "weights": weights,
        "verification": verification,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
