import type { Lesson } from "@/lib/types";

/* ------------------------------------------------------------------ code blocks (executed; outputs are real) */

const SCRATCH = String.raw`import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---- 1. SFT: next-token loss on the RESPONSE tokens only ----
V, T, n_prompt = 12, 10, 6                        # 6 prompt tokens followed by 4 response tokens
tokens = rng.integers(0, V, T)
logits = rng.normal(size=(T - 1, V))              # model outputs at positions 0..8 predict tokens 1..9
targets = tokens[1:]
nll = -log_softmax(logits)[np.arange(T - 1), targets]
mask = np.arange(1, T) >= n_prompt                # target position >= n_prompt  <=>  target is a response token
print("per-position NLL:", nll)
print(f"loss over all tokens {nll.mean():.4f} | response-only loss {nll[mask].mean():.4f} ({mask.sum()} tokens)")
grad = np.exp(log_softmax(logits)); grad[np.arange(T - 1), targets] -= 1
grad *= mask[:, None] / mask.sum()                # d(masked loss)/d logits
print("gradient norm per position:", np.linalg.norm(grad, axis=1))

# ---- 2. reward model: Bradley-Terry fit from pairwise preferences ----
K, F_ = 6, 4                                      # 6 candidate responses, 4 features each
phi = rng.normal(size=(K, F_))
w_true = np.array([1.5, -1.0, 0.5, 0.0])
r_true = phi @ w_true                             # the hidden "human" reward
i, j = rng.integers(0, K, (2, 4000))
keep = i != j
i, j = i[keep], j[keep]
y = rng.random(len(i)) < sigmoid(r_true[i] - r_true[j])   # 1 if annotator prefers i over j
chosen, rejected = np.where(y, i, j), np.where(y, j, i)
w = np.zeros(F_)
for step in range(2000):
    margin = (phi[chosen] - phi[rejected]) @ w
    loss = -np.log(sigmoid(margin)).mean()        # Bradley-Terry negative log-likelihood
    g = -((1 - sigmoid(margin))[:, None] * (phi[chosen] - phi[rejected])).mean(0)
    w -= 0.5 * g
r_hat = phi @ w
print(f"BT loss {loss:.4f} | learned w {w} vs true {w_true}")
print("true reward   :", r_true - r_true.mean())
print("learned reward:", r_hat - r_hat.mean(), " (rewards are only defined up to a constant)")

# ---- 3. the KL-regularised RLHF objective and its closed-form optimum ----
beta = 0.5
pi_ref = np.array([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
r = r_true
pi_star = pi_ref * np.exp(r / beta); pi_star /= pi_star.sum()      # pi* ∝ pi_ref · exp(r / beta)


def objective(pi):                                 # E_pi[r] - beta * KL(pi || pi_ref)
    return (pi * r).sum() - beta * (pi * np.log(pi / pi_ref)).sum()


theta = np.log(pi_ref).copy()                      # policy logits, start at the reference
for step in range(3000):                           # exact policy-gradient ascent on the objective
    pi = np.exp(log_softmax(theta))
    adv = r - beta * (np.log(pi / pi_ref) + 1)     # d objective / d pi
    theta += 0.2 * pi * (adv - (pi * adv).sum())   # chain rule through the softmax
pi_rl = np.exp(log_softmax(theta))
print("pi_ref :", pi_ref)
print("pi*    :", pi_star, f" objective {objective(pi_star):.6f}")
print("pi (PG):", pi_rl, f" objective {objective(pi_rl):.6f}")

# ---- 4. DPO: fit the policy directly to the preference pairs ----
theta = np.log(pi_ref).copy()
for step in range(3000):
    logr = log_softmax(theta) - np.log(pi_ref)     # log pi/pi_ref for every response
    h = beta * (logr[chosen] - logr[rejected])     # implicit reward margin
    loss = -np.log(sigmoid(h)).mean()
    coef = -(1 - sigmoid(h)) * beta / len(h)       # d loss / d h  times d h / d logr
    g_logr = np.bincount(chosen, coef, K) - np.bincount(rejected, coef, K)
    pi = np.exp(log_softmax(theta))
    theta -= 2.0 * (g_logr - pi * g_logr.sum())    # d logr / d theta = I - 1 pi^T
pi_dpo = np.exp(log_softmax(theta))
implied = beta * (log_softmax(theta) - np.log(pi_ref))
print(f"DPO loss {loss:.4f}")
print("pi (DPO):", pi_dpo)
print("implied reward beta*log(pi/pi_ref), centred:", implied - implied.mean())
print(f"KL(pi_DPO || pi*) = {(pi_dpo * np.log(pi_dpo / pi_star)).sum():.5f}")`;

const SCRATCH_OUT = String.raw`per-position NLL: [4.835 2.475 2.931 3.333 2.159 3.463 2.148 1.426 3.153]
loss over all tokens 2.8804 | response-only loss 2.5477 (4 tokens)
gradient norm per position: [0.    0.    0.    0.    0.    0.261 0.243 0.207 0.255]
BT loss 0.4991 | learned w [ 1.483 -0.992  0.491 -0.038] vs true [ 1.5 -1.   0.5  0. ]
true reward   : [-1.284  0.314 -0.037 -0.116 -0.881  2.005]
learned reward: [-1.303  0.294 -0.006 -0.069 -0.862  1.945]  (rewards are only defined up to a constant)
pi_ref : [0.3  0.25 0.2  0.12 0.08 0.05]
pi*    : [0.006 0.132 0.052 0.027 0.004 0.778]  objective 1.146151
pi (PG): [0.007 0.132 0.052 0.027 0.004 0.778]  objective 1.146134
DPO loss 0.4985
pi (DPO): [0.008 0.134 0.057 0.037 0.004 0.761]
implied reward beta*log(pi/pi_ref), centred: [-1.24   0.277 -0.038  0.002 -0.951  1.95 ]
KL(pi_DPO || pi*) = 0.00224`;

const TORCH = String.raw`import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

# ---- 1. SFT loss masking with ignore_index ----
logits = torch.randn(1, 9, 12)                       # (batch, positions, vocab)
labels = torch.tensor([[-100, -100, -100, -100, -100, 7, 2, 11, 4]])   # prompt targets ignored
masked = F.cross_entropy(logits.view(-1, 12), labels.view(-1), ignore_index=-100)
manual = F.cross_entropy(logits[0, 5:], labels[0, 5:])
print(f"SFT loss with ignore_index {masked.item():.4f} == loss on response tokens only {manual.item():.4f}")

# ---- a toy "assistant": 4 prompts, each with the same 6 candidate responses ----
P, K, beta = 4, 6, 0.5
r_true = torch.randn(P, K) * 1.5                     # hidden human preference (reward) per prompt/response


def make_policy():
    return nn.Sequential(nn.Linear(P, 32), nn.Tanh(), nn.Linear(32, K))   # one-hot prompt -> response logits


onehot = torch.eye(P)
policy = make_policy()
ref = copy.deepcopy(policy).requires_grad_(False)    # frozen reference (in practice: the SFT model)
logp_ref = F.log_softmax(ref(onehot), -1)

# ---- 2. preference data and a Bradley-Terry reward model ----
n = 6000
p_idx = torch.randint(0, P, (n,))
a, b = torch.randint(0, K, (2, n))
keep = a != b
p_idx, a, b = p_idx[keep], a[keep], b[keep]
pref_a = torch.rand(len(a)) < torch.sigmoid(r_true[p_idx, a] - r_true[p_idx, b])
chosen, rejected = torch.where(pref_a, a, b), torch.where(pref_a, b, a)
n_train = int(0.8 * len(chosen))

reward_model = nn.Sequential(nn.Linear(P + K, 32), nn.ReLU(), nn.Linear(32, 1))


def rm(p, k):
    return reward_model(torch.cat([F.one_hot(p, P).float(), F.one_hot(k, K).float()], -1)).squeeze(-1)


opt = torch.optim.Adam(reward_model.parameters(), lr=1e-2)
for epoch in range(300):
    tr = slice(0, n_train)
    loss = -F.logsigmoid(rm(p_idx[tr], chosen[tr]) - rm(p_idx[tr], rejected[tr])).mean()
    opt.zero_grad(); loss.backward(); opt.step()
te = slice(n_train, None)
with torch.no_grad():
    acc_rm = (rm(p_idx[te], chosen[te]) > rm(p_idx[te], rejected[te])).float().mean()
    acc_true = (r_true[p_idx[te], chosen[te]] > r_true[p_idx[te], rejected[te]]).float().mean()
    R = torch.stack([rm(torch.full((K,), p), torch.arange(K)) for p in range(P)])   # learned reward table
print(f"reward model: train BT loss {loss.item():.4f}, held-out pair accuracy {acc_rm:.3f} (true reward gets {acc_true:.3f})")


def optimum(r):                                     # pi*(y|x) ∝ pi_ref(y|x) exp(r(x,y)/beta)
    return F.softmax(logp_ref + r / beta, -1)


def kl(p, q):
    return (p * (p.log() - q.log())).sum(-1).mean().item()


# ---- 3. RLHF with PPO: reward-model score minus a KL penalty to the reference ----
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
for it in range(300):
    with torch.no_grad():
        prompts = torch.randint(0, P, (256,))
        logp_old = F.log_softmax(policy(onehot[prompts]), -1)
        y = torch.multinomial(logp_old.exp(), 1).squeeze(1)              # sample responses
        lp_old = logp_old[torch.arange(256), y]
        kl_pen = lp_old - logp_ref[prompts, y]                            # per-sample log(pi/pi_ref)
        reward = R[prompts, y] - beta * kl_pen                            # InstructGPT-style shaped reward
        base = torch.zeros(P).index_add_(0, prompts, reward) / torch.bincount(prompts, minlength=P).clamp(min=1)
        adv = reward - base[prompts]                                      # per-prompt mean baseline
    for epoch in range(4):                                                # several PPO epochs per batch
        lp = F.log_softmax(policy(onehot[prompts]), -1)[torch.arange(256), y]
        ratio = (lp - lp_old).exp()
        obj = torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv)         # clipped surrogate, eps = 0.2
        opt.zero_grad(); (-obj.mean()).backward(); opt.step()
with torch.no_grad():
    pi_ppo = F.softmax(policy(onehot), -1)
print(f"PPO : KL(pi || pi*_RM) = {kl(pi_ppo, optimum(R)):.4f}, KL(pi || pi*_true) = {kl(pi_ppo, optimum(r_true)):.4f},"
      f" KL to reference = {kl(pi_ppo, logp_ref.exp()):.4f}")

# ---- 4. DPO: no reward model, no sampling — a classification loss on the preference pairs ----
policy = copy.deepcopy(ref).requires_grad_(True)
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
tr = slice(0, n_train)
for epoch in range(500):
    logp = F.log_softmax(policy(onehot[p_idx[tr]]), -1)
    lr_c = logp.gather(1, chosen[tr, None]).squeeze(1) - logp_ref[p_idx[tr], chosen[tr]]
    lr_r = logp.gather(1, rejected[tr, None]).squeeze(1) - logp_ref[p_idx[tr], rejected[tr]]
    loss = -F.logsigmoid(beta * (lr_c - lr_r)).mean()                     # the DPO loss
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pi_dpo = F.softmax(policy(onehot), -1)
    implicit = beta * (pi_dpo.log() - logp_ref)                           # DPO's implicit reward
    acc_dpo = ((implicit[p_idx[te], chosen[te]] > implicit[p_idx[te], rejected[te]]).float().mean())
print(f"DPO : loss {loss.item():.4f}, KL(pi || pi*_true) = {kl(pi_dpo, optimum(r_true)):.4f},"
      f" implicit-reward pair accuracy {acc_dpo:.3f}")
print("prompt 0  pi_ref :", [round(v, 3) for v in logp_ref[0].exp().tolist()])
print("prompt 0  pi*    :", [round(v, 3) for v in optimum(r_true)[0].tolist()])
print("prompt 0  PPO    :", [round(v, 3) for v in pi_ppo[0].tolist()])
print("prompt 0  DPO    :", [round(v, 3) for v in pi_dpo[0].tolist()])`;

const TORCH_OUT = String.raw`SFT loss with ignore_index 2.8742 == loss on response tokens only 2.8742
reward model: train BT loss 0.5432, held-out pair accuracy 0.720 (true reward gets 0.720)
PPO : KL(pi || pi*_RM) = 0.0005, KL(pi || pi*_true) = 0.0053, KL to reference = 0.8871
DPO : loss 0.5432, KL(pi || pi*_true) = 0.0056, implicit-reward pair accuracy 0.720
prompt 0  pi_ref : [0.154, 0.135, 0.146, 0.219, 0.197, 0.15]
prompt 0  pi*    : [0.169, 0.015, 0.749, 0.02, 0.012, 0.035]
prompt 0  PPO    : [0.199, 0.011, 0.729, 0.016, 0.01, 0.036]
prompt 0  DPO    : [0.188, 0.01, 0.742, 0.016, 0.01, 0.035]`;

const KERAS = String.raw`import tensorflow as tf


def dpo_loss(policy_logp_chosen, policy_logp_rejected, ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """All inputs: (batch,) summed log-probabilities of the response tokens."""
    margin = beta * ((policy_logp_chosen - ref_logp_chosen) - (policy_logp_rejected - ref_logp_rejected))
    return -tf.reduce_mean(tf.math.log_sigmoid(margin))


def reward_model_loss(r_chosen, r_rejected):
    return -tf.reduce_mean(tf.math.log_sigmoid(r_chosen - r_rejected))   # Bradley-Terry


print(dpo_loss(tf.constant([-12.0]), tf.constant([-15.0]), tf.constant([-13.0]), tf.constant([-14.0])))`;

/* ------------------------------------------------------------------ lesson */

const lesson: Lesson = {
  id: "m20/alignment",
  title: "Instruction Tuning, RLHF and DPO",
  summary:
    "How a pretrained next-token predictor becomes an assistant: supervised fine-tuning on demonstrations with the prompt tokens masked out, Bradley–Terry reward models learned from human preference pairs, RLHF with PPO and a KL penalty to the reference model, and Direct Preference Optimization, derived from the closed-form optimum of the KL-regularised objective. Each piece is implemented on a toy problem where the exact answer is known.",
  minutes: 75,
  level: "research",
  prerequisites: ["m20/llm-pretraining", "m15/deep-rl", "m05/classification-losses", "m02/information-theory"],
  next: ["m20/llm-inference", "m19/scaling-efficiency", "m19/robustness-safety"],
  terms: ["instruction-tuning", "supervised-fine-tuning", "loss-masking", "reward-model", "bradley-terry-model", "rlhf", "kl-penalty", "reference-model", "dpo", "reward-hacking"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "A pretrained language model ([Pretraining](/learn/m20/llm-pretraining)) is a superb **continuer of documents**: asked a question, it may answer, or it may continue with five more questions, because that is what follows a question on many web pages. **Alignment** (also called post-training) turns it into a model that follows instructions, is helpful, declines harmful requests and admits uncertainty — properties that are hard to write as a loss on raw text." },
      { t: "p", text: "The standard pipeline, popularised by InstructGPT (Ouyang et al., 2022, *Training language models to follow instructions with human feedback*), has three stages. (1) **Supervised fine-tuning (SFT)** or **instruction tuning**: continue training on (prompt, ideal response) demonstrations, with the loss computed on response tokens only. (2) **Reward modelling**: humans compare two responses to the same prompt; a model learns a scalar score that predicts which one they prefer. (3) **Reinforcement learning from human feedback (RLHF)**: optimise the policy to maximise the learned reward with PPO, while a **KL penalty** keeps it close to the SFT model. The idea of learning a reward from pairwise human preferences and optimising it with RL comes from Christiano et al. (2017, *Deep reinforcement learning from human preferences*) and was applied to language by Ziegler et al. (2019) and Stiennon et al. (2020)." },
      { t: "p", text: "**Direct Preference Optimization (DPO)** (Rafailov et al., 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*) showed that stages (2) and (3) can be collapsed into a single classification-style loss on preference pairs: no separate reward model, no sampling, no PPO. Understanding why requires the closed-form solution of the KL-regularised objective, which is the mathematical heart of this lesson." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — the base model and its next-token loss, which SFT simply continues.",
        "[Deep Reinforcement Learning](/learn/m15/deep-rl) — policies, policy gradients, advantages and the PPO clipped objective.",
        "[Classification Losses](/learn/m05/classification-losses) — logistic (sigmoid) cross-entropy; the reward-model and DPO losses are both of this form.",
        "[Information Theory](/learn/m02/information-theory) — KL divergence $\\mathrm{KL}(\\pi\\|\\pi_{\\text{ref}}) = \\mathbb E_\\pi[\\log \\pi/\\pi_{\\text{ref}}]$.",
        "[Decoding](/learn/m20/decoding) — sequence log-probabilities $\\log\\pi(y\\mid x) = \\sum_t \\log \\pi(y_t\\mid x,y_{<t})$, which every loss here uses.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "Training a new chef", text: "A culinary graduate (the pretrained model) knows every technique but cooks whatever comes to mind. **SFT**: they watch a head chef plate fifty signature dishes and learn to imitate them. **Reward model**: diners compare pairs of plates (\"left is better\"), and a food critic learns to predict their verdicts with a score. **RLHF**: the chef cooks, the critic scores, and the chef adjusts toward higher scores — but the head chef insists they keep cooking recognisably like the restaurant (the KL penalty), because a chef who chases the critic's score alone discovers that the critic loves absurd amounts of truffle oil (**reward hacking**). **DPO** skips the critic: the chef directly studies the diners' pairwise verdicts and shifts probability from the losing plate to the winning one, measured relative to how they cooked before." },
      { t: "p", text: "Two ideas carry most of the weight. First, **comparisons are easier than demonstrations**: people disagree on what the perfect answer is, but agree much more reliably on which of two answers is better. Second, the optimisation must stay **anchored** to the reference model: the reward model is only trustworthy on responses resembling those it was trained on, so the policy is penalised for drifting far from it." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "Stage 1: supervised fine-tuning" },
      { t: "steps", items: [
        { title: "Data", text: "Tens of thousands of (prompt, response) pairs written or curated by people, formatted with a chat template (special tokens marking system, user and assistant turns)." },
        { title: "Loss masking", text: "Concatenate prompt and response into one sequence, run the usual next-token cross-entropy, but ignore positions whose target is a prompt token (label $-100$ in PyTorch). The model learns to produce responses, not to model user prompts." },
        { title: "Training", text: "A few epochs at a small learning rate; the result is the SFT policy $\\pi_{\\text{SFT}}$, which also serves as the **reference** $\\pi_{\\text{ref}}$ and the starting point for the later stages." },
      ] },
      { t: "h", text: "Stage 2: reward model" },
      { t: "steps", items: [
        { title: "Comparisons", text: "For a prompt $x$, sample two (or more) responses from $\\pi_{\\text{SFT}}$; a labeller marks the preferred one $y_w$ (\"winner\") and the other $y_l$." },
        { title: "Architecture", text: "Usually the SFT model with its vocabulary head replaced by a scalar head, read at the last token: $r_\\phi(x, y) \\in \\mathbb R$." },
        { title: "Loss", text: "Bradley–Terry: maximise $\\log\\sigma(r_\\phi(x,y_w) - r_\\phi(x,y_l))$. Only differences matter, so rewards are defined up to a per-prompt constant (often normalised to mean zero)." },
      ] },
      { t: "h", text: "Stage 3: RLHF with PPO" },
      { t: "steps", items: [
        { title: "Rollout", text: "Sample prompts; generate responses from the current policy $\\pi_\\theta$ (at temperature 1)." },
        { title: "Score", text: "Reward $= r_\\phi(x,y) - \\beta \\log\\frac{\\pi_\\theta(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}$; the KL term is typically applied per token, the reward-model score at the final token." },
        { title: "Advantage", text: "Subtract a baseline (a learned value function per token in InstructGPT, or the group mean in simpler variants) to reduce variance." },
        { title: "Update", text: "Several epochs of the PPO clipped surrogate (Schulman et al., 2017) on the batch, then sample again. Four models are in memory: policy, reference, reward model and value model." },
      ] },
      { t: "h", text: "DPO instead of stages 2 and 3" },
      { t: "steps", items: [
        { title: "Data", text: "The same preference triples $(x, y_w, y_l)$." },
        { title: "Log-ratios", text: "Compute $\\log\\pi_\\theta(y\\mid x)$ and $\\log\\pi_{\\text{ref}}(y\\mid x)$ for both responses (sum of token log-probs over the response tokens)." },
        { title: "Loss", text: "$-\\log\\sigma\\big(\\beta[(\\log\\pi_\\theta - \\log\\pi_{\\text{ref}})(y_w) - (\\log\\pi_\\theta - \\log\\pi_{\\text{ref}})(y_l)]\\big)$ and backpropagate into the policy only. Two models in memory, no generation during training." },
      ] },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "SFT objective" },
      { t: "math", tex: String.raw`\mathcal L_{\text{SFT}}(\theta) = -\frac{1}{\sum_t m_t}\sum_{t} m_t \log \pi_\theta(s_{t+1} \mid s_{\le t}), \qquad m_t = \mathbb 1[s_{t+1} \text{ is a response token}]`, caption: "$s$ = prompt and response concatenated; $m_t$ = loss mask. Normalising by the number of response tokens (not all tokens) keeps the loss scale independent of prompt length." },
      { t: "h", text: "Bradley–Terry reward model" },
      { t: "p", text: String.raw`Bradley and Terry (1952) model the probability that item $i$ beats item $j$ as $\sigma(r_i - r_j)$ with $\sigma(u) = 1/(1+e^{-u})$. For responses:` },
      { t: "math", tex: String.raw`P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big), \qquad \mathcal L_{\text{RM}}(\phi) = -\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]` },
      { t: "p", text: String.raw`Worked example: $r_\phi(x,y_w) = 1.2$, $r_\phi(x,y_l) = 0.3$. Then $P = \sigma(0.9) = 0.710950$ and the loss is $-\ln 0.710950 = 0.341154$. The gradient with respect to the margin is $-(1-\sigma(0.9)) = -0.289050$: the model is pushed to widen the gap, less so the more confident it already is.` },
      { t: "h", text: "The KL-regularised objective and its closed-form optimum" },
      { t: "math", tex: String.raw`\max_{\pi}\; \mathbb E_{x}\Big[\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big]`, caption: "$\\beta > 0$ trades reward against staying close to the reference policy." },
      { t: "p", text: String.raw`For each prompt this is a maximisation over a distribution. Write the objective as $\sum_y \pi(y)\big[r(y) - \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\big]$ and add a Lagrange multiplier for $\sum_y\pi(y)=1$. Setting the derivative with respect to $\pi(y)$ to zero gives $r(y) - \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} - \beta - \lambda = 0$, hence` },
      { t: "math", tex: String.raw`\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big), \qquad Z(x) = \sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)`, caption: "The reference distribution, tilted by the exponentiated reward. Small β → greedy on the reward; large β → stays at π_ref." },
      { t: "p", text: String.raw`Worked example: three responses with $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$, rewards $r = (0, 1, 2)$, $\beta = 1$. Unnormalised: $(0.5,\ 0.3e^{1},\ 0.2e^{2}) = (0.5,\ 0.815485,\ 1.477811)$, $Z = 2.793296$, so $\pi^* = (0.179,\ 0.291944,\ 0.529056)$. The best response gains probability, but the others keep some — the KL term prevents collapse onto a single answer.` },
      { t: "h", text: "Deriving DPO" },
      { t: "steps", items: [
        { title: "Invert the optimum", tex: String.raw`r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)` },
        { title: "Insert into Bradley–Terry: Z(x) cancels in the difference", tex: String.raw`P(y_w\succ y_l\mid x) = \sigma\Big(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)` },
        { title: "Fit the policy by maximum likelihood on the preferences", tex: String.raw`\mathcal L_{\text{DPO}}(\theta) = -\mathbb E\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]` },
      ] },
      { t: "p", text: String.raw`DPO is the reward-model loss with the reward **parameterised by the policy**: $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, the "implicit reward". If the preferences follow Bradley–Terry with true reward $r$, the population minimiser has $\hat r_\theta = r + c(x)$, i.e. $\pi_\theta = \pi^*$ — the same policy RLHF aims for, reached without sampling.` },
      { t: "p", text: String.raw`Gradient: with $h = \hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)$, $\nabla_\theta\mathcal L = -\beta\,\sigma(-h)\,\big[\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big]$. Pairs the policy already ranks correctly ($h \gg 0$) get small weight $\sigma(-h)$; mis-ranked pairs get weight near 1.` },
      { t: "h", text: "Worked DPO example" },
      { t: "steps", items: [
        { title: "Probabilities (β = 0.1)", text: String.raw`$\pi_\theta(y_w) = 0.30$, $\pi_{\text{ref}}(y_w) = 0.20$, $\pi_\theta(y_l) = 0.10$, $\pi_{\text{ref}}(y_l) = 0.15$.` },
        { title: "Log-ratios", tex: String.raw`\log\tfrac{0.30}{0.20} = 0.405465,\qquad \log\tfrac{0.10}{0.15} = -0.405465` },
        { title: "Margin", tex: String.raw`h = 0.1\,(0.405465 + 0.405465) = 0.081093` },
        { title: "Loss and gradient weight", tex: String.raw`\sigma(h) = 0.520262,\quad \mathcal L = -\ln 0.520262 = 0.653422,\quad \beta\,\sigma(-h) = 0.1\times0.479738 = 0.047974` },
      ] },
      { t: "p", text: "Note the loss starts at $\\ln 2 = 0.6931$ when $\\pi_\\theta = \\pi_{\\text{ref}}$ (margin 0), whatever the data: a DPO loss below 0.693 means the policy has moved toward the preferred responses relative to the reference." },
      { t: "h", text: "PPO objective used in RLHF" },
      { t: "math", tex: String.raw`\mathcal J(\theta) = \mathbb E\Big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big],\quad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\quad R = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}`, caption: "$A_t$ = advantage estimated from the shaped reward $R$; $\\epsilon \\approx 0.2$. In a language model, a state is the prompt plus tokens so far and an action is the next token." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "sim", id: "entropy", caption: "Entropy, cross-entropy and KL divergence between two distributions — the quantities that the KL penalty and every alignment loss are built from." },
      { t: "list", items: [
        "Treat p as the reference and q as the policy: make q sharper than p and watch the KL measures grow: this is what happens to $\\mathrm{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})$ when RLHF concentrates probability on high-reward responses.",
        "Give an outcome almost no mass under one distribution but substantial mass under the other: the KL divergence becomes very large (infinite at zero). The KL penalty therefore makes it expensive for the policy to produce responses the reference model considers very unlikely — exactly the off-distribution region where the reward model is unreliable.",
      ] },
      { t: "sim", id: "decoding", caption: "Alignment changes the distribution; decoding chooses from it." },
      { t: "p", text: "In the decoding simulator, watch the next-character distribution of the base model. RLHF and DPO reshape such distributions (at the level of whole responses) by the factor $\\exp(r/\\beta)$; temperature reshapes them uniformly per token without any notion of quality." },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "All four pieces in NumPy on problems where the right answer is known: the masked SFT loss (and its zero gradient on prompt positions); a Bradley–Terry reward model recovering a hidden linear reward from 4,000 noisy comparisons; exact policy-gradient ascent on the KL-regularised objective converging to the closed form $\\pi^* \\propto \\pi_{\\text{ref}}e^{r/\\beta}$; and DPO, fitted directly on the comparisons, landing on the same $\\pi^*$." },
      { t: "code", lang: "python", title: "alignment_from_scratch.py", code: SCRATCH, output: SCRATCH_OUT },
      { t: "p", text: "Reading the output: the gradient is exactly zero at the five prompt positions; the learned reward matches the true reward up to an additive constant (which Bradley–Terry cannot identify); the RL objective of the policy-gradient solution is within $2\\times10^{-5}$ of the closed-form optimum; and DPO's implicit reward $\\beta\\log(\\pi/\\pi_{\\text{ref}})$ reproduces the true reward to within finite-sample noise, with $\\mathrm{KL}(\\pi_{\\text{DPO}}\\|\\pi^*) = 0.002$ — DPO solved the RLHF problem without a reward model or any sampling." },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "A full miniature pipeline in PyTorch: SFT loss masking with `ignore_index`, a neural Bradley–Terry reward model, RLHF with PPO (clipped surrogate, KL-shaped reward, per-prompt baseline) and DPO — on a toy assistant with 4 prompts and 6 candidate responses, where the optimal policy can be computed exactly." },
          { t: "code", lang: "python", title: "rlhf_vs_dpo_torch.py", code: TORCH, output: TORCH_OUT },
          { t: "p", text: "Both methods reach the KL-regularised optimum: PPO is very close to the optimum of the *learned* reward model ($\\mathrm{KL} = 0.0005$) and therefore inherits its errors relative to the true reward ($0.0053$); DPO, which never sees a reward model, gets a similar distance to the true optimum ($0.0056$). The reward model's held-out accuracy (0.720) equals that of the true reward: with noisy Bradley–Terry labels, 72% is the ceiling here, a reminder that human preference data is intrinsically noisy." },
          { t: "callout", kind: "tip", text: "For real models, the Hugging Face TRL library provides `SFTTrainer` (with completion-only loss), `RewardTrainer`, `PPOTrainer` and `DPOTrainer`; the losses are exactly the ones written here, applied to summed token log-probabilities of the response." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "preference_losses_tf.py (not executed in CI)", code: KERAS, output: "Prints a scalar tf.Tensor: the DPO loss for one pair whose policy log-ratio margin is (−12+13) − (−15+14) = 2, i.e. −log σ(0.2) ≈ 0.598." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Method", "Signal", "Advantages", "Disadvantages", "Notes"], rows: [
        ["SFT / instruction tuning", "Demonstrations", "Simple, stable, cheap", "Limited by demonstration quality; imitates, cannot exceed it", "FLAN (Wei et al., 2022) showed multi-task instruction tuning improves zero-shot ability"],
        ["RLHF (PPO)", "Learned reward + KL", "Optimises on-policy samples; flexible rewards", "4 models in memory, sampling-heavy, sensitive hyperparameters, reward hacking", "InstructGPT (Ouyang et al., 2022)"],
        ["DPO", "Preference pairs", "No reward model, no sampling, simple loss", "Offline: limited by the data's coverage; can over-optimise pairs", "Rafailov et al. (2023)"],
        ["RLAIF / Constitutional AI", "AI-generated preferences guided by principles", "Scales labelling; principles are explicit", "Inherits the judging model's biases", "Bai et al. (2022)"],
        ["Rejection sampling / best-of-n fine-tuning", "Reward model ranks n samples; SFT on the best", "Simple, stable", "Costly sampling; weaker than RL in some settings", "Often combined with RLHF"],
        ["Group-relative policy optimisation", "Reward, baseline = group mean of several samples", "No value network", "Still needs sampling", "Popular for verifiable rewards (math, code)"],
      ] },
      { t: "p", text: "Many DPO variants change the loss shape or drop the reference model (IPO, KTO, ORPO, SimPO, among others); they share the idea of turning preferences into a supervised objective. Rewards that can be checked automatically (unit tests, exact-match math answers) are increasingly used instead of, or alongside, learned reward models." },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Chat assistants**: InstructGPT and its successors used SFT followed by RLHF; labellers preferred outputs of a 1.3B InstructGPT model over those of the 175B GPT-3 (Ouyang et al., 2022).",
        "**Harmlessness training**: Anthropic's Constitutional AI (Bai et al., 2022) replaces part of the human harmlessness labels with model-generated critiques and preferences guided by written principles.",
        "**Summarisation**: Stiennon et al. (2020) trained summarisers with RLHF that humans preferred over those trained with supervised learning alone.",
        "**Open-weight model post-training**: DPO and its variants are widely used because they need only preference data and standard fine-tuning infrastructure.",
        "**Reasoning models**: RL with verifiable rewards (tests pass, answer correct) on math and code.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "list", items: [
        "**Reward hacking / over-optimisation**: the reward model is an imperfect proxy; optimising it hard eventually lowers true quality (verbosity, sycophancy, hedging). The KL coefficient β and early stopping control this.",
        "**β**: small β follows the reward aggressively (high reward, high KL, more hacking); large β barely moves from the SFT model.",
        "**On-policy vs offline**: PPO trains on the current policy's own samples, so it can correct its own mistakes; DPO only sees the responses in the dataset, which may come from a different model.",
        "**Cost and stability**: PPO needs generation in the training loop and four models; DPO is roughly the cost of SFT on pairs.",
        "**Alignment tax**: post-training can reduce performance on some benchmarks or calibration; mixing pretraining data back in (InstructGPT's 'PPO-ptx') mitigates it.",
        "**Whose preferences?** Labeller pools, instructions and principles determine the behaviour; preference data encodes value judgements.",
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Computing the SFT loss on prompt tokens.** The model learns to imitate users and wastes capacity; worse, with long prompts the response loss is diluted. Mask prompt targets with −100.",
        "**Off-by-one in the mask.** Logits at position $t$ predict token $t+1$; the mask must be aligned with the *targets*, so the last prompt position (which predicts the first response token) must be included.",
        "**Averaging token log-probs instead of summing** when computing $\\log\\pi(y\\mid x)$ for DPO or the KL. The derivation uses sequence probabilities, which are sums of token log-probs. (Length-normalised variants exist, but they are a different objective.)",
        "**Letting gradients flow into the reference model** or forgetting to put it in eval mode (dropout changes its log-probs).",
        "**Forgetting the KL penalty** (or using β ≈ 0) in RLHF: the policy collapses onto reward-model exploits within a few hundred steps.",
        "**Reading the reward-model score as absolute quality.** It is identified only up to a per-prompt constant; compare responses to the same prompt.",
        "**Interpreting a DPO loss of 0.69 as 'not learning'** without checking: it starts at $\\ln 2$ by construction. Monitor the implicit-reward accuracy and margin too.",
        "**Sampling RLHF rollouts with top-p or low temperature**: the policy-gradient and KL estimates assume samples from $\\pi_\\theta$ itself.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the three stages of the InstructGPT/RLHF pipeline?", a: "(1) Supervised fine-tuning on human-written demonstrations; (2) training a reward model on human comparisons of pairs of model responses with a Bradley–Terry loss; (3) optimising the SFT policy with PPO to maximise the reward model's score minus a KL penalty to the SFT (reference) model." },
      { t: "qa", level: "beginner", q: "Why is the loss masked on prompt tokens during SFT?", a: "We want the model to learn to produce responses given prompts, not to model the distribution of user prompts. Masking also keeps the loss scale independent of prompt length and stops long prompts from dominating the gradient." },
      { t: "qa", level: "beginner", q: "Why collect comparisons instead of asking people to rate responses on a 1–10 scale?", a: "Absolute ratings are poorly calibrated across people and over time; relative judgements ('A is better than B') are more consistent. The Bradley–Terry model converts comparisons into a scalar score whose differences predict preference probabilities." },
      { t: "qa", level: "intermediate", q: "What role does the KL penalty play in RLHF?", a: "It keeps the policy near the reference model: the reward model is trained on SFT-like responses and is unreliable far from them, so unconstrained optimisation finds adversarial outputs that score highly but are bad (reward hacking). It also preserves diversity and fluency. Mathematically it makes the optimum π* ∝ π_ref·exp(r/β) instead of a point mass on the argmax of r." },
      { t: "qa", level: "intermediate", q: "Derive the optimal policy of the KL-regularised objective.", a: "Per prompt maximise Σ_y π(y)[r(y) − β log(π(y)/π_ref(y))] subject to Σπ = 1. Lagrangian stationarity: r(y) − β log(π(y)/π_ref(y)) − β − λ = 0 ⇒ π(y) ∝ π_ref(y) exp(r(y)/β). Normalising gives π* = π_ref exp(r/β)/Z." },
      { t: "qa", level: "intermediate", q: "Write the DPO loss and explain what each term does.", a: "L = −E log σ(β[log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)]). The log-ratios are the implicit rewards; the loss is logistic regression that the winner's implicit reward exceeds the loser's. β sets how far the policy may move from π_ref; the gradient is weighted by σ(−margin), focusing on mis-ranked pairs." },
      { t: "qa", level: "advanced", q: "Why does the partition function Z(x) not appear in the DPO loss, and why does that matter?", a: "Z(x) depends only on the prompt. Writing r = β log(π*/π_ref) + β log Z(x) and plugging into the Bradley–Terry probability, which depends only on the difference of rewards for the same prompt, the β log Z(x) terms cancel. That matters because Z(x) is a sum over all possible responses and is intractable; without the cancellation the reparameterisation would be useless." },
      { t: "qa", level: "advanced", q: "When might PPO-based RLHF outperform DPO, and vice versa?", a: "PPO is on-policy: it samples from the current policy and gets reward feedback on those exact responses, so it can explore and correct behaviours absent from the preference data, and it can use rewards that are not pairwise (e.g. unit tests). DPO is offline and limited by the coverage of its dataset, and it can push down the probability of both responses or drift on responses never seen. DPO wins on simplicity, stability and cost, and performs comparably when the preference data is on-distribution (e.g. sampled from the SFT model) and plentiful. Iterated/online DPO narrows the gap." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A reward model scores the chosen response 0.5 and the rejected one 1.5. What is its Bradley–Terry loss on this pair? (3 decimals)", answer: 1.313, tolerance: 0.002, explain: "−log σ(0.5 − 1.5) = −log σ(−1) = log(1 + e) = 1.313. The model ranks the pair wrongly, so the loss is above ln 2." },
        { kind: "numeric", q: "π_ref = (0.5, 0.5), rewards (0, ln 3), β = 1. What is π*(response 2)?", answer: 0.75, tolerance: 0.001, explain: "Unnormalised (0.5, 0.5·3) = (0.5, 1.5); normalised (0.25, 0.75)." },
        { kind: "numeric", q: "What is the DPO loss for any pair when π_θ = π_ref? (4 decimals)", answer: 0.6931, tolerance: 0.0002, explain: "Both log-ratios are 0, the margin is 0, and −log σ(0) = ln 2 = 0.6931." },
        { q: "In the DPO derivation, which quantity cancels and why is that essential?", options: ["β, because it appears in both terms", "The partition function Z(x), because the Bradley–Terry model depends only on reward differences for the same prompt; Z(x) is intractable", "π_ref, because it is fixed", "The sigmoid, because it is monotonic"], answer: 1, explain: "β log Z(x) is identical for y_w and y_l, so it drops out of the difference." },
        { q: "What happens in RLHF if β → 0?", options: ["The policy stays equal to the reference", "The policy maximises the learned reward without restraint and is prone to reward hacking", "Training becomes identical to SFT", "The KL divergence becomes zero"], answer: 1, explain: "π* ∝ π_ref exp(r/β) concentrates on the reward's argmax as β → 0." },
        { q: "Why must SFT labels be aligned with targets (shifted by one) when masking?", options: ["Because the logits at position t predict token t+1, so the mask must select positions whose target is a response token", "Because PyTorch requires it", "Because the first token is always BOS", "It does not matter"], answer: 0, explain: "Masking by input position instead of target position drops the prediction of the first response token and adds the prediction of the token after it." },
        { kind: "numeric", q: "DPO with β = 0.2: log-ratios are +0.5 for y_w and −0.5 for y_l. What is the gradient weight β·σ(−h)? (4 decimals)", answer: 0.09, tolerance: 0.0002, explain: "h = 0.2·(0.5 + 0.5) = 0.2; σ(−0.2) = 0.450166; × 0.2 = 0.090033." },
      ] },
      { t: "exercise", title: "The optimal policy by hand", prompt: "Four responses have $\\pi_{\\text{ref}} = (0.4, 0.3, 0.2, 0.1)$ and rewards $r = (1, 0, 2, 3)$. Compute $\\pi^*$ for $\\beta = 1$ and for $\\beta = 0.5$, and the expected reward under each.", hint: "π* ∝ π_ref · exp(r/β); normalise.", solution: [
        { t: "steps", items: [
          { title: "β = 1, unnormalised", tex: String.raw`(0.4e,\ 0.3,\ 0.2e^{2},\ 0.1e^{3}) = (1.087313,\ 0.3,\ 1.477811,\ 2.008554),\ Z = 4.873678` },
          { title: "β = 1, normalised", tex: String.raw`\pi^* = (0.223099,\ 0.061555,\ 0.303223,\ 0.412123),\quad \mathbb E[r] = 2.065913` },
          { title: "β = 0.5, unnormalised", tex: String.raw`(0.4e^{2},\ 0.3,\ 0.2e^{4},\ 0.1e^{6}) = (2.955622,\ 0.3,\ 10.919630,\ 40.342879)` },
          { title: "β = 0.5, normalised", tex: String.raw`Z = 54.518132,\quad \pi^* = (0.054214,\ 0.005503,\ 0.200294,\ 0.739990),\quad \mathbb E[r] = 2.674771` },
        ] },
        { t: "p", text: "Halving β roughly doubles the exponent: the policy concentrates on the highest-reward response (0.74) and the expected reward rises from 2.07 to 2.67, at the price of a larger KL from the reference — the reward/KL trade-off in two lines of arithmetic." },
      ] },
      { t: "exercise", title: "Implement DPO for a real language model", prompt: "Given a causal LM `policy`, a frozen copy `ref`, and a batch of (prompt_ids, chosen_ids, rejected_ids), write the function that computes the DPO loss. Pay attention to (a) which positions contribute to $\\log\\pi(y\\mid x)$, (b) summing vs averaging, (c) gradients through the reference.", hint: "Concatenate prompt and response, take log_softmax of the logits at positions len(prompt)−1 … end−1, gather the response tokens, sum.", solution: [
        { t: "code", lang: "python", title: "dpo_lm_sketch.py", code: String.raw`import torch
import torch.nn.functional as F


def response_logp(model, prompt_ids, response_ids):
    """Sum of log p(response token | prefix) over the response tokens only. 1-D id tensors."""
    ids = torch.cat([prompt_ids, response_ids])[None]
    logits = model(ids).logits[0, :-1]                       # position t predicts token t+1
    logp = F.log_softmax(logits, -1)
    start = len(prompt_ids) - 1                             # logits that predict response tokens
    return logp[start:].gather(1, response_ids[:, None]).sum()


def dpo_loss(policy, ref, prompt_ids, chosen_ids, rejected_ids, beta=0.1):
    with torch.no_grad():                                   # the reference is frozen
        ref_c = response_logp(ref, prompt_ids, chosen_ids)
        ref_r = response_logp(ref, prompt_ids, rejected_ids)
    pol_c = response_logp(policy, prompt_ids, chosen_ids)
    pol_r = response_logp(policy, prompt_ids, rejected_ids)
    return -F.logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r)))` },
        { t: "p", text: "This is the same computation as the toy `DPO` block in the PyTorch tab, with the tabular log-probabilities replaced by summed token log-probabilities. In practice the chosen and rejected sequences are padded and batched, with a mask playing the role of the slice `logp[start:]`." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Pipeline: pretrain → SFT (demonstrations, prompt tokens masked) → reward model (pairwise preferences) → RL with a KL penalty; or SFT → DPO.",
        String.raw`Bradley–Terry: $P(y_w\succ y_l) = \sigma(r_w - r_l)$; loss $-\log\sigma(r_w - r_l)$; rewards identified up to a constant.`,
        String.raw`KL-regularised objective $\mathbb E_\pi[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ has optimum $\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$.`,
        String.raw`DPO: $-\log\sigma\big(\beta[\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}]\big)$; $Z(x)$ cancels; implicit reward $\beta\log\pi_\theta/\pi_{\text{ref}}$.`,
        "PPO-RLHF: on-policy, flexible, expensive (4 models + generation). DPO: offline, simple, cheap.",
        "β controls the reward–KL trade-off; too little regularisation → reward hacking.",
        "Sequence log-probabilities are sums over response tokens; the reference model is frozen.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference) — serving the aligned model, and the generation speed that RLHF rollouts depend on.",
        "[Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency) — SFT and DPO are usually run with parameter-efficient adapters.",
        "[Adversarial Examples, Robustness and AI Safety](/learn/m19/robustness-safety) — what alignment does and does not guarantee.",
        "Revisit [Deep Reinforcement Learning](/learn/m15/deep-rl) for the full PPO algorithm with value functions and GAE.",
      ] },
    ],
  },
};

export default lesson;
