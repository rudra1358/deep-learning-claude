import type { Lesson } from "@/lib/types";

const RL_NUMPY = String.raw`import numpy as np

# ---------------------------------------------------------------------------
# The gridworld of the simulator: 5 x 6, walls #, goal G (+1), pit P (-1), start S
# ---------------------------------------------------------------------------
LAYOUT = [".....G",
          ".###.P",
          "...#..",
          "##....",
          "S..P.."]
ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
N_S, N_A = ROWS * COLS, 4
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left
ARROW = "^>v<"
GAMMA = 0.9
cell = lambda s: LAYOUT[s // COLS][s % COLS]
START = next(s for s in range(N_S) if cell(s) == "S")
FREE = [s for s in range(N_S) if cell(s) in ".S"]   # non-wall, non-terminal states


def step(s, a):
    """Deterministic dynamics: returns next state, reward, done."""
    r, c = divmod(s, COLS)
    r2, c2 = r + MOVES[a][0], c + MOVES[a][1]
    if 0 <= r2 < ROWS and 0 <= c2 < COLS and LAYOUT[r2][c2] != "#":
        s = r2 * COLS + c2
    if cell(s) == "G":
        return s, 1.0, True
    if cell(s) == "P":
        return s, -1.0, True
    return s, 0.0, False


def show(values, policy=None):
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = r * COLS + c
            if cell(s) == "#":
                row.append("   ####")
            elif cell(s) in "GP":
                row.append(f"      {cell(s)}")
            elif policy is None:
                row.append(f"{values[s]:7.3f}")
            else:
                row.append(f"{values[s]:6.3f}{ARROW[policy[s]]}")
        print(" ".join(row))


# ---- 1. policy evaluation (Bellman expectation equation) for the uniform random policy ----
V = np.zeros(N_S)
for sweep in range(1000):
    V_new = np.zeros(N_S)
    for s in FREE:
        for a in range(N_A):
            s2, r, done = step(s, a)
            V_new[s] += 0.25 * (r + (0 if done else GAMMA * V[s2]))
    delta = np.max(np.abs(V_new - V))
    V = V_new
    if delta < 1e-10:
        break
print(f"V of the uniform random policy (policy evaluation, {sweep + 1} sweeps):")
show(V)

# ---- 2. value iteration (Bellman optimality equation) ----
Q_star = np.zeros((N_S, N_A))
for sweep in range(1000):
    Q_new = np.zeros((N_S, N_A))
    for s in FREE:
        for a in range(N_A):
            s2, r, done = step(s, a)
            Q_new[s, a] = r + (0 if done else GAMMA * Q_star[s2].max())
    delta = np.max(np.abs(Q_new - Q_star))
    Q_star = Q_new
    if delta < 1e-10:
        break
print(f"\nV* and greedy policy (value iteration, {sweep + 1} sweeps):")
show(Q_star.max(1), Q_star.argmax(1))


def n_optimal(Q):
    """States whose greedy action is optimal under Q*."""
    return sum(Q_star[s, Q[s].argmax()] >= Q_star[s].max() - 1e-9 for s in FREE)


# ---- 3. tabular Q-learning, epsilon-greedy, exploring starts ----
rng = np.random.default_rng(0)
ALPHA, EPS, EPISODES = 0.5, 0.3, 2000
Q = np.zeros((N_S, N_A))
for ep in range(1, EPISODES + 1):
    s = rng.choice(FREE)                            # exploring start
    for t in range(100):
        if rng.random() < EPS:
            a = rng.integers(N_A)
        else:                                       # greedy with random tie-breaking
            a = rng.choice(np.flatnonzero(Q[s] == Q[s].max()))
        s2, r, done = step(s, a)
        target = r + (0 if done else GAMMA * Q[s2].max())
        Q[s, a] += ALPHA * (target - Q[s, a])       # TD update towards the Bellman target
        s = s2
        if done:
            break
    if ep in (10, 100, 500, 2000):
        print(f"Q-learning episode {ep:4d}: greedy action optimal in {n_optimal(Q)}/{len(FREE)} states, "
              f"max |V - V*| = {np.max(np.abs(Q.max(1) - Q_star.max(1))[FREE]):.4f}")
print("Q-learning V and greedy policy:")
show(Q.max(1), Q.argmax(1))`;
const RL_NUMPY_OUT = String.raw`V of the uniform random policy (policy evaluation, 149 sweeps):
  0.005   0.029   0.066   0.132   0.256       G
 -0.017    ####    ####    ####  -0.361       P
 -0.046  -0.096  -0.188    ####  -0.389  -0.551
   ####    ####  -0.364  -0.552  -0.427  -0.398
 -0.203  -0.293  -0.513       P  -0.560  -0.392

V* and greedy policy (value iteration, 11 sweeps):
 0.656>  0.729>  0.810>  0.900>  1.000>       G
 0.590^    ####    ####    ####  0.900^       P
 0.531^  0.478>  0.531v    ####  0.810^  0.729<
   ####    ####  0.590>  0.656>  0.729^  0.656^
 0.430>  0.478>  0.531^       P  0.656^  0.590^
Q-learning episode   10: greedy action optimal in 10/21 states, max |V - V*| = 0.9000
Q-learning episode  100: greedy action optimal in 21/21 states, max |V - V*| = 0.3936
Q-learning episode  500: greedy action optimal in 21/21 states, max |V - V*| = 0.0001
Q-learning episode 2000: greedy action optimal in 21/21 states, max |V - V*| = 0.0000
Q-learning V and greedy policy:
 0.656>  0.729>  0.810>  0.900>  1.000>       G
 0.590^    ####    ####    ####  0.900^       P
 0.531^  0.478>  0.531v    ####  0.810^  0.729<
   ####    ####  0.590>  0.656>  0.729^  0.656<
 0.430>  0.478>  0.531^       P  0.656^  0.590^`;
const REINFORCE_NUMPY = String.raw`import numpy as np


class CartPole:
    """Cart-pole balancing (Barto, Sutton & Anderson 1983), same physics constants as the classic benchmark.
    State (x, x_dot, theta, theta_dot); actions 0 = push left, 1 = push right; reward +1 per step;
    the episode ends when |theta| > 12 degrees, |x| > 2.4, or after MAX_STEPS steps."""

    GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    THETA_MAX, X_MAX, MAX_STEPS = 12 * np.pi / 180, 2.4, 200

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.s.copy()

    def step(self, action):
        x, x_dot, th, th_dot = self.s
        f = self.FORCE if action == 1 else -self.FORCE
        cos, sin = np.cos(th), np.sin(th)
        m_tot, pml = self.M_CART + self.M_POLE, self.M_POLE * self.HALF_LEN
        temp = (f + pml * th_dot ** 2 * sin) / m_tot
        th_acc = (self.GRAVITY * sin - cos * temp) / (self.HALF_LEN * (4 / 3 - self.M_POLE * cos ** 2 / m_tot))
        x_acc = temp - pml * th_acc * cos / m_tot
        self.s = np.array([x + self.DT * x_dot, x_dot + self.DT * x_acc, th + self.DT * th_dot, th_dot + self.DT * th_acc])
        self.t += 1
        fell = abs(self.s[0]) > self.X_MAX or abs(self.s[2]) > self.THETA_MAX
        return self.s.copy(), 1.0, fell or self.t >= self.MAX_STEPS


def policy(theta, s):
    """Linear softmax policy over 2 actions: logits = s @ theta, theta has shape (4, 2)."""
    z = s @ theta
    p = np.exp(z - z.max())
    return p / p.sum()


rng = np.random.default_rng(0)
env = CartPole(rng)
theta = np.zeros((4, 2))                  # uniform random policy at the start
GAMMA, LR, BATCH = 0.99, 0.1, 10         # discount, step size, episodes per gradient estimate

for it in range(1, 61):
    grad, lengths = np.zeros_like(theta), []
    all_G, all_glog = [], []
    for _ in range(BATCH):
        s, done, traj = env.reset(), False, []
        while not done:
            p = policy(theta, s)
            a = rng.choice(2, p=p)
            s2, r, done = env.step(a)
            traj.append((s, a, r, p))
            s = s2
        lengths.append(len(traj))
        G = 0.0
        for s_t, a_t, r_t, p_t in reversed(traj):     # reward-to-go G_t = r_t+1 + gamma G_t+1
            G = r_t + GAMMA * G
            onehot = np.eye(2)[a_t]
            all_G.append(G)
            all_glog.append(np.outer(s_t, onehot - p_t))  # grad of log pi(a_t|s_t) w.r.t. theta
    G = np.array(all_G)
    adv = (G - G.mean()) / (G.std() + 1e-8)            # baseline (mean) + scale normalisation
    grad = np.tensordot(adv, np.array(all_glog), axes=1) / BATCH
    theta += LR * grad                                 # gradient ascent on J(theta)
    if it % 10 == 0 or it == 1:
        print(f"iteration {it:3d}  episodes {it * BATCH:4d}  mean episode length {np.mean(lengths):6.1f}  (max 200)")

# evaluate the final policy greedily (always take the more probable action)
lengths = []
for _ in range(20):
    s, done, n = env.reset(), False, 0
    while not done:
        s, _, done = env.step(int(np.argmax(policy(theta, s))))
        n += 1
    lengths.append(n)
print("greedy evaluation over 20 episodes: mean length", np.mean(lengths))
print("theta (rows: x, x_dot, angle, angular velocity; cols: left, right) =\n", np.round(theta, 3))`;
const REINFORCE_NUMPY_OUT = String.raw`iteration   1  episodes   10  mean episode length   21.6  (max 200)
iteration  10  episodes  100  mean episode length   46.4  (max 200)
iteration  20  episodes  200  mean episode length  136.1  (max 200)
iteration  30  episodes  300  mean episode length  188.6  (max 200)
iteration  40  episodes  400  mean episode length  200.0  (max 200)
iteration  50  episodes  500  mean episode length  197.0  (max 200)
iteration  60  episodes  600  mean episode length  190.9  (max 200)
greedy evaluation over 20 episodes: mean length 200.0
theta (rows: x, x_dot, angle, angular velocity; cols: left, right) =
 [[ 0.329 -0.329]
 [-0.866  0.866]
 [-1.386  1.386]
 [-3.09   3.09 ]]`;
const DQN_TORCH = String.raw`import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPole:
    """Cart-pole balancing with the classic physics constants; +1 reward per step, at most 200 steps."""

    GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    THETA_MAX, X_MAX, MAX_STEPS = 12 * np.pi / 180, 2.4, 200

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.s.copy()

    def step(self, action):
        x, x_dot, th, th_dot = self.s
        f = self.FORCE if action == 1 else -self.FORCE
        cos, sin = np.cos(th), np.sin(th)
        m_tot, pml = self.M_CART + self.M_POLE, self.M_POLE * self.HALF_LEN
        temp = (f + pml * th_dot ** 2 * sin) / m_tot
        th_acc = (self.GRAVITY * sin - cos * temp) / (self.HALF_LEN * (4 / 3 - self.M_POLE * cos ** 2 / m_tot))
        x_acc = temp - pml * th_acc * cos / m_tot
        self.s = np.array([x + self.DT * x_dot, x_dot + self.DT * x_acc, th + self.DT * th_dot, th_dot + self.DT * th_acc])
        self.t += 1
        fell = abs(self.s[0]) > self.X_MAX or abs(self.s[2]) > self.THETA_MAX
        truncated = self.t >= self.MAX_STEPS
        return self.s.copy(), 1.0, fell, truncated     # terminal (fell) and time-limit are different things


torch.manual_seed(0)
torch.set_num_threads(1)
random.seed(0)
env = CartPole(np.random.default_rng(0))


def q_net():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


online, target = q_net(), q_net()
target.load_state_dict(online.state_dict())        # target network starts as a copy
opt = torch.optim.Adam(online.parameters(), lr=1e-3)
buffer = deque(maxlen=10_000)                       # experience replay memory
GAMMA, BATCH, WARMUP, SYNC_EVERY = 0.99, 64, 500, 250
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 5_000
TRAIN_EVERY = 2                                     # one gradient step per 2 environment steps

step_count, lengths = 0, []
for episode in range(1, 151):
    s, done, n = env.reset(), False, 0
    while not done:
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step_count / EPS_DECAY_STEPS)
        if random.random() < eps:
            a = random.randrange(2)
        else:
            with torch.no_grad():
                a = int(online(torch.tensor(s, dtype=torch.float32)).argmax())
        s2, r, fell, truncated = env.step(a)
        buffer.append((s, a, r, s2, float(fell)))       # store the transition
        s, done, n = s2, fell or truncated, n + 1
        step_count += 1

        if len(buffer) >= WARMUP and step_count % TRAIN_EVERY == 0:
            batch = random.sample(buffer, BATCH)        # uniform sampling breaks temporal correlation
            S, A, R, S2, D = (torch.tensor(np.array(col), dtype=torch.float32) for col in zip(*batch))
            q_sa = online(S).gather(1, A.long()[:, None]).squeeze(1)       # Q(s, a) for the actions taken
            with torch.no_grad():                                           # TD target from the FROZEN network
                y = R + GAMMA * (1 - D) * target(S2).max(1).values
            loss = F.smooth_l1_loss(q_sa, y)                                # Huber loss, as in Mnih et al. 2015
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step_count % SYNC_EVERY == 0:
            target.load_state_dict(online.state_dict())                     # periodic hard update
    lengths.append(n)
    if episode % 25 == 0:
        print(f"episode {episode:3d}  steps {step_count:6d}  epsilon {eps:.2f}  "
              f"mean length of last 25 episodes {np.mean(lengths[-25:]):6.1f}")

# greedy evaluation (no exploration)
eval_len = []
for _ in range(10):
    s, done, n = env.reset(), False, 0
    while not done:
        with torch.no_grad():
            s, _, fell, truncated = env.step(int(online(torch.tensor(s, dtype=torch.float32)).argmax()))
        done, n = fell or truncated, n + 1
    eval_len.append(n)
print("greedy evaluation, 10 episodes:", eval_len)
with torch.no_grad():
    print(f"Q(start state) = {online(torch.zeros(4)).numpy().round(2)}  "
          f"(a 200-step episode is worth at most (1 - 0.99^200) / 0.01 = {(1 - 0.99 ** 200) / 0.01:.1f})")`;
const DQN_TORCH_OUT = String.raw`episode  25  steps    487  epsilon 0.91  mean length of last 25 episodes   19.5
episode  50  steps    984  epsilon 0.81  mean length of last 25 episodes   19.9
episode  75  steps   1546  epsilon 0.71  mean length of last 25 episodes   22.5
episode 100  steps   3286  epsilon 0.38  mean length of last 25 episodes   69.6
episode 125  steps   7238  epsilon 0.05  mean length of last 25 episodes  158.1
episode 150  steps  11047  epsilon 0.05  mean length of last 25 episodes  152.4
greedy evaluation, 10 episodes: [159, 188, 145, 158, 177, 178, 173, 150, 143, 174]
Q(start state) = [37.48 37.51]  (a 200-step episode is worth at most (1 - 0.99^200) / 0.01 = 86.6)`;
const PPO_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VecCartPole:
    """N independent cart-poles simulated together with NumPy arrays (classic physics constants).
    Finished environments are reset automatically; max 200 steps per episode."""

    G, MC, MP, L, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    TH_MAX, X_MAX, MAX_STEPS = 12 * np.pi / 180, 2.4, 200

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.s = rng.uniform(-0.05, 0.05, (n, 4))
        self.t = np.zeros(n, int)

    def step(self, a):
        x, xd, th, thd = self.s.T
        f = np.where(a == 1, self.FORCE, -self.FORCE)
        cos, sin = np.cos(th), np.sin(th)
        mt, pml = self.MC + self.MP, self.MP * self.L
        temp = (f + pml * thd ** 2 * sin) / mt
        th_acc = (self.G * sin - cos * temp) / (self.L * (4 / 3 - self.MP * cos ** 2 / mt))
        x_acc = temp - pml * th_acc * cos / mt
        self.s = np.stack([x + self.DT * xd, xd + self.DT * x_acc, th + self.DT * thd, thd + self.DT * th_acc], 1)
        self.t += 1
        fell = (np.abs(self.s[:, 0]) > self.X_MAX) | (np.abs(self.s[:, 2]) > self.TH_MAX)
        trunc = self.t >= self.MAX_STEPS
        done = fell | trunc
        ep_len = np.where(done, self.t, 0)
        self.s[done] = self.rng.uniform(-0.05, 0.05, (done.sum(), 4))   # auto-reset
        self.t[done] = 0
        return self.s.copy(), np.ones(self.n), fell, done, ep_len


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        self.v = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, s):
        return torch.distributions.Categorical(logits=self.pi(s)), self.v(s).squeeze(-1)


torch.manual_seed(0)
torch.set_num_threads(1)
rng = np.random.default_rng(0)
N_ENV, T = 8, 128                       # rollout: 8 environments x 128 steps = 1024 transitions
GAMMA, LAM, CLIP = 0.99, 0.95, 0.2      # discount, GAE lambda, PPO clip range epsilon
EPOCHS, MB = 4, 256                     # optimisation epochs per rollout, minibatch size
envs = VecCartPole(N_ENV, rng)
ac = ActorCritic()
opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
s = envs.s.copy()
finished = []

for it in range(1, 31):
    S, A, LOGP, R, FELL, DONE, V = [], [], [], [], [], [], []
    with torch.no_grad():                                         # 1. collect a rollout with the current policy
        for t in range(T):
            st = torch.tensor(s, dtype=torch.float32)
            dist, v = ac(st)
            a = dist.sample()
            s2, r, fell, done, ep_len = envs.step(a.numpy())
            S.append(st); A.append(a); LOGP.append(dist.log_prob(a)); V.append(v)
            R.append(r); FELL.append(fell); DONE.append(done)
            finished += ep_len[done].tolist()
            s = s2
        v_last = ac(torch.tensor(s, dtype=torch.float32))[1].numpy()
    V_np = torch.stack(V).numpy()
    adv = np.zeros((T, N_ENV))                                    # 2. generalised advantage estimation
    gae = np.zeros(N_ENV)
    for t in reversed(range(T)):
        v_next = v_last if t == T - 1 else V_np[t + 1]
        # bootstrap from V(s') unless the pole fell; at a time-limit reset the next stored state belongs
        # to a new episode, so the chain of advantages is cut there as well
        delta = R[t] + GAMMA * v_next * (1 - FELL[t]) - V_np[t]
        delta = np.where(DONE[t] & ~FELL[t], R[t] + GAMMA * V_np[t] - V_np[t], delta)   # truncation: approx V(s') ~ V(s)
        gae = delta + GAMMA * LAM * (1 - DONE[t]) * gae
        adv[t] = gae
    ret = adv + V_np                                              # value targets
    S_b, A_b, LOGP_b = torch.cat(S), torch.cat(A), torch.cat(LOGP)
    ADV_b = torch.tensor(adv.reshape(-1), dtype=torch.float32)
    RET_b = torch.tensor(ret.reshape(-1), dtype=torch.float32)
    ADV_b = (ADV_b - ADV_b.mean()) / (ADV_b.std() + 1e-8)

    for epoch in range(EPOCHS):                                   # 3. several epochs of minibatch updates
        idx = torch.randperm(N_ENV * T)
        for start in range(0, N_ENV * T, MB):
            mb = idx[start:start + MB]
            dist, v = ac(S_b[mb])
            ratio = torch.exp(dist.log_prob(A_b[mb]) - LOGP_b[mb])          # pi_new / pi_old
            surr = torch.min(ratio * ADV_b[mb], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV_b[mb])
            loss = -surr.mean() + 0.5 * F.mse_loss(v, RET_b[mb]) - 0.01 * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()
    if it % 5 == 0:
        with torch.no_grad():
            frac = ((ratio - 1).abs() > CLIP).float().mean().item()
        print(f"iteration {it:2d}  env steps {it * N_ENV * T:5d}  episodes finished {len(finished):4d}  "
              f"mean length (last 20) {np.mean(finished[-20:]):6.1f}  clipped (last minibatch) {frac:.2f}")

# greedy evaluation
test = VecCartPole(10, np.random.default_rng(123))
s, lengths, alive = test.s.copy(), np.zeros(10, int), np.ones(10, bool)
for t in range(200):
    with torch.no_grad():
        a = ac.pi(torch.tensor(s, dtype=torch.float32)).argmax(1).numpy()
    s, _, fell, done, _ = test.step(a)
    lengths += alive
    alive &= ~done
print("greedy evaluation, 10 episodes:", lengths.tolist())`;
const PPO_TORCH_OUT = String.raw`iteration  5  env steps  5120  episodes finished  173  mean length (last 20)   55.2  clipped (last minibatch) 0.00
iteration 10  env steps 10240  episodes finished  253  mean length (last 20)   80.8  clipped (last minibatch) 0.01
iteration 15  env steps 15360  episodes finished  295  mean length (last 20)  110.5  clipped (last minibatch) 0.02
iteration 20  env steps 20480  episodes finished  332  mean length (last 20)  146.7  clipped (last minibatch) 0.00
iteration 25  env steps 25600  episodes finished  363  mean length (last 20)  163.3  clipped (last minibatch) 0.00
iteration 30  env steps 30720  episodes finished  394  mean length (last 20)  174.8  clipped (last minibatch) 0.00
greedy evaluation, 10 episodes: [194, 200, 200, 200, 200, 175, 177, 200, 199, 182]`;
const DQN_KERAS = String.raw`import numpy as np
import tensorflow as tf

# Q-network and its frozen copy
def q_net():
    return tf.keras.Sequential([tf.keras.Input((4,)),
                                tf.keras.layers.Dense(64, activation="relu"),
                                tf.keras.layers.Dense(64, activation="relu"),
                                tf.keras.layers.Dense(2)])

online, target = q_net(), q_net()
target.set_weights(online.get_weights())
opt = tf.keras.optimizers.Adam(1e-3)
huber = tf.keras.losses.Huber()
GAMMA = 0.99


@tf.function
def train_step(S, A, R, S2, D):
    y = R + GAMMA * (1.0 - D) * tf.reduce_max(target(S2), axis=1)       # TD target, no gradient
    with tf.GradientTape() as tape:
        q = tf.gather(online(S), A, axis=1, batch_dims=1)               # Q(s, a) of the actions taken
        loss = huber(y, q)
    grads = tape.gradient(loss, online.trainable_variables)
    opt.apply_gradients(zip(grads, online.trainable_variables))
    return loss

# The environment loop, replay buffer, epsilon schedule and periodic
# target.set_weights(online.get_weights()) are the same as in the PyTorch program.`;

const lesson: Lesson = {
  id: "m15/deep-rl",
  title: "Deep Reinforcement Learning",
  summary:
    "Learning from reward instead of labels: MDPs, returns and discounting, value functions and Bellman equations, value iteration and Q-learning, Deep Q-Networks with replay and target networks, policy gradients (REINFORCE), actor-critic with advantages, and PPO's clipped objective — derived, worked by hand and built in NumPy and PyTorch on home-made gridworld and cart-pole environments.",
  minutes: 95,
  level: "advanced",
  prerequisites: ["m02/probability", "m03/training-basics", "m04/backpropagation", "m06/gradient-descent"],
  next: ["m20/alignment", "m15/graph-neural-networks", "m15/multimodal"],
  terms: ["markov-decision-process", "policy", "return", "discount-factor", "value-function", "action-value-function", "bellman-equation", "q-learning", "epsilon-greedy", "deep-q-network", "experience-replay", "target-network", "policy-gradient", "reinforce", "advantage", "actor-critic", "ppo"],
  sections: {
    intro: [
      {
        t: "p",
        text: "In supervised learning, every input comes with the right answer. In **reinforcement learning** (RL), an **agent** acts in an **environment**: at each step it sees a **state**, picks an **action**, and gets a number (the **reward**) plus the next state. Nobody says which action was right; the agent must find, by trial and error, behaviour that collects the most reward over time. Two things make this hard: an action's effects may show up many steps later (the **credit-assignment** problem), and the agent must balance trying new actions (**exploration**) against using what it knows (**exploitation**).",
      },
      {
        t: "p",
        text: "**Deep** RL uses neural networks for what the agent learns — a value function, a policy, or both — so it can handle huge or continuous state spaces such as raw pixels. Two landmarks: the **Deep Q-Network** (DQN) of Mnih et al. (2015) learned many Atari games from pixels with one architecture and one set of hyper-parameters, and policy-gradient methods led to **Proximal Policy Optimization** (PPO, Schulman et al. 2017), today's default on-policy algorithm and the optimiser used in reinforcement learning from human feedback for language models.",
      },
      {
        t: "p",
        text: "We build the theory (MDPs, returns, value functions, Bellman equations), then follow two families: **value-based** (Q-learning → DQN) and **policy-based** (REINFORCE → actor-critic → PPO). Everything runs on small home-made environments, so every number reproduces on a CPU in seconds.",
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`Goal: find a policy $\pi$ maximising $J(\pi) = \mathbb E_\pi\big[\sum_{t\ge0}\gamma^t r_{t+1}\big]$. Value-based: learn $Q^*(s,a) = \mathbb E[r + \gamma\max_{a'}Q^*(s',a')]$ and act greedily. Policy-based: ascend $\nabla J = \mathbb E\big[\nabla\log\pi_\theta(a\mid s)\,A^\pi(s,a)\big]$.`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Probability](/learn/m02/probability) — expectations and conditional distributions; returns and values are expectations over random trajectories.",
          "[Training basics](/learn/m03/training-basics) — losses, mini-batches and the training loop. DQN is regression onto moving targets; PPO minimises a stand-in (surrogate) loss.",
          "[Backpropagation](/learn/m04/backpropagation) — every method here backpropagates a loss through a network; the RL part is how the loss and targets are built.",
          "[Gradient descent](/learn/m06/gradient-descent) and [Adam](/learn/m06/momentum-adaptive) — policy gradients are stochastic gradient *ascent* on expected return.",
          "[Classification losses](/learn/m05/classification-losses) (helpful) — $\\log\\pi(a\\mid s)$ of a softmax policy is the negative cross-entropy; its gradient is onehot − probabilities.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Learning a city without a map",
        text: "You move to a new city and want the fastest way to a favourite café. Nobody gives directions; you only notice when you arrive. At first you wander (exploration). Over time you remember, for each corner, how far the café felt from there — a **value**. Once a corner next to the café is known to be close, the corners leading to it inherit that, one step at a time. This backing-up of value is the **Bellman equation**, and it is what Q-learning does. Or skip values and adjust habits: turns from quick trips become more likely, turns from slow trips less likely — the **policy gradient**. Comparing a trip with what you *expected* from that corner (the **advantage**) tells you far more than the raw trip time — the idea behind actor-critic methods.",
      },
      {
        t: "p",
        text: "Two things make RL harder than supervised learning. The data depend on the agent: a bad policy visits useless states, and changing the policy changes the data. And feedback is delayed and sparse: in this lesson's gridworld the only non-zero rewards are +1 at the goal and −1 at the pit, so credit for every in-between decision must flow back through long chains of states.",
      },
      {
        t: "p",
        text: "**Discounting** with a factor $\\gamma < 1$ makes reward now worth more than the same reward later. With $\\gamma = 0.9$ and a goal 9 moves away, reaching it is worth $0.9^8 = 0.430$ from the start. So the shortest safe path is also the most valuable, and values encode distance to the goal.",
      },
    ],
    mechanism: [
      { t: "h", text: "The agent–environment loop" },
      {
        t: "steps",
        items: [
          { title: "Observe", text: "The agent sees state $s_t$ (a grid cell; cart and pole positions and speeds; or a stack of game frames)." },
          { title: "Act", text: "It samples an action from its **policy**, $a_t \\sim \\pi(\\cdot\\mid s_t)$ — a distribution over actions (stochastic) or one fixed action (deterministic)." },
          { title: "Environment transition", text: "The environment returns a reward $r_{t+1}$ and next state $s_{t+1}$, drawn from its rules $p(s', r\\mid s, a)$, which the agent does not know." },
          { title: "Learn", text: "The agent uses $(s_t, a_t, r_{t+1}, s_{t+1})$ to improve its values and/or policy — right away (online TD), after storing it (replay), or when the episode ends (Monte Carlo)." },
          { title: "Repeat until the episode ends", text: "An **episode** ends in a terminal state (goal reached, pole fallen) or is cut by a time limit — which is *not* a terminal state of the MDP, a difference that matters for bootstrapping." },
        ],
      },
      { t: "h", text: "Q-learning, step by step" },
      {
        t: "steps",
        items: [
          { title: "Table", text: "Keep $Q(s, a)$ for every state–action pair, starting at 0." },
          { title: "ε-greedy action", text: "With probability ε pick a random action; otherwise pick $\\arg\\max_a Q(s,a)$ (break ties at random)." },
          { title: "Step", text: "Take it; observe $r$ and $s'$." },
          { title: "TD target", text: "One step of the Bellman optimality equation using current estimates (no bootstrap if $s'$ is terminal).", tex: String.raw`y = r + \gamma\max_{a'}Q(s', a')` },
          { title: "Move towards it", text: "$y - Q(s,a)$ is the **temporal-difference (TD) error**; α is the step size.", tex: String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\big(y - Q(s,a)\big)` },
          { title: "Act greedily at the end", text: "The output is the greedy policy $\\arg\\max_a Q(s,a)$. Q-learning is **off-policy**: it learns the greedy policy's values while behaving ε-greedily." },
        ],
      },
      { t: "h", text: "From Q-learning to DQN" },
      {
        t: "steps",
        items: [
          { title: "Function approximation", text: "Replace the table with a network $Q_\\theta(s, \\cdot)$ that outputs one value per action, so similar states share what is learned." },
          { title: "Experience replay", text: "Store transitions $(s, a, r, s', \\text{done})$ in a buffer (e.g. the last $10^4$–$10^6$) and train on random mini-batches. This breaks the link between consecutive transitions and reuses each many times." },
          { title: "Target network", text: "Compute targets with a copy $Q_{\\theta^-}$ refreshed from $\\theta$ only every $C$ steps. Otherwise each update moves the very target it is chasing, destabilising training.", tex: String.raw`y = r + \gamma(1 - \text{done})\max_{a'}Q_{\theta^-}(s', a')` },
          { title: "Regression step", text: "Minimise the Huber loss between $Q_\\theta(s,a)$ and $y$. The gradient flows only through $Q_\\theta(s,a)$, never through the target." },
          { title: "Decaying ε", text: "Start fully random (ε = 1) and lower it to a small value (0.05) over the first few thousand steps." },
        ],
      },
      { t: "h", text: "Policy gradients and PPO, step by step" },
      {
        t: "steps",
        items: [
          { title: "Parameterise the policy", text: "A network outputs action probabilities $\\pi_\\theta(a\\mid s)$ (softmax for discrete actions; a Gaussian mean and variance for continuous ones)." },
          { title: "Collect trajectories", text: "Run the current policy (on-policy) for some steps, often in several environments at once." },
          { title: "Score each action", text: "REINFORCE uses the return-to-go $G_t$ minus a baseline. Actor-critic methods use an **advantage** estimate $\\hat A_t$ from a learned value function $V_\\phi$ (the critic)." },
          { title: "Make good actions more likely", text: "Gradient ascent on $\\sum_t \\log\\pi_\\theta(a_t\\mid s_t)\\hat A_t$. Actions that did better than expected become more likely." },
          { title: "PPO: reuse data safely", text: "Optimise a clipped surrogate for several epochs on the same rollout. Clipping the probability ratio $\\pi_\\theta/\\pi_{\\theta_{old}}$ to $[1-\\epsilon, 1+\\epsilon]$ removes the reward for moving the policy far from the one that collected the data." },
        ],
      },
    ],
    math: [
      { t: "h", text: "1. Markov decision processes" },
      {
        t: "p",
        text: String.raw`An MDP is a tuple $(\mathcal S, \mathcal A, p, r, \gamma)$: states $\mathcal S$, actions $\mathcal A$, transition probabilities $p(s'\mid s, a)$, expected rewards $r(s, a)$ and a discount factor $\gamma \in [0, 1)$. The **Markov property**: the next state and reward depend only on the current state and action, not on earlier history. A policy $\pi(a\mid s)$ gives action probabilities per state.`,
      },
      { t: "h", text: "2. Returns" },
      { t: "math", tex: String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1} = r_{t+1} + \gamma G_{t+1}`, caption: "The discounted sum of future rewards. Its recursive form (this reward + γ × next return) makes dynamic programming and TD learning possible." },
      { t: "p", text: "Example: rewards $(1, 0, 2)$ then the end, with $\\gamma = 0.9$, give $G_0 = 1 + 0.9\\cdot0 + 0.81\\cdot2 = 2.62$. With +1 per step forever, $G = 1/(1-\\gamma)$. For cart-pole capped at 200 steps with $\\gamma = 0.99$, the maximum is $(1 - 0.99^{200})/0.01 = 86.6$." },
      { t: "h", text: "3. Value functions and the Bellman equations" },
      {
        t: "p",
        text: String.raw`The **state-value function** $V^\pi(s) = \mathbb E_\pi[G_t\mid s_t = s]$ is the expected return from $s$ when following $\pi$; the **action-value function** $Q^\pi(s,a) = \mathbb E_\pi[G_t\mid s_t = s, a_t = a]$ is the same after first taking $a$. Plugging in $G_t = r_{t+1} + \gamma G_{t+1}$ gives the **Bellman expectation equations** — "value here = reward now + discounted value next":`,
      },
      { t: "math", tex: String.raw`V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s', r} p(s', r\mid s, a)\big[r + \gamma V^\pi(s')\big],\qquad Q^\pi(s,a) = \sum_{s', r} p(s', r\mid s, a)\Big[r + \gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s', a')\Big]` },
      {
        t: "p",
        text: String.raw`The best possible values $V^*(s) = \max_\pi V^\pi(s)$ and $Q^*(s,a) = \max_\pi Q^\pi(s,a)$ satisfy the **Bellman optimality equations** (Bellman, 1957). Any policy that is greedy with respect to $Q^*$ is optimal:`,
      },
      { t: "math", tex: String.raw`Q^*(s,a) = \sum_{s', r} p(s', r\mid s, a)\big[r + \gamma\max_{a'}Q^*(s', a')\big],\qquad \pi^*(s) = \arg\max_a Q^*(s,a)` },
      {
        t: "p",
        text: String.raw`Both equations are contractions (by $\gamma$ in the max-norm): applying the right-hand side repeatedly converges to one unique answer — **policy evaluation** for $V^\pi$, **value iteration** for $Q^*$. In the gridworld (deterministic moves, +1 at G, −1 at P, 0 otherwise, $\gamma = 0.9$) value iteration converges in 11 sweeps. A cell next to G has $V^* = 1$, two moves away $0.9$, and the start S, 9 moves away, $V^* = 0.9^8 = 0.430467$. The uniform random policy has $V^\pi(S) = -0.203$: it falls into the nearby pit more often than it finds the goal.`,
      },
      { t: "h", text: "4. Q-learning as stochastic approximation" },
      {
        t: "p",
        text: String.raw`Q-learning (Watkins, 1989) replaces the expectation in the optimality equation with one sampled transition and takes a small step: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$. Watkins and Dayan (1992) proved it converges to $Q^*$ with probability 1 if every state–action pair is visited infinitely often and step sizes shrink appropriately. **Worked update**: $Q(s,a) = 0.4$, $r = 0$, $\gamma = 0.9$, $\max_{a'}Q(s',a') = 0.8$, $\alpha = 0.5$. Target $= 0 + 0.9\cdot0.8 = 0.72$, TD error $= 0.32$, new $Q = 0.4 + 0.5\cdot0.32 = 0.56$.`,
      },
      {
        t: "p",
        text: "The visiting condition matters. In the simulator with a fixed start and ε = 0.1, the path from S becomes optimal quickly, but after 600 episodes only 15 of 21 states have an optimal greedy action — cells far from the path are rarely visited. Exploring starts (a random start cell each episode) with ε = 0.3 make all 21 optimal after 97 episodes with the default seed.",
      },
      { t: "h", text: "5. DQN" },
      {
        t: "p",
        text: String.raw`With a network $Q_\theta$, Q-learning becomes regression onto bootstrapped targets (targets built from the network's own estimates). DQN minimises, over mini-batches $\mathcal B$ from the replay buffer,`,
      },
      { t: "math", tex: String.raw`\mathcal L(\theta) = \frac{1}{|\mathcal B|}\sum_{(s,a,r,s',d)\in\mathcal B}\ell_{\delta}\Big(Q_\theta(s,a) - \big[r + \gamma(1 - d)\max_{a'}Q_{\theta^-}(s', a')\big]\Big)`, caption: "ℓ_δ is the Huber loss (quadratic for small errors, linear for large ones), θ⁻ the target-network weights, and d = 1 only for true terminal states." },
      {
        t: "p",
        text: String.raw`Function approximation + bootstrapping + off-policy data can diverge together — Sutton and Barto's "deadly triad"; replay and target networks are the stabilisers that made it work. The max in the target also causes **overestimation**: $\mathbb E[\max_a \hat Q] \ge \max_a\mathbb E[\hat Q]$, so the max picks up lucky noise. **Double DQN** (van Hasselt, Guez and Silver, 2016) chooses the action with the online network and scores it with the target network: $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'}Q_\theta(s',a'))$.`,
      },
      { t: "h", text: "6. The policy-gradient theorem and REINFORCE" },
      {
        t: "p",
        text: String.raw`For a parameterised policy, $J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$, where $\tau = (s_0, a_0, r_1, s_1, \dots)$ is a trajectory. Its probability is $p_\theta(\tau) = p(s_0)\prod_t\pi_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t, a_t)$. The unknown environment rules do not depend on $\theta$, so $\nabla_\theta\log p_\theta(\tau) = \sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$. Using $\nabla p = p\nabla\log p$ (the score-function or log-derivative trick):`,
      },
      { t: "math", tex: String.raw`\nabla_\theta J = \mathbb E_{\tau}\big[R(\tau)\nabla_\theta\log p_\theta(\tau)\big] = \mathbb E_\tau\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big]` },
      {
        t: "p",
        text: String.raw`In plain words: raise each action's log-probability, weighted by how good the return after it was. Two noise reductions are built in: an action cannot affect earlier rewards, so $R(\tau)$ becomes the return-to-go $G_t$; and any **baseline** $b(s_t)$ that does not depend on the action can be subtracted without bias, because $\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)] = \sum_a\nabla\pi(a\mid s) = \nabla 1 = 0$. This is **REINFORCE** (Williams, 1992). For a softmax policy with logits $z$, $\nabla_z\log\pi(a\mid s) = \mathrm{onehot}(a) - \pi(\cdot\mid s)$; e.g. $\pi = (0.7, 0.3)$ with action 1 gives $(-0.7, 0.7)$, scaled by the (baselined) return.`,
      },
      { t: "h", text: "7. Advantages and actor-critic" },
      {
        t: "p",
        text: String.raw`The best baseline is $V^\pi(s)$, which gives the **advantage** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: how much better action $a$ is than the policy's average in $s$. An **actor-critic** method learns a critic $V_\phi(s)$ by regression and estimates the advantage with the TD error $\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ (low variance, some bias) or with the Monte-Carlo $G_t - V_\phi(s_t)$ (no bias from $V$, high variance). **Generalised advantage estimation** (Schulman et al., 2016) blends the two with $\lambda\in[0,1]$:`,
      },
      { t: "math", tex: String.raw`\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}`, caption: "λ = 0 gives the one-step TD error, λ = 1 the Monte-Carlo advantage; λ = 0.95 is common." },
      { t: "h", text: "8. PPO's clipped objective" },
      {
        t: "p",
        text: String.raw`A plain policy gradient uses each batch for one step; bigger steps on the same data can collapse the policy, as the estimate is only valid near $\pi_{\theta_{old}}$. TRPO (Schulman et al., 2015) limits the KL divergence between old and new policy; PPO (Schulman et al., 2017) gets a similar effect with a clipped surrogate. With the probability ratio $\rho_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{old}}(a_t\mid s_t)$:`,
      },
      { t: "math", tex: String.raw`L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb E}_t\Big[\min\big(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big]` },
      {
        t: "p",
        text: String.raw`The full loss maximises $L^{\mathrm{CLIP}}$, minimises the critic's squared error and adds an entropy bonus for exploration. **Worked values** with $\epsilon = 0.2$: $\rho = 1.5$, $\hat A = 2$ gives $\min(3, 1.2\cdot2) = 2.4$ — pushing a good action's probability past $1.2\times$ earns nothing more. $\rho = 0.6$, $\hat A = -1$ gives $\min(-0.6, 0.8\cdot(-1)) = -0.8$ — no extra reward for making a bad action even rarer. But $\rho = 1.5$, $\hat A = -1$ gives $\min(-1.5, -1.2) = -1.5$: when the policy moved the wrong way, the unclipped (more pessimistic) term is kept. So the objective is a lower bound on the unclipped one, and the gradient pulls the ratio back.`,
      },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator runs tabular Q-learning in the From-Scratch 5 × 6 gridworld. Each Step takes one action and applies one update; the side panel shows its exact numbers, and the chart compares the greedy policy with the optimal one from value iteration.",
      },
      { t: "sim", id: "rl-gridworld", caption: "Q-learning with ε-greedy exploration: values and greedy arrows per cell, the update just applied, the episode-return curve and the share of states whose greedy action is optimal." },
      {
        t: "list",
        items: [
          "Press **Step** a few times. All Q-values are 0, so the agent wanders and nothing changes until it first enters G or P; then one Q-value becomes ±0.5 (α = 0.5 times reward ±1). Value spreads back one cell at a time, only when the agent steps into a cell whose value is already non-zero.",
          "Switch **per tick** to *episode* and press Play. With the defaults (α = 0.5, γ = 0.9, ε = 0.3, random starts, seed 7) the greedy action is optimal in all 21 states from episode 97 on, and the gold greedy path from S takes the optimal 9 moves. Values keep improving after that: $V(S)$ is 0.208 at episode 100 and reaches $0.9^8 = 0.430$ around episode 300.",
          "Switch the view to **Q(s,a)** to see the four action values per cell. Next to the pit, the action that enters P converges to −1, and those cells never choose it.",
          "Set **start = S** and ε = 0.1, then press *+100 episodes* several times. The path from S becomes optimal, but cells away from it keep red (non-optimal) arrows: they are almost never updated. That is why the theorem needs every state–action pair visited infinitely often.",
          "Set **γ = 0.5**: values fall off as $0.5^k$; cells far from G are near 0. Set **ε = 0**: with random starts the agent still learns, but with a fixed start it can get stuck repeating the first route that worked.",
          "Set **α = 1**: fine in this deterministic world (the target is exact once the next state's value is right); with random rewards or transitions, Q would jump to every noisy sample.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "**Program 1** — dynamic programming and Q-learning on the gridworld: evaluate the uniform random policy with the Bellman expectation equation, find $Q^*$ by value iteration, then learn $Q$ by tabular Q-learning from samples and compare.",
      },
      { t: "code", lang: "python", title: "gridworld_dp_and_q_learning.py", code: RL_NUMPY, output: RL_NUMPY_OUT },
      {
        t: "list",
        items: [
          "Policy evaluation needs 149 sweeps (to $10^{-10}$), value iteration only 11: once values along the shortest paths are set, the max ignores everything else.",
          "Q-learning only sees transitions, not the rules. After 100 episodes its greedy policy is optimal everywhere, though values are still off by up to 0.39; by episode 500 they match $V^*$ to $10^{-4}$.",
          "Cell (3,5) shows ← in the Q-learning policy and ↑ in value iteration: both actions have value 0.656 (a tie), so both are optimal.",
        ],
      },
      {
        t: "p",
        text: "**Program 2** — REINFORCE on a home-made cart-pole with the physics of Barto, Sutton and Anderson's classic benchmark (1983): a pole hinged on a cart, pushes of 10 N left or right, +1 reward per step. The episode ends when the pole tilts past 12° or the cart leaves ±2.4, capped at 200 steps. The policy is linear: logits = state · θ.",
      },
      { t: "code", lang: "python", title: "reinforce_cartpole.py", code: REINFORCE_NUMPY, output: REINFORCE_NUMPY_OUT },
      {
        t: "list",
        items: [
          "Each step's gradient is $(\\mathrm{onehot}(a) - \\pi)\\,s^\\top$ times its normalised return-to-go: subtracting the batch mean is the baseline; dividing by the standard deviation lets one learning rate work throughout.",
          "From the random policy's 21.6 steps, mean episode length reaches the 200-step cap within 400 episodes; the final greedy policy balances all 200 steps in 20 test episodes.",
          "The learned weights make sense: the 'right' logit grows with the pole angle and, most strongly, its angular velocity — push in the direction the pole is falling.",
          "Policy gradients are not automatically easy. The same REINFORCE with a tabular softmax policy on the gridworld (sparse ±1 rewards, pit near the start) collapsed in our runs to bumping into walls until the 100-step time-out (return 0): it learns to avoid the pit long before it finds the goal. Q-learning with exploring starts solves it easily.",
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
                text: "**DQN** with a 10,000-transition replay buffer, a target network synced every 250 steps, linearly decaying ε, Huber loss and one gradient step per two environment steps. The cart-pole is defined inline (no Gymnasium). The `fell` flag (true terminal) and the 200-step time limit are kept apart, so the TD target still bootstraps after a time-out.",
              },
              { t: "code", lang: "python", title: "dqn_cartpole.py", code: DQN_TORCH, output: DQN_TORCH_OUT },
              {
                t: "list",
                items: [
                  "Nothing improves while ε is high (episodes 1–75): the data are almost random. Once ε drops below ≈0.4, episode lengths jump from ~20 to ~150 steps.",
                  "Learning is noisy: the 25-episode average dips from 158.1 to 152.4 at the end, and the greedy policy balances 143–188 steps, not the full 200. DQN results vary a lot with seeds and hyper-parameters; report several seeds in real work.",
                  "$Q(s_0) \\approx 37.5$ is well below the 86.6 a 200-step episode would be worth: the values still lag behind the improved policy, as in the tabular case.",
                ],
              },
              {
                t: "p",
                text: "**PPO** (actor-critic with GAE and clipping) on 8 cart-poles simulated side by side in NumPy. Each iteration collects 8 × 128 = 1,024 transitions, computes advantages, then runs 4 epochs of minibatch updates on clipped surrogate + value loss − entropy bonus.",
              },
              { t: "code", lang: "python", title: "ppo_cartpole.py", code: PPO_TORCH, output: PPO_TORCH_OUT },
              {
                t: "list",
                items: [
                  "Mean episode length climbs steadily from 55 to 175 over 30,720 environment steps. The greedy policy balances for 175–200 steps, hitting the cap in 5 of 10 test episodes.",
                  "The clipped fraction (last minibatch of each iteration) stays at 0–2%: with lr = 3e-4 and 4 epochs the policy rarely leaves the trust region, so clipping is a safety net, not the main brake.",
                  "Time-limit cut-offs are handled approximately: the automatic reset overwrites the state after a time-out, so the program bootstraps with $V(s_t)$ instead of the lost $V(s_{t+1})$. Gymnasium-style APIs return the final observation separately for exact bootstrapping.",
                  "PPO used ~31k environment steps, DQN ~11k: off-policy replay is more sample-efficient, while PPO is simpler to tune, parallelises across environments, and is the usual pick when simulation is cheap.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "The core of DQN in Keras: an online network, a target network and a compiled training step. The rest of the loop matches the PyTorch program." },
              { t: "code", lang: "python", title: "dqn_keras.py (not executed in CI)", code: DQN_KERAS, output: "Defines the networks and a train_step that returns the Huber loss of one mini-batch; nothing is printed on its own." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Method", "Learns", "Strengths", "Weaknesses", "Typical use"],
        rows: [
          ["Value iteration / policy iteration", "$V^*$ or $Q^*$ from a known model", "exact, fast for small MDPs", "needs $p(s'\\mid s,a)$; tabular", "planning, small discrete problems"],
          ["Tabular Q-learning (Watkins 1989)", "$Q^*$ from samples, off-policy", "simple; convergence guarantees", "one entry per state–action; no generalisation", "small discrete MDPs"],
          ["SARSA", "$Q^\\pi$ of the behaviour policy (on-policy)", "safer near hazards while exploring", "learns the ε-greedy policy's values", "online control where exploring is costly"],
          ["DQN (Mnih et al. 2015)", "$Q_\\theta$ with replay + target net", "pixels to actions; sample-efficient replay", "discrete actions only; overestimation; unstable", "Atari, discrete control"],
          ["Double / Dueling / Prioritised DQN, Rainbow (Hessel et al. 2018)", "improved $Q_\\theta$", "less overestimation, better credit; Rainbow combines six extensions", "more parts to tune", "discrete control benchmarks"],
          ["REINFORCE (Williams 1992)", "$\\pi_\\theta$ from Monte-Carlo returns", "simple; any differentiable policy; continuous actions", "very high variance; on-policy; needs many samples", "teaching, simple tasks"],
          ["Actor-critic / A2C / A3C (Mnih et al. 2016)", "$\\pi_\\theta$ and $V_\\phi$", "lower variance via advantages; online updates", "critic bias; on-policy", "parallel simulation"],
          ["PPO (Schulman et al. 2017)", "$\\pi_\\theta$, $V_\\phi$ with clipped surrogate", "robust, simple, several epochs per batch", "on-policy; many implementation details matter", "default on-policy method; RLHF"],
          ["DDPG, TD3, SAC", "deterministic or max-entropy actor + Q critics, off-policy", "continuous actions, replay efficiency (SAC: stable, entropy-regularised)", "more hyper-parameters; Q overestimation (TD3 addresses it)", "robotics, continuous control"],
          ["Model-based (e.g. AlphaZero's MCTS, world models)", "a model of the environment used for planning", "far fewer real samples; lookahead", "model errors compound", "games, robotics with simulators"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Games** — DQN on Atari (Mnih et al. 2015); AlphaGo (Silver et al. 2016) combined policy and value networks with tree search and self-play RL; AlphaZero extended self-play to chess and shogi.",
          "**Language-model alignment** — reinforcement learning from human feedback (Christiano et al. 2017; InstructGPT, Ouyang et al. 2022) fine-tunes a language model with PPO against a learned reward model; see [Alignment](/learn/m20/alignment).",
          "**Robotics and control** — walking and grasping policies trained in simulation (PPO, SAC), then moved to hardware with domain randomisation.",
          "**Chip design** — Mirhoseini et al. (2021) used RL with a graph-neural-network policy for chip floorplanning.",
          "**Recommendation and advertising** — today's recommendation changes tomorrow's user (often handled as contextual bandits, the one-step special case).",
          "**Operations** — resource allocation, scheduling and industrial control, given a simulator or logged data.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Learns from reward alone**: no labels needed; can find strategies people did not think of; handles delayed effects.",
          "**Needs many samples**: thousands to millions of interactions even for small tasks (11k–31k steps for cart-pole here); practical mainly with simulators or cheap interaction.",
          "**Unstable and seed-sensitive**: bootstrapped targets, shifting data and noisy gradients make results vary widely between runs and with small implementation details.",
          "**Reward design is hard**: agents optimise exactly what is rewarded. Sparse rewards give little signal; shaped rewards invite **reward hacking** (exploiting loopholes instead of solving the task).",
          "**Exploration** is still an open problem when rewards are sparse (our REINFORCE gridworld run never found the goal).",
          "**Value-based vs policy-based**: value methods are off-policy and sample-efficient but need a max over actions (hard for continuous actions); policy methods handle any action space and stochastic policies but are on-policy and noisier. Actor-critic combines both.",
          "**Sim-to-real gap and safety**: exploring on real systems can be dangerous or expensive, and policies trained in simulation may fail on hardware.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Treating time-outs as terminal states.** A state cut off at 200 steps is not worth 0 afterwards; done = 1 teaches the agent that nearing the time limit is bad. Bootstrap through cut-offs; set done = 1 only for true terminal states.",
          "**Backpropagating through the TD target.** Treat the target $r + \\gamma\\max Q(s')$ as a constant (`torch.no_grad()` / target network); otherwise the network chases its own predictions.",
          "**Too little exploration.** ε = 0, or ε decayed before any reward is seen, freezes the first mediocre behaviour. With zero-initialised Q-tables, also break ties randomly.",
          "**Forgetting to mask terminal states**: the target at a terminal transition is just $r$. Bootstrapping from the next episode's reset state leaks value across episodes.",
          "**Using the wrong sign in policy gradients**: the policy loss to *minimise* is $-\\log\\pi(a\\mid s)\\hat A$. Also, advantages must not carry gradients into the actor (detach them).",
          "**Reusing on-policy data as if off-policy.** REINFORCE/A2C gradients assume current-policy data; many epochs on old data without importance ratios and clipping (as in PPO) bias and destabilise training.",
          "**Not normalising advantages or observations**, or letting a big value loss dominate a shared network — common reasons PPO fails.",
          "**Judging from one seed.** RL curves differ widely between seeds; report means and spreads over several, plus the evaluation protocol (greedy vs stochastic, number of episodes).",
          "**Reward hacking**: rewarding a proxy (e.g. distance moved) that can be maxed without solving the task. Watch behaviour, not only return.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "How does reinforcement learning differ from supervised learning?", a: "There are no labelled correct actions, only a scalar reward, often delayed. The data depend on the agent's own behaviour (not i.i.d.; the distribution shifts), so it must explore and assign credit for rewards to earlier decisions. The goal is expected cumulative (discounted) reward, not a per-example loss." },
      { t: "qa", level: "beginner", q: "What are a policy, a value function and an action-value function?", a: "A policy $\\pi(a\\mid s)$ maps states to (probabilities of) actions. $V^\\pi(s)$ is the expected return when starting in $s$ and following $\\pi$; $Q^\\pi(s,a)$ is the expected return when taking $a$ in $s$ and then following $\\pi$. $V^\\pi(s) = \\sum_a\\pi(a\\mid s)Q^\\pi(s,a)$, and acting greedily on $Q^*$ is optimal." },
      { t: "qa", level: "beginner", q: "What is the exploration–exploitation trade-off and how does ε-greedy address it?", a: "Exploiting picks the action you currently think is best; exploring tries others that might be better. Pure exploitation can lock in a bad habit; pure exploration never uses what was learned. ε-greedy takes a random action with probability ε and the greedy one otherwise, usually lowering ε during training." },
      { t: "qa", level: "intermediate", q: "Write the Q-learning update and explain why it is off-policy.", a: "$Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma\\max_{a'}Q(s',a') - Q(s,a)]$. The target uses the max over next actions — the value of acting greedily afterwards — whatever the behaviour policy actually does next. So it learns $Q^*$ from any data that explores enough (ε-greedy, old policies, replay). SARSA uses the action actually taken next, so it is on-policy." },
      { t: "qa", level: "intermediate", q: "Why does DQN need experience replay and a target network?", a: "Consecutive transitions are highly correlated and the data shift as the policy changes, so SGD on that stream is unstable and forgets. Replay samples stored transitions at random, decorrelating mini-batches and reusing data. Without a target network, the target $r + \\gamma\\max Q_\\theta(s')$ moves with every update of θ, which can cause oscillation or divergence. A periodically copied $\\theta^-$ holds targets fixed for many steps." },
      { t: "qa", level: "intermediate", q: "Derive why subtracting a baseline does not bias the policy gradient.", a: "$\\mathbb E_{a\\sim\\pi}[b(s)\\nabla\\log\\pi(a\\mid s)] = b(s)\\sum_a\\pi(a\\mid s)\\frac{\\nabla\\pi(a\\mid s)}{\\pi(a\\mid s)} = b(s)\\nabla\\sum_a\\pi(a\\mid s) = b(s)\\nabla 1 = 0$. So the expectation is unchanged, while a good baseline (≈ $V(s)$) cuts the variance of the estimate a lot." },
      { t: "qa", level: "intermediate", q: "Compute the discounted return of rewards (0, 0, 1) with γ = 0.9, and the one-step TD target when r = 0, γ = 0.9 and V(s') = 0.5.", a: "$G = 0 + 0.9\\cdot0 + 0.81\\cdot1 = 0.81$. TD target $= 0 + 0.9\\cdot0.5 = 0.45$." },
      { t: "qa", level: "advanced", q: "Explain PPO's clipped objective and what problem it solves.", a: "Policy-gradient estimates are only valid near the data-collecting policy, so big or many steps on one batch can collapse performance. PPO maximises $\\mathbb E[\\min(\\rho A, \\mathrm{clip}(\\rho, 1-\\epsilon, 1+\\epsilon)A)]$ with $\\rho = \\pi_\\theta/\\pi_{old}$. For positive advantages the objective stops rising once $\\rho > 1+\\epsilon$; for negative ones once $\\rho < 1-\\epsilon$; the min keeps the pessimistic term when the policy moved the wrong way. This approximates TRPO's KL trust region with first-order optimisation, allowing several epochs of minibatch SGD per rollout." },
      { t: "qa", level: "advanced", q: "What is the 'deadly triad' and how do DQN-style methods cope with it?", a: "Function approximation + bootstrapping + off-policy learning together can make value estimates diverge (Sutton and Barto). DQN reduces the risk with target networks (slower targets), replay (more stable data), Huber loss and gradient clipping; Double DQN cuts the max bias; n-step returns lean less on bootstrapping. None removes the risk, which is why deep value-based RL is hyper-parameter-sensitive." },
      { t: "qa", level: "advanced", q: "Compare GAE with λ = 0 and λ = 1.", a: "λ = 0 gives $\\hat A_t = \\delta_t = r + \\gamma V(s_{t+1}) - V(s_t)$: low variance, but biased whenever $V$ is wrong. λ = 1 gives $\\sum_l\\gamma^l\\delta_{t+l} = G_t - V(s_t)$: unbiased with respect to $V$'s errors (V only acts as a baseline) but high variance. A middle λ (0.9–0.97) trades the two; it is an exponentially weighted average of n-step advantage estimates." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Rewards $r_1 = 1$, $r_2 = 0$, $r_3 = 2$, then the episode ends; $\\gamma = 0.9$. What is $G_0$?", answer: 2.62, tolerance: 0.001, explain: "$1 + 0.9\\cdot0 + 0.81\\cdot2 = 2.62$." },
          { kind: "numeric", q: "Q-learning: $Q(s,a) = 0.4$, $r = 0$, $\\gamma = 0.9$, $\\max_{a'}Q(s',a') = 0.8$, $\\alpha = 0.5$. New $Q(s,a)$?", answer: 0.56, tolerance: 0.001, explain: "Target 0.72, TD error 0.32, $0.4 + 0.5\\cdot0.32 = 0.56$." },
          { kind: "numeric", q: "In the gridworld, the goal is 9 moves from S, only reaching G is rewarded (+1) and $\\gamma = 0.9$. What is $V^*(S)$ (3 decimals)?", answer: 0.43, tolerance: 0.001, explain: "The reward comes on the 9th move: $\\gamma^8 = 0.430$." },
          { q: "Which statement about Q-learning is correct?", options: ["It needs the transition probabilities", "It is off-policy: it learns greedy-policy values from ε-greedy behaviour", "It only works with neural networks", "It updates only at the end of each episode"], answer: 1, explain: "It is model-free, works with tables, updates every step and learns $Q^*$ from any behaviour that explores." },
          { q: "What does the DQN target network do?", options: ["Selects exploratory actions", "Provides slowly changing regression targets", "Stores transitions", "Predicts rewards"], answer: 1, explain: "Its weights are copied from the online network only every C steps, so targets do not move with every update." },
          { kind: "numeric", q: "PPO with $\\epsilon = 0.2$: ratio $\\rho = 1.5$, advantage $\\hat A = 2$. Value of the clipped objective term?", answer: 2.4, tolerance: 0.001, explain: "$\\min(1.5\\cdot2,\\ 1.2\\cdot2) = 2.4$." },
          { q: "Why can a baseline be subtracted in REINFORCE?", options: ["It makes the gradient exact", "Its expected contribution $\\mathbb E[b(s)\\nabla\\log\\pi(a\\mid s)]$ is zero", "It is required for convergence of the critic", "It removes the need for exploration"], answer: 1, explain: "The score function has zero mean under the policy, so the baseline changes the variance, not the expectation." },
          { q: "An episode is cut off by a 200-step time limit while the pole is still upright. In the TD target for that last transition you should…", options: ["Use only r (treat as terminal)", "Bootstrap from V(s') as usual", "Discard the transition", "Use a reward of −1"], answer: 1, explain: "A time-out is not a terminal state of the MDP; the future is still worth something." },
        ],
      },
      {
        t: "exercise",
        title: "Bellman backups by hand",
        prompt: "A corridor of states A – B – C – T, where T is terminal. In A, B and C the actions are left and right. Moving right from C enters T with reward +1; all other moves give 0; moving left from A stays in A. With $\\gamma = 0.9$: (a) compute $V^*$ for A, B, C. (b) Evaluate the uniform random policy (left/right with probability 0.5) by solving its Bellman equations. (c) Apply one Q-learning update to $Q(B, \\text{right}) = 0$ after the transition B → C with $r = 0$, $\\alpha = 0.5$, if currently $Q(C, \\text{right}) = 1$ and $Q(C, \\text{left}) = 0.3$.",
        hint: "For (b), write $V(A) = 0.5\\cdot0.9V(A) + 0.5\\cdot0.9V(B)$ and similarly for B and C, then solve the 3 × 3 linear system.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) optimal values", tex: String.raw`V^*(C) = 1,\quad V^*(B) = 0.9,\quad V^*(A) = 0.81` },
              { title: "(b) equations", tex: String.raw`V_A = 0.45V_A + 0.45V_B,\quad V_B = 0.45V_A + 0.45V_C,\quad V_C = 0.45V_B + 0.5` },
              { title: "(b) eliminate", tex: String.raw`V_A = \tfrac{0.45}{0.55}V_B = 0.818182\,V_B,\qquad V_B(1 - 0.45\cdot0.818182) = 0.45V_C \Rightarrow V_B = 0.712230\,V_C` },
              { title: "(b) solve", tex: String.raw`V_C(1 - 0.45\cdot0.712230) = 0.5 \Rightarrow V_C = 0.735839,\quad V_B = 0.524087,\quad V_A = 0.428798` },
              { title: "(c) update", tex: String.raw`y = 0 + 0.9\max(1, 0.3) = 0.9,\quad Q(B,\text{right}) \leftarrow 0 + 0.5(0.9 - 0) = 0.45` },
            ],
          },
          { t: "p", text: "Check: $0.45\\cdot0.524087 + 0.5 = 0.735839 = V_C$. The random policy's values are lower (0.43 vs 0.81 in A) because half its moves go the wrong way. In (c) the target uses the best next action in C, not the one the agent will actually take — that is what makes Q-learning off-policy." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: Double DQN",
        prompt: "Turn `dqn_cartpole.py` into Double DQN: choose the next action with the online network and score it with the target network. Also log the mean of $\\max_a Q_\\theta(s, a)$ over each training mini-batch for both versions, and compare it with the discounted return the agent actually gets. Run both versions with three seeds each.",
        hint: "Only the target line changes: `a2 = online(S2).argmax(1, keepdim=True)` and `y = R + GAMMA * (1 - D) * target(S2).gather(1, a2).squeeze(1)`, all inside `torch.no_grad()`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "with torch.no_grad():\n    a2 = online(S2).argmax(1, keepdim=True)                 # action chosen by the online net\n    y = R + GAMMA * (1 - D) * target(S2).gather(1, a2).squeeze(1)   # evaluated by the target net\nq_log.append(online(S).max(1).values.mean().item())        # track value estimates",
          },
          { t: "p", text: "Standard DQN takes the max of noisy estimates and scores it with the same network, pushing targets up. Double DQN separates choosing from scoring, so its logged Q-values should be lower and closer to the returns really achieved. On a short task like this the effect on episode length may be small and vary between seeds — hence three seeds each, and value estimates as well as scores." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "MDP $(\\mathcal S, \\mathcal A, p, r, \\gamma)$; return $G_t = \\sum_k\\gamma^k r_{t+k+1} = r_{t+1} + \\gamma G_{t+1}$.",
          "$V^\\pi$, $Q^\\pi$; Bellman expectation and optimality equations; value iteration converges because the backup is a γ-contraction.",
          "Q-learning: $Q \\leftarrow Q + \\alpha(r + \\gamma\\max_{a'}Q(s',a') - Q)$; off-policy; needs every pair visited (exploring starts / enough ε).",
          "DQN: neural $Q_\\theta$ + experience replay + target network + Huber loss + ε decay; Double DQN reduces overestimation.",
          "Policy gradient: $\\nabla J = \\mathbb E[\\nabla\\log\\pi_\\theta(a\\mid s)(G_t - b(s))]$ (REINFORCE); baselines cut variance without bias.",
          "Actor-critic: advantage $A = Q - V$, estimated with TD errors / GAE $\\hat A_t = \\delta_t + \\gamma\\lambda\\hat A_{t+1}$.",
          "PPO: maximise $\\mathbb E[\\min(\\rho\\hat A, \\mathrm{clip}(\\rho, 1\\pm\\epsilon)\\hat A)]$ + value loss + entropy bonus, several epochs per rollout.",
          "Results here: gridworld Q-learning optimal in all 21 states after 100 episodes; linear REINFORCE balances cart-pole for 200 steps after ~400 episodes; DQN 143–188 steps after 11k steps; PPO 175–200 after 31k steps.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Alignment](/learn/m20/alignment) — RLHF applies PPO (and related methods) to language models with a learned reward model.",
          "[Graph Neural Networks](/learn/m15/graph-neural-networks) — GNN policies and value functions for states that are graphs (molecules, chips, combinatorial optimisation).",
          "[Multimodal Models](/learn/m15/multimodal) — vision-language models increasingly act as the eyes of robots and agents.",
          "[Experimentation](/learn/m17/experimentation) — RL results depend heavily on the seed, so sound experimental practice matters even more here.",
        ],
      },
    ],
  },
};

export default lesson;
