import type { Lesson } from "@/lib/types";

const REG_NUMPY = String.raw`import numpy as np
from sklearn import metrics as skm


# ---------------- metrics from scratch ----------------
def mae(y, p):
    return np.mean(np.abs(y - p))


def mse(y, p):
    return np.mean((y - p) ** 2)


def rmse(y, p):
    return np.sqrt(mse(y, p))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)              # error of the model
    ss_tot = np.sum((y - np.mean(y)) ** 2)     # error of always predicting the mean of y
    return 1 - ss_res / ss_tot


def mape(y, p):
    return np.mean(np.abs((y - p) / y))


# ---------------- 1) worked example, checked against scikit-learn ----------------
y = np.array([3.0, 5.0, 2.0, 7.0, 4.0])
p = np.array([2.5, 5.5, 2.0, 8.0, 3.0])
ours = [mae(y, p), mse(y, p), rmse(y, p), r2(y, p), mape(y, p)]
sk = [skm.mean_absolute_error(y, p), skm.mean_squared_error(y, p), np.sqrt(skm.mean_squared_error(y, p)),
      skm.r2_score(y, p), skm.mean_absolute_percentage_error(y, p)]
print("MAE {:.4f}  MSE {:.4f}  RMSE {:.4f}  R2 {:.4f}  MAPE {:.4f}".format(*ours))
print("matches sklearn:", np.allclose(ours, sk))
print(f"R2 of constant prediction 6.0: {r2(y, np.full(5, 6.0)):.4f}   of the mean 4.2: {r2(y, np.full(5, 4.2)):.4f}")

# ---------------- 2) residual analysis: a straight line fitted to curved, heteroscedastic data ----------------
rng = np.random.default_rng(0)
x = rng.uniform(0, 4, 400)
y = 1 + 0.5 * x ** 2 + rng.normal(0, 0.2 + 0.3 * x)      # true curve is quadratic; noise grows with x
tr, te = np.arange(300), np.arange(300, 400)

for degree in (1, 2):
    coef = np.polyfit(x[tr], y[tr], degree)
    pred = np.polyval(coef, x)
    res = y[te] - pred[te]
    print(f"\ndegree {degree}: test MAE {mae(y[te], pred[te]):.3f}  RMSE {rmse(y[te], pred[te]):.3f}  R2 {r2(y[te], pred[te]):.3f}")
    bins = np.digitize(x[te], [1, 2, 3])                # four x-ranges: [0,1) [1,2) [2,3) [3,4]
    print("  mean residual per x-bin:", np.round([res[bins == b].mean() for b in range(4)], 3))
    print("  residual std per x-bin: ", np.round([res[bins == b].std() for b in range(4)], 3))

# ---------------- 3) negative R2 on test: a model that extrapolates badly ----------------
coef5 = np.polyfit(x[x < 3], y[x < 3], 5)                # fit only x < 3, then predict x >= 3
far = x >= 3
print(f"\ndegree-5 fit on x<3, evaluated on x>=3: R2 {r2(y[far], np.polyval(coef5, x[far])):.3f}"
      f"   (predicting the test mean would give 0.000)")

# ---------------- 4) MAPE blows up when targets approach zero ----------------
y_small = np.array([100.0, 50.0, 10.0, 1.0, 0.1])
p_small = y_small + 1.0                                   # every prediction is off by exactly 1
print(f"\nall errors = 1.0  ->  MAE {mae(y_small, p_small):.2f}  MAPE {mape(y_small, p_small):.2%}")
print("per-example APE:", np.round(np.abs((y_small - p_small) / y_small), 3))`;

const REG_NUMPY_OUT = String.raw`MAE 0.6000  MSE 0.5000  RMSE 0.7071  R2 0.8311  MAPE 0.1319
matches sklearn: True
R2 of constant prediction 6.0: -1.0946   of the mean 4.2: 0.0000

degree 1: test MAE 0.777  RMSE 1.009  R2 0.838
  mean residual per x-bin: [ 0.351 -0.367 -0.72   0.6  ]
  residual std per x-bin:  [0.618 0.687 0.797 1.211]

degree 2: test MAE 0.642  RMSE 0.816  R2 0.894
  mean residual per x-bin: [-0.128  0.19  -0.165  0.242]
  residual std per x-bin:  [0.443 0.685 0.801 1.133]

degree-5 fit on x<3, evaluated on x>=3: R2 -0.268   (predicting the test mean would give 0.000)

all errors = 1.0  ->  MAE 1.00  MAPE 222.60%
per-example APE: [ 0.01  0.02  0.1   1.   10.  ]`;

const REG_TORCH = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- data: y = 2x + 1 plus RIGHT-SKEWED noise (exponential, mean 1, median ln 2 = 0.693) ----
def make(n):
    x = torch.rand(n, 1) * 4
    noise = torch.distributions.Exponential(1.0).sample((n, 1))
    return x, 2 * x + 1 + noise

x_tr, y_tr = make(2000)
x_te, y_te = make(2000)


def metrics(y, p):
    err = y - p
    mae = err.abs().mean()
    mse = (err ** 2).mean()
    r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return mae.item(), mse.item(), mse.sqrt().item(), r2.item()


for name, loss_fn in [("MSE loss", nn.MSELoss()), ("L1 loss ", nn.L1Loss())]:
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss_fn(model(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        p = model(x_te)
        mae, mse, rmse, r2 = metrics(y_te, p)
        offset = (p - 2 * x_te).mean().item() - 1          # how far above the noise-free line 2x + 1
        med_res = (y_te - p).median().item()
    print(f"{name}: slope {model.weight.item():.3f}  offset above 2x+1 {offset:.3f}  "
          f"test MAE {mae:.4f}  MSE {mse:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}  median residual {med_res:+.3f}")`;

const REG_TORCH_OUT = String.raw`MSE loss: slope 1.969  offset above 2x+1 1.001  test MAE 0.7427  MSE 1.0231  RMSE 1.0115  R2 0.8373  median residual -0.329
L1 loss : slope 2.004  offset above 2x+1 0.700  test MAE 0.6954  MSE 1.1114  RMSE 1.0542  R2 0.8233  median residual -0.029`;

const REG_KERAS = String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
x = rng.uniform(0, 4, (4000, 1)).astype("float32")
y = (2 * x + 1 + rng.exponential(1.0, (4000, 1))).astype("float32")

tf.keras.utils.set_random_seed(0)
model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1)])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.05),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
             tf.keras.metrics.RootMeanSquaredError(name="rmse"),
             tf.keras.metrics.R2Score(name="r2")],          # R2Score exists in recent Keras versions
)
model.fit(x[:2000], y[:2000], epochs=50, batch_size=200, verbose=0)
print(dict(zip(model.metrics_names, model.evaluate(x[2000:], y[2000:], verbose=0))))`;

const lesson: Lesson = {
  id: "m17/regression-metrics",
  title: "Regression Metrics",
  summary: "How to grade a model that predicts numbers: MAE, MSE, RMSE and R² (and why R² can be negative), the traps of MAPE, and residual analysis — looking at the errors themselves to see *how* the model is wrong, not just how much.",
  minutes: 45,
  level: "beginner",
  prerequisites: ["m17/data-splits", "m05/regression-losses", "m03/linear-logistic-regression"],
  next: ["m17/experimentation", "m17/classification-metrics", "m18/mlops"],
  terms: ["mae", "mse", "rmse", "r-squared", "mape", "residual", "residual-analysis", "heteroscedasticity"],
  sections: {
    intro: [
      { t: "p", text: "A regression model predicts a number: a price, a temperature, a delivery time. Its error on one example is the **residual**, $e_i = y_i - \\hat y_i$ (true value minus prediction). A metric squeezes all residuals on held-out data into one number, and your choice decides what \"good\" means. Is one error of 10 worse than ten errors of 1? Are errors in the target's units, in percent, or relative to a dumb baseline?" },
      { t: "p", text: "Four metrics cover most real work:" },
      { t: "list", items: [
        "**MAE** (mean absolute error) — the average size of an error, in the target's units.",
        "**MSE** (mean squared error) and its square root **RMSE** — punish big errors far more than small ones.",
        "**R²** (coefficient of determination) — compares your squared error with always guessing the average. No units.",
        "**MAPE** (mean absolute percentage error) — error as a percent of the true value. Business-friendly, but breaks near zero.",
      ] },
      { t: "p", text: "A metric is not a loss, though they overlap. The loss is what training minimizes (so it must be differentiable); the metric is what you report and decide with. Losses are covered in [Regression Losses](/learn/m05/regression-losses). Here we focus on evaluation and **residual analysis**, which no single number replaces." },
      { t: "callout", kind: "key", text: "Report an error in the target's units (MAE or RMSE), a score against a baseline (R² or skill vs a naive forecast), and look at the residuals. Prefer RMSE when big errors cost disproportionately more." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Train/validation/test splits](/learn/m17/data-splits) — metrics use held-out data; training R² says little about new data.",
        "[Regression losses](/learn/m05/regression-losses) — MSE is minimized by the conditional mean, MAE by the conditional median. This explains which model wins which metric.",
        "[Linear regression](/learn/m03/linear-logistic-regression) — least squares, fitted values and residuals.",
        "Mean, variance and standard deviation from [Probability](/learn/m02/probability) — R² is one minus a ratio of two variances.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Arriving at the station", text: "Train A is 2 minutes late every day. Train B is on time 9 days out of 10 but 20 minutes late on the tenth. Both average 2 minutes late (MAE). But with a 10-minute connection, train B makes you miss it. **RMSE** sees this: 2 minutes for A, $\\sqrt{400/10} = 6.3$ minutes for B. **R²** asks: compared with assuming the average delay every day, how much of the daily variation does the timetable predict?" },
      { t: "p", text: "MAE counts every unit of error the same. MSE squares errors, so one error of 10 counts as much as 100 errors of 1 — the worst predictions and outliers rule it. RMSE takes the square root to return to the target's units. It is always ≥ MAE, and the gap shows how uneven the errors are." },
      { t: "p", text: "R² is relative: 1 is perfect, 0 is \"no better than predicting the mean of the evaluation targets\", and **negative** is worse than that — which really happens on test data. Residual plots answer what numbers can't: is the error random noise, or a pattern (a missed curve, errors growing with the target) a better model could fix?" },
    ],
    mechanism: [
      { t: "steps", items: [
        { title: "Predict on held-out data", text: "Compute $\\hat y_i$ in the **original units**. Undo any target scaling or log transform first, unless you mean to evaluate on that scale." },
        { title: "Compute residuals", text: "$e_i = y_i - \\hat y_i$. A positive residual means the model predicted too low." },
        { title: "Aggregate", text: "MAE = mean of $|e_i|$; MSE = mean of $e_i^2$; RMSE = $\\sqrt{\\text{MSE}}$; R² = $1 - \\sum e_i^2 / \\sum (y_i - \\bar y)^2$; MAPE = mean of $|e_i / y_i|$." },
        { title: "Compare against baselines", text: "Predict the training mean, the last value (persistence) for time series, or use a plain linear model. A deep model that can't beat these is useless." },
        { title: "Analyse residuals", text: "Plot residuals against predictions and key features. Bin them and check each bin's mean (bias) and spread. Draw a histogram. Inspect the biggest errors one by one." },
        { title: "Break down by segment", text: "Compute metrics per region, product, period or range of $y$. A good overall number can hide a failing segment." },
      ] },
      { t: "h", text: "Reading residual patterns" },
      { t: "table", head: ["What residuals look like (vs prediction or feature)", "What it means", "Usual fix"], rows: [
        ["Random cloud around 0, even spread", "Structure captured; the rest is noise", "Nothing structural; more data or features"],
        ["U-shape or wave (mean residual changes with x)", "Systematic misfit: a missing curve or interaction", "Add features or use a more flexible model"],
        ["Funnel (spread grows with x or ŷ)", "Heteroscedasticity: noise grows with the level", "Model log(y), use relative metrics, predict variance (Gaussian NLL)"],
        ["Overall mean residual ≠ 0", "Bias: always predicting too high or too low", "Check target transforms, intercept, train/test shift"],
        ["Pattern over time", "Drift, seasonality or autocorrelation", "Time features, retraining, time-series models"],
        ["A few huge residuals", "Outliers, data errors or rare situations", "Inspect them; robust loss; separate model"],
      ] },
    ],
    math: [
      { t: "h", text: "Definitions" },
      { t: "p", text: "Take $n$ evaluation examples with true values $y_i$, predictions $\\hat y_i$, residuals $e_i = y_i - \\hat y_i$ and mean target $\\bar y = \\frac1n\\sum_i y_i$:" },
      { t: "math", tex: String.raw`\text{MAE} = \frac1n\sum_{i=1}^n |e_i|,\qquad \text{MSE} = \frac1n\sum_{i=1}^n e_i^2,\qquad \text{RMSE} = \sqrt{\text{MSE}}` },
      { t: "math", tex: String.raw`R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}},\qquad SS_{\text{res}} = \sum_{i=1}^n e_i^2,\qquad SS_{\text{tot}} = \sum_{i=1}^n (y_i - \bar y)^2` },
      { t: "p", text: "In words: $SS_{\\text{res}}$ is your model's total squared error; $SS_{\\text{tot}}$ is that of always guessing $\\bar y$. R² is the share of baseline error your model removes." },
      { t: "math", tex: String.raw`\text{MAPE} = \frac1n\sum_{i=1}^n \left|\frac{y_i - \hat y_i}{y_i}\right|` },
      { t: "h", text: "How they relate" },
      { t: "list", items: [
        "$\\text{MAE} \\le \\text{RMSE} \\le \\sqrt{n}\\,\\text{MAE}$. The left part is Jensen's inequality ($\\text{mean}(|e|)^2 \\le \\text{mean}(e^2)$), equal only when all $|e_i|$ are equal. A large RMSE/MAE ratio means a few large errors.",
        "Divide top and bottom of R² by $n$: $R^2 = 1 - \\text{MSE}/\\text{Var}(y)$, where $\\text{Var}(y)$ is the (population) variance of the evaluation targets — the MSE of the constant guess $\\bar y$. So R² is the fraction of target variance explained.",
        "Bias–variance view: $\\text{MSE} = \\bar e^2 + \\text{Var}(e)$ — the squared mean residual (systematic bias) plus the spread of the residuals.",
      ] },
      { t: "h", text: "Why R² can be negative" },
      { t: "p", text: "Nothing forces $SS_{\\text{res}}$ below $SS_{\\text{tot}}$. On the training set, a least-squares linear model with an intercept can always fall back to predicting $\\bar y$, so its training R² lies in $[0, 1]$ and equals the squared correlation between $y$ and $\\hat y$. On **new data** — or for any model not fitted by least squares with an intercept — predictions can be worse than the evaluation-set mean, so $SS_{\\text{res}} > SS_{\\text{tot}}$ and $R^2 < 0$." },
      { t: "p", text: "Caveat: the baseline is the mean of the *evaluation* targets, which a real model can't know in advance. So negative test R² means \"worse than a constant that peeked at the answers\" — not necessarily useless, but a strong warning." },
      { t: "h", text: "Which metric rewards which prediction?" },
      { t: "p", text: "For a fixed input, the constant $c$ minimizing $\\mathbb{E}[(Y-c)^2]$ is $c = \\mathbb{E}[Y]$, the mean; minimizing $\\mathbb{E}|Y-c|$ gives the median. So an MSE-trained model aims at the conditional mean and wins on RMSE; an L1-trained model aims at the conditional median and wins on MAE. With skewed noise these differ — see the PyTorch code below (derivations in [Regression Losses](/learn/m05/regression-losses))." },
      { t: "h", text: "MAPE traps" },
      { t: "list", items: [
        "**Undefined at $y_i = 0$** and explodes near zero: an error of 1 on a target of 0.1 is 1000%.",
        "**Lopsided**: for positive targets, predicting too low is capped at 100% APE (predict 0), but too high has no limit. So minimizing MAPE biases predictions low.",
        "Alternatives: **sMAPE** (divides by $(|y|+|\\hat y|)/2$; bounded but still unstable near zero), **WAPE** $= \\sum|e_i| / \\sum|y_i|$ (MAE relative to the total; robust and popular in demand forecasting), or **MASE** (MAE divided by a naive forecast's MAE; Hyndman and Koehler, 2006, *Another look at measures of forecast accuracy*).",
      ] },
      { t: "h", text: "Worked example" },
      { t: "p", text: "True values $y = (3, 5, 2, 7, 4)$, predictions $\\hat y = (2.5, 5.5, 2, 8, 3)$." },
      { t: "steps", items: [
        { title: "Residuals", tex: String.raw`e = y - \hat y = (0.5,\ -0.5,\ 0,\ -1,\ 1)` },
        { title: "MAE", tex: String.raw`\text{MAE} = \frac{0.5 + 0.5 + 0 + 1 + 1}{5} = \frac{3}{5} = 0.6` },
        { title: "MSE and RMSE", tex: String.raw`\text{MSE} = \frac{0.25 + 0.25 + 0 + 1 + 1}{5} = \frac{2.5}{5} = 0.5,\qquad \text{RMSE} = \sqrt{0.5} = 0.7071` },
        { title: "Total sum of squares", text: "$\\bar y = 21/5 = 4.2$.", tex: String.raw`SS_{\text{tot}} = 1.2^2 + 0.8^2 + 2.2^2 + 2.8^2 + 0.2^2 = 1.44 + 0.64 + 4.84 + 7.84 + 0.04 = 14.8` },
        { title: "R²", tex: String.raw`R^2 = 1 - \frac{2.5}{14.8} = 1 - 0.1689 = 0.8311` },
        { title: "MAPE", tex: String.raw`\text{MAPE} = \tfrac15\left(\tfrac{0.5}{3} + \tfrac{0.5}{5} + \tfrac{0}{2} + \tfrac{1}{7} + \tfrac{1}{4}\right) = \tfrac15(0.1667 + 0.1 + 0 + 0.1429 + 0.25) = 0.1319` },
        { title: "Always predicting 6", text: "Residuals $(-3, -1, -4, 1, -2)$, $SS_{\\text{res}} = 9 + 1 + 16 + 1 + 4 = 31$.", tex: String.raw`R^2 = 1 - \frac{31}{14.8} = -1.0946 \quad (\text{predicting } \bar y = 4.2 \text{ gives exactly } 0)` },
      ] },
    ],
    visual: [
      { t: "p", text: "The loss explorer shows the penalty each metric gives one residual — the curve's shape is what that metric cares about." },
      { t: "sim", id: "loss-functions", caption: "Open the regression panel: MSE (r²) vs MAE (|r|), Huber and log-cosh." },
      { t: "list", items: [
        "Compare MSE and MAE at residuals 0.5 and 3. MSE is smaller below 1 and grows quadratically above, so big errors dominate RMSE but not MAE.",
        "In the gradient curves, MAE's gradient has constant size; MSE's grows with the residual. An MSE-trained model spends its effort on the biggest errors.",
        "Huber is quadratic near zero and linear in the tails — a middle ground when rare outliers shouldn't dominate.",
      ] },
      { t: "p", text: "For residual patterns, revisit the fitting simulator: a degree-1 fit to curved data leaves systematic, wave-shaped residuals; the right degree leaves noise." },
      { t: "sim", id: "fitting", caption: "Compare the gaps between curve and points at low degree (systematic residuals) and at the best degree (random residuals)." },
    ],
    scratch: [
      { t: "p", text: "Five metrics from scratch (checked against scikit-learn), residuals of a too-simple model on curved data with growing noise, a negative test R² from bad extrapolation, and MAPE near zero." },
      { t: "code", lang: "python", title: "regression_metrics_from_scratch.py", code: REG_NUMPY, output: REG_NUMPY_OUT },
      { t: "list", items: [
        "The worked example's numbers (0.6, 0.5, 0.7071, 0.8311, 0.1319) match scikit-learn exactly.",
        "**Degree 1 has R² = 0.838**, which sounds fine. But its binned mean residuals go $+0.35, -0.37, -0.72, +0.60$: positive at both ends, negative in the middle — a straight line through a curve. Degree 2's bin means are smaller with no systematic shape, and test RMSE drops from 1.009 to 0.816.",
        "In both, residual std grows from about 0.4–0.6 in the first bin to over 1.1 in the last: **heteroscedasticity** (noise that changes with the level). It is in the data (noise std $0.2 + 0.3x$), not a model error. No mean model removes it, and RMSE is dominated by high $x$.",
        "A degree-5 polynomial fitted only on $x < 3$ and tested on $x \\ge 3$ gets **R² = −0.268**: its extrapolation is worse than the constant test mean.",
        "Every prediction is off by exactly 1, so MAE = 1.00 — yet MAPE = 222.6%. All of that comes from the target 0.1, whose APE is 1000%.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Two identical linear models train on data with **right-skewed** noise (exponential, mean 1, median $\\ln 2 = 0.693$), one with MSE loss and one with L1. Both are scored on every metric with tensor operations." },
          { t: "code", lang: "python", title: "loss_vs_metric_pytorch.py", code: REG_TORCH, output: REG_TORCH_OUT },
          { t: "list", items: [
            "The MSE-trained line sits 1.001 above the noise-free line (the noise **mean**); the L1 line sits 0.700 above (the noise **median**, 0.693). Each learned the statistic its loss aims at.",
            "So each wins its own metric: the MSE model has lower MSE/RMSE (1.023 vs 1.111) and higher R²; the L1 model has lower MAE (0.695 vs 0.743) and a median residual near 0.",
            "Takeaway: **train with the loss that matches the metric you'll be judged on**, or at least know which one you're optimizing.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras has streaming regression metrics. `R2Score` exists only in newer Keras/TensorFlow; on older versions, compute R² from MSE and the target variance." },
          { t: "code", lang: "python", title: "regression_metrics_keras.py (not executed in CI)", code: REG_KERAS, output: "Prints a dict with the test loss (MSE), mae, rmse and r2 of the fitted line — values close to the PyTorch MSE model's; the exact numbers depend on the TensorFlow version." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Metric", "Pros", "Cons", "Use when"], rows: [
        ["MAE", "Target units; robust to outliers; easy to explain", "Not differentiable at 0 (as a loss); ignores error concentration", "All errors cost in proportion; outliers present"],
        ["MSE", "Smooth; matches Gaussian likelihood; splits into bias² + variance", "Squared units; dominated by outliers", "Training loss; large errors very costly"],
        ["RMSE", "Target units; punishes large errors", "Sensitive to outliers; can't be added across subsets", "Standard reporting when large errors matter"],
        ["R²", "No units; compares to a mean baseline", "Depends on evaluation-set target variance; can be negative; not comparable across datasets", "Explaining variance; quick sanity check"],
        ["MAPE", "Percent; scale-free; business-friendly", "Undefined at 0; explodes near 0; biased toward predicting low", "Strictly positive targets far from 0"],
        ["WAPE / sMAPE", "Scale-free, steadier than MAPE", "WAPE dominated by large items; sMAPE still unstable near 0", "Demand/sales forecasting"],
        ["MASE", "Scale-free; compares to a naive forecast", "Needs a sensible naive baseline", "Time series across many scales"],
        ["RMSLE", "Errors on log scale = relative errors; punishes predicting low more", "Requires y > −1; harder to interpret", "Targets spanning orders of magnitude (prices, counts)"],
        ["Quantile (pinball) loss / coverage", "Evaluates prediction intervals", "Needs quantile predictions", "Forecasting with uncertainty"],
        ["Gaussian NLL", "Evaluates predicted mean and variance", "Needs a variance output", "Probabilistic regression"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Demand and sales forecasting** uses WAPE, MASE and quantile losses: volumes span orders of magnitude, and planners need ranges, not single guesses.",
        "**Weather and energy-load forecasting** reports RMSE/MAE plus *skill scores* against persistence or climatology — R² with a better baseline.",
        "**Real-estate and pricing models** often use RMSLE or relative errors: a 10 000 error matters more on a cheap item than an expensive one.",
        "**Vision regression** (depth, keypoints, box coordinates) reports MAE/RMSE in pixels or metres, plus relative error for depth.",
        "**Scientific and medical measurement** plots residuals against the measured range (Bland–Altman-style) to catch level-dependent bias.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**MAE vs RMSE** is a cost judgment: does error cost grow linearly or quadratically? Reporting both, plus their ratio, says more than either alone.",
        "**R² is relative** to the spread of the evaluation targets: the same model scores higher on a test set with a wider range of $y$. Compare R² only on the same data.",
        "**Scale-free metrics** (MAPE, WAPE, MASE) combine series of very different sizes, but each fails near zero or with intermittent demand.",
        "**Single numbers hide structure.** Residuals and per-segment metrics reveal it, at the cost of manual work.",
        "**Point metrics ignore uncertainty.** If decisions need risk estimates, evaluate predicted distributions (pinball loss, coverage, NLL).",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Computing metrics on scaled or log-transformed targets** and reporting them as original units. Undo the transform, or state the scale.",
        "**Reporting training R².** It is optimistic and, for least squares with an intercept, never negative. Report test or CV R².",
        "**Believing R² is always between 0 and 1.** On held-out data it can be any value ≤ 1; negative means worse than predicting the evaluation mean.",
        "**Using MAPE with zeros or tiny targets.** It is undefined or ruled by a few points; use WAPE, MASE or MAE.",
        "**Comparing RMSE across datasets with different scales.** Normalize (by the target's standard deviation, i.e. via R²) or use scale-free metrics.",
        "**Averaging RMSE over folds or groups** as if additive. Average MSE, then take the root — or say you average per-fold RMSE.",
        "**Skipping residual plots.** A decent R² can hide systematic misfit (see degree 1) — exactly the clue you need to improve.",
        "**Ignoring the baseline.** An RMSE of 3 means nothing until compared with predicting the mean or last value.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between MAE and RMSE?", a: "MAE averages absolute errors; RMSE is the square root of the average squared error. Both use the target's units. RMSE punishes large errors more, so RMSE ≥ MAE; a large ratio points to a few big errors or outliers. MAE is more robust and easier to explain." },
      { t: "qa", level: "beginner", q: "What does R² measure?", a: "The fraction of the target's variance the model explains: $R^2 = 1 - SS_{res}/SS_{tot}$. 1 is perfect, 0 equals predicting the mean of the targets, and negative means worse than that." },
      { t: "qa", level: "beginner", q: "What is a residual and why plot it?", a: "The residual is $y - \\hat y$. Plotting them against predictions or features shows what single numbers hide: curves (missing non-linearity), funnels (variance grows with the level), drift over time, outliers." },
      { t: "qa", level: "intermediate", q: "How can R² be negative?", a: "Nothing caps $SS_{res}$ at $SS_{tot}$. On test data (or for models not fitted by least squares with an intercept), predictions can be worse than the evaluation targets' constant mean, so $SS_{res} > SS_{tot}$. Usual causes: overfitting, extrapolation, distribution shift, or a bug like evaluating on the wrong target scale." },
      { t: "qa", level: "intermediate", q: "What are the problems with MAPE?", a: "It is undefined when any target is 0 and explodes for small targets. It punishes predicting high more than low (under-predicting a positive target is at most 100% APE), so optimized forecasts drift low. It also overweights small-volume items. Alternatives: WAPE, MASE, sMAPE (with caveats) or RMSLE." },
      { t: "qa", level: "intermediate", q: "You trained with MSE but your stakeholders judge MAE. What happens and what do you do?", a: "MSE training aims at the conditional mean; MAE is minimized by the conditional median. With skewed noise these differ, so the model is systematically off for MAE (the PyTorch demo: offsets 1.001 vs 0.700). Train with L1 (or Huber with small δ, or a 0.5-quantile loss), or adjust predictions afterwards, and select models by validation MAE." },
      { t: "qa", level: "advanced", q: "Show that the training R² of OLS with an intercept equals the squared correlation between y and ŷ.", a: "With an intercept, OLS residuals are orthogonal to the fitted values and sum to 0. So $SS_{tot} = SS_{reg} + SS_{res}$ with $SS_{reg} = \\sum(\\hat y_i - \\bar y)^2$, giving $R^2 = SS_{reg}/SS_{tot}$. Also $\\mathrm{Cov}(y, \\hat y) = \\mathrm{Cov}(\\hat y + e, \\hat y) = \\mathrm{Var}(\\hat y)$, so $\\mathrm{corr}^2 = \\mathrm{Var}(\\hat y)^2/(\\mathrm{Var}(y)\\mathrm{Var}(\\hat y)) = \\mathrm{Var}(\\hat y)/\\mathrm{Var}(y) = R^2$. Neither identity holds on test data or without an intercept." },
      { t: "qa", level: "advanced", q: "Your residuals fan out as the prediction grows. What does that imply for metrics and modelling?", a: "Heteroscedasticity: noise size depends on the level. RMSE/MSE get dominated by the high-level region; MAE less so. Options: model $\\log y$ (makes multiplicative noise additive; evaluate with RMSLE or relative errors), weight the loss, or also predict a variance (Gaussian NLL) and check interval coverage. Report metrics per range of $y$." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Residuals are (2, −1, 0, −3, 2). What is the MAE?", answer: 1.6, tolerance: 0.001, explain: "$(2+1+0+3+2)/5 = 8/5 = 1.6$." },
        { kind: "numeric", q: "Same residuals (2, −1, 0, −3, 2). What is the RMSE? (3 decimals)", answer: 1.897, tolerance: 0.001, explain: "MSE $= (4+1+0+9+4)/5 = 3.6$; RMSE $= \\sqrt{3.6} = 1.897$." },
        { kind: "numeric", q: "A model's test MSE is 4 and the test targets' variance is 16. What is R²?", answer: 0.75, tolerance: 0.001, explain: "$R^2 = 1 - \\text{MSE}/\\text{Var}(y) = 1 - 4/16 = 0.75$." },
        { kind: "numeric", q: "Test MSE is 25 and the test targets' variance is 20. What is R²?", answer: -0.25, tolerance: 0.001, explain: "$1 - 25/20 = -0.25$: worse than predicting the mean." },
        { q: "Which statement about RMSE and MAE is always true?", options: ["RMSE ≤ MAE", "RMSE ≥ MAE", "RMSE = MAE²", "They are uncorrelated"], answer: 1, explain: "By Jensen's inequality, the mean of squares is at least the square of the mean. So RMSE ≥ MAE, equal only when all absolute errors are the same." },
        { q: "True value y = 0.5, prediction 1.5. What is this example's absolute percentage error?", options: ["50%", "100%", "200%", "66.7%"], answer: 2, explain: "$|0.5 - 1.5|/0.5 = 2 = 200\\%$ — small targets make APE explode." },
        { q: "Residuals are positive for small and large x and negative in the middle. What is the most likely problem?", options: ["Heteroscedastic noise", "A missing non-linear (e.g. quadratic) effect", "Data leakage", "Too much regularization of the intercept"], answer: 1, explain: "A U-shaped mean residual means the model's form misses a curve." },
      ] },
      { t: "exercise", title: "Metrics by hand", prompt: "True values $y = (10, 12, 8, 15, 5)$, predictions $\\hat y = (11, 11, 9, 12, 7)$. Compute MAE, MSE, RMSE, R² and MAPE.", hint: "$\\bar y = 10$.", solution: [
        { t: "steps", items: [
          { title: "Residuals", tex: String.raw`e = (-1,\ 1,\ -1,\ 3,\ -2)` },
          { title: "MAE, MSE, RMSE", tex: String.raw`\text{MAE} = 8/5 = 1.6,\quad \text{MSE} = (1+1+1+9+4)/5 = 3.2,\quad \text{RMSE} = 1.7889` },
          { title: "R²", tex: String.raw`SS_{\text{tot}} = 0 + 4 + 4 + 25 + 25 = 58,\quad SS_{\text{res}} = 16,\quad R^2 = 1 - 16/58 = 0.7241` },
          { title: "MAPE", tex: String.raw`\tfrac15(0.1 + 0.0833 + 0.125 + 0.2 + 0.4) = 0.1817` },
        ] },
        { t: "code", lang: "python", title: "check_regression.py", code: String.raw`import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

y = np.array([10, 12, 8, 15, 5.0])
p = np.array([11, 11, 9, 12, 7.0])
print(mean_absolute_error(y, p), mean_squared_error(y, p), round(np.sqrt(mean_squared_error(y, p)), 4),
      round(r2_score(y, p), 4), round(mean_absolute_percentage_error(y, p), 4))`, output: "1.6 3.2 1.7889 0.7241 0.1817" },
      ] },
      { t: "exercise", title: "A residual-analysis report", prompt: "Write `residual_report(x, y, pred, n_bins=4)`. Split examples into equal-count bins of `x` (use quantiles) and print, per bin, the count, mean residual, residual standard deviation and RMSE. Also print the overall RMSE/MAE ratio. Run it on the degree-1 and degree-2 fits from the from-scratch program and explain what changes.", hint: "`np.quantile(x, np.linspace(0, 1, n_bins + 1))` gives bin edges; `np.digitize` assigns bins.", solution: [
        { t: "code", lang: "python", code: String.raw`def residual_report(x, y, pred, n_bins=4):
    res = y - pred
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        r = res[which == b]
        print(f"x in [{edges[b]:.2f}, {edges[b + 1]:.2f}]  n={r.size:3d}  mean {r.mean():+.3f}  "
              f"std {r.std():.3f}  rmse {np.sqrt(np.mean(r ** 2)):.3f}")
    print(f"RMSE/MAE = {np.sqrt(np.mean(res ** 2)) / np.mean(np.abs(res)):.3f}")` },
        { t: "p", text: "For degree 1, the mean residual changes sign across bins in a U-shape (systematic misfit); for degree 2 the means stay near zero. In both, per-bin std and RMSE grow with $x$ because the noise grows. A better mean model can't fix that — only model it (e.g. a variance output) or report it per range." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Residual $e_i = y_i - \\hat y_i$. MAE $= \\frac1n\\sum|e_i|$, MSE $= \\frac1n\\sum e_i^2$, RMSE $= \\sqrt{\\text{MSE}}$; MAE ≤ RMSE.",
        "$R^2 = 1 - SS_{res}/SS_{tot} = 1 - \\text{MSE}/\\text{Var}(y)$: 1 is perfect, 0 = predicting the mean, negative = worse than the mean (possible on test data).",
        "MSE is minimized by the mean, MAE by the median — train with the loss that matches the metric you report.",
        "MAPE $= \\frac1n\\sum|e_i/y_i|$: undefined at 0, explodes near 0, biased toward predicting low; prefer WAPE/MASE/RMSLE where they fit.",
        "Always compare with a baseline and check residuals (vs prediction, features, time): patterns = fixable misfit; funnels = heteroscedasticity.",
        "Compute metrics in original units, on held-out data, and per segment.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — use these metrics as the target of systematic searches.",
        "[Classification Metrics, ROC and PR Curves](/learn/m17/classification-metrics) — the same job for discrete labels.",
        "[Monitoring, Drift and MLOps](/learn/m18/mlops) — residuals over time in production are the first sign of drift.",
      ] },
    ],
  },
};

export default lesson;
