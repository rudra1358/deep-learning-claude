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
  summary: "How to measure a model that predicts numbers: MAE, MSE, RMSE and R² (including why R² can be negative), the pitfalls of MAPE, and residual analysis — the diagnostic that tells you *how* a regression model is wrong, not just how much.",
  minutes: 45,
  level: "beginner",
  prerequisites: ["m17/data-splits", "m05/regression-losses", "m03/linear-logistic-regression"],
  next: ["m17/experimentation", "m17/classification-metrics", "m18/mlops"],
  terms: ["mae", "mse", "rmse", "r-squared", "mape", "residual", "residual-analysis", "heteroscedasticity"],
  sections: {
    intro: [
      { t: "p", text: "A regression model predicts a continuous quantity — a price, a temperature, a delivery time, a bounding-box coordinate. Its error on one example is the **residual** $e_i = y_i - \\hat y_i$. A regression metric condenses all residuals on a held-out set into one number. Which number you choose decides what \"good\" means: is one error of 10 worse than ten errors of 1? Are errors measured in the target's units, in percent, or relative to a trivial baseline?" },
      { t: "p", text: "Four metrics cover most practice. **MAE** (mean absolute error) is the average size of an error in the target's units. **MSE** (mean squared error) and its square root **RMSE** penalize large errors disproportionately. **R²** (coefficient of determination) compares the model's squared error with that of always predicting the mean, giving a unit-free score. **MAPE** (mean absolute percentage error) expresses errors relative to the target — intuitive for business reporting but treacherous near zero." },
      { t: "p", text: "Metrics are not losses, although they overlap: the loss is what the optimizer minimizes during training (and must be differentiable); the metric is what you report and decide on. The loss side — MSE, MAE, Huber, quantile losses and the statistics they estimate — is covered in [Regression Losses](/learn/m05/regression-losses). Here we focus on evaluation, and on **residual analysis**, which no single number replaces." },
      { t: "callout", kind: "key", text: "Report an error metric in the target's units (MAE or RMSE), a baseline-relative score (R² or skill vs a naive forecast), and look at the residuals. Choose MAE vs RMSE according to whether large errors are disproportionately costly." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Train/validation/test splits](/learn/m17/data-splits) — metrics are computed on held-out data; R² on the training set says little about generalization.",
        "[Regression losses](/learn/m05/regression-losses) — MSE is minimized by the conditional mean, MAE by the conditional median; this lesson relies on that fact to explain which model wins which metric.",
        "[Linear regression](/learn/m03/linear-logistic-regression) — least squares, fitted values and residuals.",
        "Mean, variance and standard deviation from [Probability](/learn/m02/probability) — R² is literally one minus a ratio of two variances.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Arriving at the station", text: "A train that is 2 minutes late every day and a train that is on time 9 days out of 10 but 20 minutes late on the tenth both have an average delay (MAE) of 2 minutes. But if you have a connection with 10 minutes to spare, the second train makes you miss it. **RMSE** captures that: it is 2 minutes for the first train and $\\sqrt{400/10} = 6.3$ minutes for the second. **R²** asks a different question: compared with just assuming the average delay every day, how much of the day-to-day variation does the timetable predict?" },
      { t: "p", text: "MAE treats every unit of error the same. MSE squares errors, so a single error of 10 counts as much as 100 errors of 1 — it is dominated by the worst predictions and by outliers. RMSE takes the square root to return to the target's units, but it is always at least as large as MAE, and the gap between them tells you how uneven the errors are." },
      { t: "p", text: "R² puts the error on a relative scale: 1 means perfect, 0 means \"no better than predicting the mean of the evaluation targets\", and **negative** means worse than that trivial predictor — entirely possible on test data. Residual plots then answer the question the numbers cannot: are the errors random noise, or do they follow a pattern (a missed curve, errors growing with the target) that a better model could exploit?" },
    ],
    mechanism: [
      { t: "steps", items: [
        { title: "Predict on held-out data", text: "Compute $\\hat y_i$ for every validation/test example, in the **original units** (undo any target scaling or log transform first, unless you deliberately evaluate on the transformed scale)." },
        { title: "Compute residuals", text: "$e_i = y_i - \\hat y_i$. Positive residuals mean under-prediction." },
        { title: "Aggregate", text: "MAE = mean of $|e_i|$; MSE = mean of $e_i^2$; RMSE = $\\sqrt{\\text{MSE}}$; R² = $1 - \\sum e_i^2 / \\sum (y_i - \\bar y)^2$; MAPE = mean of $|e_i / y_i|$." },
        { title: "Compare against baselines", text: "Predict-the-training-mean, last-value (persistence) for time series, or a simple linear model. A deep model that does not beat these is not useful." },
        { title: "Analyse residuals", text: "Plot residuals against predictions and against each important feature; bin them and look at the mean (bias) and spread per bin; histogram them; check the largest errors one by one." },
        { title: "Break down by segment", text: "Compute metrics per region, product, time period or range of $y$ — a good global number can hide a failing segment." },
      ] },
      { t: "h", text: "Reading residual patterns" },
      { t: "table", head: ["Pattern in residuals vs prediction/feature", "Meaning", "Typical fix"], rows: [
        ["Random cloud around 0, constant spread", "Model captures the structure; remaining error is noise", "Nothing structural; more data or features to reduce noise"],
        ["U-shape or wave (mean residual changes with x)", "Systematic misfit: missing non-linearity or interaction", "Add features, a more flexible model"],
        ["Funnel (spread grows with x or ŷ)", "Heteroscedasticity: noise scales with the level", "Model log(y), use relative metrics, predict variance (Gaussian NLL)"],
        ["Mean residual ≠ 0 overall", "Bias: systematic over/under-prediction", "Check target transforms, intercept, train/test shift"],
        ["Pattern against time", "Drift, seasonality or autocorrelation", "Time features, retraining, time-series models"],
        ["A few huge residuals", "Outliers, data errors or rare regimes", "Inspect them; robust loss; separate model"],
      ] },
    ],
    math: [
      { t: "h", text: "Definitions" },
      { t: "p", text: "For $n$ evaluation examples with targets $y_i$, predictions $\\hat y_i$, residuals $e_i = y_i - \\hat y_i$ and target mean $\\bar y = \\frac1n\\sum_i y_i$:" },
      { t: "math", tex: String.raw`\text{MAE} = \frac1n\sum_{i=1}^n |e_i|,\qquad \text{MSE} = \frac1n\sum_{i=1}^n e_i^2,\qquad \text{RMSE} = \sqrt{\text{MSE}}` },
      { t: "math", tex: String.raw`R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}},\qquad SS_{\text{res}} = \sum_{i=1}^n e_i^2,\qquad SS_{\text{tot}} = \sum_{i=1}^n (y_i - \bar y)^2` },
      { t: "math", tex: String.raw`\text{MAPE} = \frac1n\sum_{i=1}^n \left|\frac{y_i - \hat y_i}{y_i}\right|` },
      { t: "h", text: "Relations between them" },
      { t: "list", items: [
        "$\\text{MAE} \\le \\text{RMSE} \\le \\sqrt{n}\\,\\text{MAE}$. The left inequality is Jensen's inequality ($\\text{mean}(|e|)^2 \\le \\text{mean}(e^2)$), with equality only when all $|e_i|$ are equal. A large RMSE/MAE ratio signals a few large errors.",
        "Dividing numerator and denominator of R² by $n$: $R^2 = 1 - \\text{MSE}/\\text{Var}(y)$, where $\\text{Var}(y)$ is the (population) variance of the evaluation targets — i.e. the MSE of the constant predictor $\\bar y$. So R² is the fraction of the target variance the model explains.",
        "Bias–variance view: $\\text{MSE} = \\bar e^2 + \\text{Var}(e)$ — squared mean residual (systematic bias) plus residual variance.",
      ] },
      { t: "h", text: "Why R² can be negative" },
      { t: "p", text: "Nothing bounds $SS_{\\text{res}}$ by $SS_{\\text{tot}}$. On the training set, a least-squares linear model with an intercept can always fall back to predicting $\\bar y$, so its training R² lies in $[0, 1]$ and equals the squared correlation between $y$ and $\\hat y$. On **new data** — or for any model not fitted by least squares with an intercept — predictions can be worse than the evaluation-set mean, giving $SS_{\\text{res}} > SS_{\\text{tot}}$ and $R^2 < 0$. Note the baseline is the mean of the *evaluation* targets, which a real model cannot know in advance; a negative test R² therefore means the model is worse than an oracle constant, not necessarily useless — but it is a strong warning." },
      { t: "h", text: "Which metric does which prediction optimize?" },
      { t: "p", text: "For a fixed input, choosing a constant $c$ to minimize $\\mathbb{E}[(Y-c)^2]$ gives $c = \\mathbb{E}[Y]$ (the mean); minimizing $\\mathbb{E}|Y-c|$ gives the median. So a model trained with MSE targets the conditional mean and will score better on RMSE; a model trained with L1 targets the conditional median and will score better on MAE. With skewed noise the two differ — demonstrated in the PyTorch code below (derivations in [Regression Losses](/learn/m05/regression-losses))." },
      { t: "h", text: "MAPE pitfalls" },
      { t: "list", items: [
        "**Undefined at $y_i = 0$** and explosive near zero: an error of 1 on a target of 0.1 is 1000%.",
        "**Asymmetric**: for positive targets, under-prediction is capped at 100% APE (predict 0), over-prediction is unbounded — so minimizing MAPE favours predictions that are biased low.",
        "Alternatives: **sMAPE** (divides by $(|y|+|\\hat y|)/2$; bounded but still unstable near zero), **WAPE** $= \\sum|e_i| / \\sum|y_i|$ (MAE relative to the total; robust and popular in demand forecasting), or **MASE** (MAE relative to a naive forecast's MAE; Hyndman and Koehler, 2006, *Another look at measures of forecast accuracy*).",
      ] },
      { t: "h", text: "Worked example" },
      { t: "p", text: "Targets $y = (3, 5, 2, 7, 4)$, predictions $\\hat y = (2.5, 5.5, 2, 8, 3)$." },
      { t: "steps", items: [
        { title: "Residuals", tex: String.raw`e = y - \hat y = (0.5,\ -0.5,\ 0,\ -1,\ 1)` },
        { title: "MAE", tex: String.raw`\text{MAE} = \frac{0.5 + 0.5 + 0 + 1 + 1}{5} = \frac{3}{5} = 0.6` },
        { title: "MSE and RMSE", tex: String.raw`\text{MSE} = \frac{0.25 + 0.25 + 0 + 1 + 1}{5} = \frac{2.5}{5} = 0.5,\qquad \text{RMSE} = \sqrt{0.5} = 0.7071` },
        { title: "Total sum of squares", text: "$\\bar y = 21/5 = 4.2$.", tex: String.raw`SS_{\text{tot}} = 1.2^2 + 0.8^2 + 2.2^2 + 2.8^2 + 0.2^2 = 1.44 + 0.64 + 4.84 + 7.84 + 0.04 = 14.8` },
        { title: "R²", tex: String.raw`R^2 = 1 - \frac{2.5}{14.8} = 1 - 0.1689 = 0.8311` },
        { title: "MAPE", tex: String.raw`\text{MAPE} = \tfrac15\left(\tfrac{0.5}{3} + \tfrac{0.5}{5} + \tfrac{0}{2} + \tfrac{1}{7} + \tfrac{1}{4}\right) = \tfrac15(0.1667 + 0.1 + 0 + 0.1429 + 0.25) = 0.1319` },
        { title: "A constant prediction of 6", text: "Residuals $(-3, -1, -4, 1, -2)$, $SS_{\\text{res}} = 9 + 1 + 16 + 1 + 4 = 31$.", tex: String.raw`R^2 = 1 - \frac{31}{14.8} = -1.0946 \quad (\text{predicting } \bar y = 4.2 \text{ gives exactly } 0)` },
      ] },
    ],
    visual: [
      { t: "p", text: "The loss explorer shows the per-example penalty each metric assigns to a residual — the shape of the curve is what each metric cares about." },
      { t: "sim", id: "loss-functions", caption: "Use the regression panel: MSE (r²) against MAE (|r|), Huber and log-cosh." },
      { t: "list", items: [
        "Compare MSE and MAE at a residual of 0.5 and at 3: MSE is smaller than MAE below 1 and grows quadratically beyond — this is why RMSE is dominated by large errors and MAE is not.",
        "Look at the gradient curves: MAE's gradient has constant magnitude, MSE's grows with the residual. A model trained with MSE spends its effort on the biggest errors.",
        "Huber is quadratic near zero and linear in the tails: a compromise metric/loss when occasional outliers should not dominate.",
      ] },
      { t: "p", text: "For residual patterns, revisit the polynomial fitting simulator: a degree-1 fit to the curved data leaves systematic, wave-shaped residuals; the right degree leaves noise." },
      { t: "sim", id: "fitting", caption: "Compare the gap between the curve and the points at low degree (systematic residuals) and at the best degree (random residuals)." },
    ],
    scratch: [
      { t: "p", text: "All five metrics from scratch (checked against scikit-learn), residual analysis of a too-simple model on curved, heteroscedastic data, a negative test R² from bad extrapolation, and MAPE's behaviour near zero." },
      { t: "code", lang: "python", title: "regression_metrics_from_scratch.py", code: REG_NUMPY, output: REG_NUMPY_OUT },
      { t: "list", items: [
        "The worked example's numbers (0.6, 0.5, 0.7071, 0.8311, 0.1319) match scikit-learn's implementations exactly.",
        "**Degree 1 has R² = 0.838** — which sounds respectable — but its binned mean residuals go $+0.35, -0.37, -0.72, +0.60$: positive at both ends and negative in the middle, the signature of a straight line through a curve. The degree-2 fit's bin means are smaller and alternate without a systematic shape, and its test RMSE drops from 1.009 to 0.816.",
        "In both models the residual standard deviation grows from about 0.4–0.6 in the first bin to over 1.1 in the last: **heteroscedasticity**. That part is in the data (noise std $0.2 + 0.3x$), not a model error — no mean model can remove it, and RMSE will be dominated by the high-$x$ region.",
        "A degree-5 polynomial fitted only on $x < 3$ and evaluated on $x \\ge 3$ gets **R² = −0.268**: its extrapolation is worse than the constant test mean.",
        "Every prediction is off by exactly 1, so MAE = 1.00, yet MAPE = 222.6% — driven entirely by the target 0.1, whose APE is 1000%.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Two identical linear models are trained on data with **right-skewed** noise (exponential with mean 1 and median $\\ln 2 = 0.693$), one with MSE loss and one with L1 loss, then both are scored on all metrics computed with tensor operations." },
          { t: "code", lang: "python", title: "loss_vs_metric_pytorch.py", code: REG_TORCH, output: REG_TORCH_OUT },
          { t: "list", items: [
            "The MSE-trained line sits 1.001 above the noise-free line (the noise **mean**); the L1-trained line sits 0.700 above it (the noise **median**, 0.693). Each model learned the statistic its loss targets.",
            "Consequently each wins its own metric: the MSE model has lower MSE/RMSE (1.023 vs 1.111) and higher R²; the L1 model has lower MAE (0.695 vs 0.743) and a median residual near 0.",
            "Lesson: **train with the loss that matches the metric you will be judged on**, or at least know which one you are optimizing.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras exposes streaming regression metrics. `R2Score` was added in newer Keras/TensorFlow releases; with older versions, compute R² from MSE and the target variance yourself." },
          { t: "code", lang: "python", title: "regression_metrics_keras.py (not executed in CI)", code: REG_KERAS, output: "Prints a dict with the test loss (MSE), mae, rmse and r2 of the fitted line — values close to the PyTorch MSE model's; the exact numbers depend on the TensorFlow version." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Metric", "Advantages", "Disadvantages", "Use when"], rows: [
        ["MAE", "Target units; robust to outliers; easy to explain", "Not differentiable at 0 (as a loss); ignores error concentration", "All errors cost proportionally; outliers present"],
        ["MSE", "Smooth; matches Gaussian likelihood; decomposes into bias² + variance", "Squared units; dominated by outliers", "Training loss; large errors very costly"],
        ["RMSE", "Target units; penalizes large errors", "Sensitive to outliers; not additive across subsets", "Standard reporting when large errors matter"],
        ["R²", "Unit-free; compares to a mean baseline", "Depends on target variance of the evaluation set; can be negative; not comparable across datasets", "Explaining variance; quick sanity check"],
        ["MAPE", "Percent; scale-free; business-friendly", "Undefined at 0; explodes near 0; biased toward under-prediction", "Strictly positive targets far from 0"],
        ["WAPE / sMAPE", "Scale-free, more stable than MAPE", "WAPE dominated by large items; sMAPE still unstable near 0", "Demand/sales forecasting"],
        ["MASE", "Scale-free; compares to a naive forecast", "Needs a sensible naive baseline", "Time series across many scales"],
        ["RMSLE", "Errors on log scale = relative errors; penalizes under-prediction more", "Requires y > −1; less interpretable", "Targets spanning orders of magnitude (prices, counts)"],
        ["Quantile (pinball) loss / coverage", "Evaluates prediction intervals", "Needs quantile predictions", "Uncertainty-aware forecasting"],
        ["Gaussian NLL", "Evaluates predicted mean and variance", "Needs a variance output", "Probabilistic regression"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Demand and sales forecasting** use WAPE, MASE and quantile losses, because product volumes span orders of magnitude and planning needs intervals, not just point forecasts.",
        "**Weather and energy-load forecasting** report RMSE/MAE together with *skill scores* relative to persistence or climatology — the same idea as R² with a better baseline.",
        "**Real-estate and pricing models** commonly use RMSLE or MAPE-style relative errors, since a 10 000 error matters more on a cheap item than on an expensive one.",
        "**Computer vision regressions** — depth estimation, pose keypoints, bounding-box coordinates — report MAE/RMSE in pixels or metres, and relative error for depth.",
        "**Scientific and medical measurements** analyse residuals against the measured range (Bland–Altman-style plots) to detect level-dependent bias.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**MAE vs RMSE** encodes a cost judgment: linear vs quadratic cost of error size. Reporting both, plus their ratio, describes the error distribution better than either alone.",
        "**R² is relative** to the variance of the evaluation targets: the same model gets a higher R² on a test set with a wider range of $y$. Compare R² only on the same data.",
        "**Scale-free metrics** (MAPE, WAPE, MASE) allow aggregating across series of different magnitudes but each has failure modes near zero or with intermittent demand.",
        "**Aggregate metrics hide structure**; residual analysis and per-segment metrics reveal it, at the cost of manual inspection.",
        "**Point metrics ignore uncertainty**; if downstream decisions need risk estimates, evaluate predictive distributions (pinball loss, coverage, NLL).",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Computing metrics on standardized or log-transformed targets** and reporting them as if in original units. Invert the transform before evaluation (or state clearly which scale is used).",
        "**Reporting training R².** It is optimistic and, for least squares with an intercept, never negative. Report test or CV R².",
        "**Believing R² ∈ [0, 1].** On held-out data it can be any value ≤ 1; a negative value means worse than predicting the evaluation mean.",
        "**Using MAPE with zeros or tiny targets.** It is undefined or dominated by a few points. Use WAPE, MASE or MAE.",
        "**Comparing RMSE across datasets with different scales.** Normalize (by the target's standard deviation, i.e. via R²) or use scale-free metrics.",
        "**Averaging RMSE over folds or groups** as if it were additive. Average MSE, then take the root — or say you are averaging per-fold RMSE.",
        "**Skipping residual plots.** A decent R² can coexist with systematic misfit (as the degree-1 example shows) — exactly the information needed to improve the model.",
        "**Ignoring the baseline.** An RMSE of 3 means nothing until compared with predicting the mean or the last value.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between MAE and RMSE?", a: "MAE averages absolute errors; RMSE is the square root of the average squared error. Both are in the target's units. RMSE penalizes large errors more, so RMSE ≥ MAE, and a large ratio indicates a few big errors or outliers. MAE is more robust and easier to explain." },
      { t: "qa", level: "beginner", q: "What does R² measure?", a: "The fraction of the variance of the target that the model explains: $R^2 = 1 - SS_{res}/SS_{tot}$. 1 is perfect, 0 equals predicting the mean of the targets, and negative values mean worse than that." },
      { t: "qa", level: "beginner", q: "What is a residual and why plot it?", a: "The residual is $y - \\hat y$. Plotting residuals against predictions or features reveals systematic patterns — curvature (missing non-linearity), funnels (variance grows with the level), drift over time, outliers — that aggregate metrics hide." },
      { t: "qa", level: "intermediate", q: "How can R² be negative?", a: "$SS_{res}$ is not bounded by $SS_{tot}$. On test data (or for models not fitted by least squares with an intercept) predictions can be worse than the constant mean of the evaluation targets, so $SS_{res} > SS_{tot}$. Typical causes: overfitting, extrapolation, distribution shift, or a bug such as evaluating on the wrong target scale." },
      { t: "qa", level: "intermediate", q: "What are the problems with MAPE?", a: "It is undefined when any target is 0 and explodes for small targets; it penalizes over-prediction more than under-prediction (APE of an under-prediction of a positive target is at most 100%), biasing optimized forecasts low; and it weights small-volume items heavily. Alternatives: WAPE, MASE, sMAPE (with caveats) or RMSLE." },
      { t: "qa", level: "intermediate", q: "You trained with MSE but your stakeholders judge MAE. What happens and what do you do?", a: "MSE training targets the conditional mean, MAE is minimized by the conditional median; with skewed noise they differ and the model is systematically off for MAE (the PyTorch demo shows 1.001 vs 0.700 offsets). Train with L1 (or Huber with small δ, or a 0.5-quantile loss), or post-adjust the predictions, and select models on validation MAE." },
      { t: "qa", level: "advanced", q: "Show that the training R² of OLS with an intercept equals the squared correlation between y and ŷ.", a: "With an intercept, OLS residuals are orthogonal to the fitted values and sum to 0, so $SS_{tot} = SS_{reg} + SS_{res}$ with $SS_{reg} = \\sum(\\hat y_i - \\bar y)^2$, giving $R^2 = SS_{reg}/SS_{tot}$. Also $\\mathrm{Cov}(y, \\hat y) = \\mathrm{Cov}(\\hat y + e, \\hat y) = \\mathrm{Var}(\\hat y)$, so $\\mathrm{corr}^2 = \\mathrm{Var}(\\hat y)^2/(\\mathrm{Var}(y)\\mathrm{Var}(\\hat y)) = \\mathrm{Var}(\\hat y)/\\mathrm{Var}(y) = R^2$. Neither identity holds on test data or without an intercept." },
      { t: "qa", level: "advanced", q: "Your residuals fan out as the prediction grows. What does that imply for metrics and modelling?", a: "Heteroscedasticity: the noise scale depends on the level. RMSE/MSE will be dominated by the high-level region; MAE less so. Options: model $\\log y$ (turns multiplicative noise additive; evaluate with RMSLE or relative errors), weight the loss, or predict a variance too (Gaussian NLL) and evaluate interval coverage. Report metrics per range of $y$." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Residuals are (2, −1, 0, −3, 2). What is the MAE?", answer: 1.6, tolerance: 0.001, explain: "$(2+1+0+3+2)/5 = 8/5 = 1.6$." },
        { kind: "numeric", q: "Same residuals (2, −1, 0, −3, 2). What is the RMSE? (3 decimals)", answer: 1.897, tolerance: 0.001, explain: "MSE $= (4+1+0+9+4)/5 = 3.6$; RMSE $= \\sqrt{3.6} = 1.897$." },
        { kind: "numeric", q: "A model's test MSE is 4 and the test targets' variance is 16. What is R²?", answer: 0.75, tolerance: 0.001, explain: "$R^2 = 1 - \\text{MSE}/\\text{Var}(y) = 1 - 4/16 = 0.75$." },
        { kind: "numeric", q: "Test MSE is 25 and the test targets' variance is 20. What is R²?", answer: -0.25, tolerance: 0.001, explain: "$1 - 25/20 = -0.25$: worse than predicting the mean." },
        { q: "Which statement about RMSE and MAE is always true?", options: ["RMSE ≤ MAE", "RMSE ≥ MAE", "RMSE = MAE²", "They are uncorrelated"], answer: 1, explain: "By Jensen's inequality, the mean of squares is at least the square of the mean: RMSE ≥ MAE, with equality when all absolute errors are equal." },
        { q: "Target y = 0.5, prediction 1.5. What is this example's absolute percentage error?", options: ["50%", "100%", "200%", "66.7%"], answer: 2, explain: "$|0.5 - 1.5|/0.5 = 2 = 200\\%$ — small targets make APE explode." },
        { q: "Residuals are positive for small and large x and negative in the middle. What is the most likely problem?", options: ["Heteroscedastic noise", "A missing non-linear (e.g. quadratic) effect", "Data leakage", "Too much regularization of the intercept"], answer: 1, explain: "A systematic U-shaped mean residual means the model's functional form misses curvature." },
      ] },
      { t: "exercise", title: "Metrics by hand", prompt: "Targets $y = (10, 12, 8, 15, 5)$, predictions $\\hat y = (11, 11, 9, 12, 7)$. Compute MAE, MSE, RMSE, R² and MAPE.", hint: "$\\bar y = 10$.", solution: [
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
      { t: "exercise", title: "A residual-analysis report", prompt: "Write a function `residual_report(x, y, pred, n_bins=4)` that splits examples into equal-count bins of `x` (use quantiles) and prints, per bin, the count, mean residual, residual standard deviation and RMSE. Also print the overall RMSE/MAE ratio. Use it on the degree-1 and degree-2 fits from the from-scratch program and explain what changes.", hint: "`np.quantile(x, np.linspace(0, 1, n_bins + 1))` gives bin edges; `np.digitize` assigns bins.", solution: [
        { t: "code", lang: "python", code: String.raw`def residual_report(x, y, pred, n_bins=4):
    res = y - pred
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        r = res[which == b]
        print(f"x in [{edges[b]:.2f}, {edges[b + 1]:.2f}]  n={r.size:3d}  mean {r.mean():+.3f}  "
              f"std {r.std():.3f}  rmse {np.sqrt(np.mean(r ** 2)):.3f}")
    print(f"RMSE/MAE = {np.sqrt(np.mean(res ** 2)) / np.mean(np.abs(res)):.3f}")` },
        { t: "p", text: "For the degree-1 fit, the mean residual changes sign across bins in a U-shape (systematic misfit); for degree 2 the means hover near zero. In both, the per-bin std and RMSE grow with $x$ because the noise itself grows — that part cannot be fixed by a better mean model, only modelled (e.g. a variance output) or reported per range." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Residual $e_i = y_i - \\hat y_i$. MAE $= \\frac1n\\sum|e_i|$, MSE $= \\frac1n\\sum e_i^2$, RMSE $= \\sqrt{\\text{MSE}}$; MAE ≤ RMSE.",
        "$R^2 = 1 - SS_{res}/SS_{tot} = 1 - \\text{MSE}/\\text{Var}(y)$: 1 perfect, 0 = predicting the mean, negative = worse than the mean (possible on test data).",
        "MSE is minimized by the mean, MAE by the median — train with the loss matching the reported metric.",
        "MAPE $= \\frac1n\\sum|e_i/y_i|$: undefined at 0, explodes near 0, biased toward under-prediction; prefer WAPE/MASE/RMSLE when appropriate.",
        "Always compare with a baseline and analyse residuals (vs prediction, features, time): patterns = fixable misfit; funnels = heteroscedasticity.",
        "Compute metrics in original units, on held-out data, and per segment.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — use these metrics as the objective of systematic searches.",
        "[Classification Metrics, ROC and PR Curves](/learn/m17/classification-metrics) — the counterpart for discrete labels.",
        "[Monitoring, Drift and MLOps](/learn/m18/mlops) — residuals over time in production are the first signal of drift.",
      ] },
    ],
  },
};

export default lesson;
