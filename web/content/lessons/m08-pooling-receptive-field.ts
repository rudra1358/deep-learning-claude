import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m08/pooling-receptive-field",
  title: "Pooling, Receptive Fields and CNN Architecture",
  summary:
    "Max and average pooling and their gradients (checked against autograd), stride-2 convolutions as learned downsampling, global average and adaptive pooling versus flatten + fully connected heads, how much shift invariance pooling really gives, the receptive-field formula with jumps and dilation, the effective receptive field, and a complete small CNN trained on real digits.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m08/convolution", "m08/images-as-tensors", "m04/backpropagation"],
  next: ["m08/conv-variants", "m08/cnn-architectures", "m07/normalization"],
  terms: ["max-pooling", "average-pooling", "global-average-pooling", "adaptive-pooling", "receptive-field", "effective-receptive-field", "jump", "flatten"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A convolution layer (with same padding) keeps the image size and looks only at a small window. A real image classifier still needs two things: a way to **shrink** feature maps, so later layers are cheap and see more of the image, and a way to turn a stack of maps into one vector of class scores. This lesson covers both, plus the idea that ties them together: the **receptive field** — the patch of the input image one neuron can see.",
      },
      {
        t: "p",
        text: "**Pooling** layers summarise each small window of a feature map with a fixed rule: the largest value (**max pooling**) or the mean (**average pooling**). They have no parameters. A 2×2 window with stride 2 halves height and width (removing 75% of the values) and makes the result less sensitive to small shifts. Modern networks often use **stride-2 convolutions** instead, which shrink the map with learned weights. At the end, **global average pooling** (“Network In Network”, Lin, Chen and Yan, 2013) averages each map to one number. It replaces the huge fully connected layers of older CNNs: 513 thousand parameters instead of 123.6 million for VGG-16-sized features.",
      },
      {
        t: "p",
        text: "The receptive field explains why depth and downsampling matter. Each 3×3 convolution widens the view by only 2 pixels at its own resolution, but after every stride-2 step those 2 pixels stand for twice as many input pixels. We derive the exact formula (VGG-16's last pooling layer sees 212×212 pixels). Then we measure the **effective** receptive field with backpropagation: as Luo, Li, Urtasun and Zemel (2016) showed, influence is packed near the centre and grows much more slowly. We finish with a small CNN, every shape and parameter counted, trained on real data.",
      },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[The Convolution Operation](/learn/m08/convolution) — kernels, stride, padding, the output-size formula ⌊(n + 2p − k)/s⌋ + 1 and parameter counts. Pooling and receptive fields appeared there briefly; here we go deeper.",
          "[Images as Tensors and Why CNNs](/learn/m08/images-as-tensors) — NCHW tensors, channels as feature maps, and the built-in assumptions (locality, weight sharing, equivariance).",
          "[Backpropagation](/learn/m04/backpropagation) — upstream gradients and the chain rule, used here for pooling's backward pass and to measure effective receptive fields.",
          "For the arithmetic: floor/ceil and reading a layer table (shape, parameters).",
        ],
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Managers passing reports up the chain",
        text: "Picture a company where each field worker watches a 3×3 block of streets. A team lead gets reports from 2×2 neighbouring workers and passes up only the most alarming one (max pooling) or the average activity (average pooling). Regional managers do the same with team leads, and so on. Nobody reads every raw report, yet the CEO hears about events anywhere in the city: the CEO's **receptive field** is the whole map. And if an incident moves one street over but stays in the same team's patch, the team lead's report does not change.",
      },
      {
        t: "p",
        text: "Max pooling asks “is the feature anywhere in this little window?” and forgets exactly where. Average pooling asks “how much of the feature is in this window?”. Either way the map gets smaller, so the next 3×3 convolution covers a region twice as wide in the original image. That is the main job of downsampling: grow the receptive field fast and cut computation.",
      },
      {
        t: "p",
        text: "At the end, a classifier must answer “is there a cat?”, not “is there a cat at position (3, 5)?”. **Global average pooling** averages each feature map over all positions, so a “cat-ear detector” map gives the same evidence wherever the ear is. The older option, **flattening** the maps into one long vector for a fully connected layer, keeps one weight per position and needs far more parameters. Finally, not every pixel a neuron can see matters equally: the **effective receptive field** is a bright centre fading toward the edges.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Why shrink the map at all?" },
      {
        t: "list",
        items: [
          "**Compute.** A convolution costs C_out·C_in·k²·H_out·W_out multiply-adds. Halving H and W cuts that by 4. This is why networks can double their channels after each downsampling and keep the cost per stage roughly constant.",
          "**Receptive field.** Each downsampling by 2 doubles the jump — the number of input pixels between neighbouring units — so every later 3×3 convolution adds twice as much view. Without downsampling, ten 3×3 layers see only 21×21 pixels.",
          "**Robustness to small shifts.** Summarising a window throws away exact positions inside it, so small shifts and distortions matter less (measured below — the effect is smaller than often claimed).",
        ],
      },
      { t: "h", text: "Max pooling" },
      {
        t: "steps",
        items: [
          { title: "Pick window size k and stride s", text: "Most common: k = 2, s = 2 (2×2 blocks that don't overlap). AlexNet used overlapping 3×3 windows with stride 2; ResNet's stem uses 3×3, stride 2, padding 1." },
          { title: "Work on each channel separately", text: "Pooling never mixes channels: a C×H×W input gives a C×H_out×W_out output with the same C." },
          { title: "Keep the maximum of each window", text: "Each output is the largest of the k² values. The layer remembers where that maximum was (the argmax) for the backward pass." },
          { title: "Output size", text: "Same formula as convolution: ⌊(n + 2p − k)/s⌋ + 1. For k = s = 2, p = 0 this is ⌊n/2⌋, so an odd last row or column is dropped." },
        ],
      },
      {
        t: "p",
        text: "Max pooling suits ReLU maps, where a value means “this pattern was found here”: it keeps the strongest detection in each neighbourhood. With padding, PyTorch pads max pooling with −∞ (so padding never wins), not zeros.",
      },
      { t: "h", text: "Average pooling" },
      {
        t: "p",
        text: "Average pooling outputs the mean of each window. It is smoother — every input counts — and works like a blur followed by subsampling. This reduces **aliasing** (fine detail turning into false coarse patterns when pixels are skipped) compared with max pooling. LeNet-5 used a trainable version (scaled average plus bias). Today it shows up mostly as global average pooling at the end of networks, and as 2×2 average pooling in some shortcut paths and transition layers (e.g. DenseNet's).",
      },
      { t: "h", text: "Gradients through pooling" },
      {
        t: "list",
        items: [
          "**Max pooling routes** the gradient: each output's upstream gradient goes entirely to the one input that was the maximum; the others in the window get 0. Only the “winners” learn from that output. If windows overlap (k > s), an input that wins several windows gets the **sum** of their gradients.",
          "**Average pooling spreads** the gradient: each of the k² inputs in a window gets 1/k² of the output's upstream gradient.",
          "**Ties** in max pooling are broken by a convention (PyTorch picks the first maximum in the window). The max has no derivative at a tie, and this choice is a valid subgradient.",
          "Neither layer has parameters, so there is no weight gradient. The math section works both rules out with numbers; the code checks them against PyTorch autograd.",
        ],
      },
      { t: "h", text: "Stride-2 convolution: learned downsampling" },
      {
        t: "p",
        text: "A stride-2 convolution computes only every other output position, so it shrinks the map and transforms features in one step, with learned weights. Springenberg, Dosovitskiy, Brox and Riedmiller (“Striving for Simplicity: The All Convolutional Net”, 2014) showed that swapping max pooling for strided convolutions costs little or no accuracy; ResNet, MobileNet and most later models downsample this way. It adds parameters and compute, but learns what to keep instead of hard-coding “max”. Like any subsampling without blurring first, it aliases. Zhang (“Making Convolutional Networks Shift-Invariant Again”, 2019) proposed blurring before subsampling to win back some shift stability.",
      },
      { t: "h", text: "Flattening and the fully connected head" },
      {
        t: "p",
        text: "After the last convolutional stage, the C×H×W tensor must become class scores. The classic way (LeNet, AlexNet, VGG) is to **flatten** it into a vector of C·H·W numbers — a reshape, no parameters — and apply fully connected layers. Every (channel, position) pair gets its own weight, so the head can use *where* a feature appeared. But it is expensive: in VGG-16, flattening 512×7×7 = 25,088 values into a 4,096-unit layer costs 102.8 million weights, and the three FC layers hold 123.6 million of the network's 138.4 million parameters (89%). It also locks the input size, because the flattened length must match the weight matrix.",
      },
      { t: "h", text: "Global average pooling (GAP)" },
      {
        t: "p",
        text: "GAP averages each of the C feature maps over all H×W positions, giving exactly C numbers whatever the spatial size; one linear layer C → classes follows. Network In Network (2013) proposed it to replace fully connected layers. It has no parameters, acts as a built-in regulariser (no position-specific weights to overfit), accepts any input size and makes the prediction translation invariant. For VGG-16-sized features, GAP + one linear layer has 513,000 parameters versus 123,642,856 for the flatten + 3 FC head — 241× fewer. ResNet-50's head is GAP + one 2,048 → 1,000 linear layer (2,049,000 parameters). Bonus: because the classifier is linear on per-map averages, its weights can be projected back onto the maps to show where each class's evidence lies — Class Activation Mapping (CAM), Zhou et al. (2016).",
      },
      { t: "h", text: "Adaptive pooling" },
      {
        t: "p",
        text: "**Adaptive pooling** fixes the *output* size instead of the window. `nn.AdaptiveAvgPool2d((7, 7))` makes 7×7 maps from any input by choosing the window edges for you (bin i covers rows ⌊i·n/m⌋ to ⌈(i+1)·n/m⌉ − 1, so bins may overlap by one). GAP is the case output size 1×1. torchvision's VGG uses adaptive 7×7 pooling before its flatten, so non-224×224 inputs still give the 25,088-long vector its FC layers expect. Spatial pyramid pooling (He et al., 2014) joins adaptive poolings at several output sizes (e.g. 1×1, 2×2, 4×4).",
      },
      { t: "h", text: "Invariance vs equivariance: what pooling really gives" },
      {
        t: "p",
        text: "Convolution is translation **equivariant**: shift the input and the feature map shifts by the same amount. A classifier needs **invariance**: the same answer wherever the object is. Pooling is often said to give invariance, but local pooling only partly tolerates *small* shifts. A 2×2 max pool gives the same output if a feature moves within its 2×2 block; if it crosses a block edge, the pooled map moves by one coarse cell. Our NumPy experiment runs four random 3×3 filters with ReLU on a small “T” shape and measures how much the result changes when the input shifts by 1–4 pixels:",
      },
      {
        t: "table",
        head: ["Shift (px)", "Conv map: equivariance error", "Conv map: relative change", "After one 2×2 max-pool", "After two 2×2 max-pools", "After global average pooling"],
        rows: [
          ["1", "0", "0.859", "0.761", "0.476", "≈ 0 (2.5e-17)"],
          ["2", "0", "1.182", "1.037", "0.674", "0"],
          ["3", "0", "1.255", "1.256", "0.935", "≈ 0 (2.5e-17)"],
          ["4", "0", "1.322", "1.275", "1.096", "0"],
        ],
      },
      {
        t: "list",
        items: [
          "The convolution is exactly equivariant (error 0: the shifted map equals the map of the shifted input). Yet as a representation it changes completely — a relative change near 1 means old and new maps barely overlap.",
          "One 2×2 max pool reduces the change only a little (0.86 → 0.76 for a 1-pixel shift). Two pooling layers do better, but the benefit shrinks as the shift grows. So pooling gives **tolerance to small shifts that grows with depth**, not invariance.",
          "Global average pooling is exactly invariant here (up to float round-off). The shifted shape stays inside the image, so the maps only move and their averages stay the same. This — not local pooling — is where most of a modern CNN's translation invariance comes from, together with data augmentation.",
        ],
      },
      { t: "h", text: "Receptive field" },
      {
        t: "p",
        text: "The **receptive field** (RF) of a unit is the set of input pixels that can affect its value. It depends on each layer's kernel size k, stride s and dilation d. (A dilated kernel spreads its k taps d pixels apart, so it spans an **effective kernel size** k_eff = d(k − 1) + 1.) Track two numbers per layer: r, the RF width in input pixels, and the **jump** j, the distance in input pixels between neighbouring units of that layer (the product of all strides so far). A new layer's window covers k_eff units of the previous layer — (k_eff − 1) gaps of j_{l−1} input pixels — on top of the previous field:",
      },
      {
        t: "list",
        items: [
          "r_l = r_{l−1} + (k_eff,l − 1) · j_{l−1}, with r_0 = 1",
          "j_l = j_{l−1} · s_l, with j_0 = 1",
          "Pooling layers count exactly like convolutions (k, s). ReLU, BatchNorm and other element-wise layers don't change r or j; a 1×1 convolution adds nothing.",
          "Padding doesn't change the RF size, only where it sits. Near the borders part of the field falls on padding, so border units see fewer real pixels.",
        ],
      },
      {
        t: "p",
        text: "Two consequences. (1) Growth per layer is proportional to the current jump, so layers *after* downsampling grow the RF fastest. (2) Order matters for cost too: an early stride-2 layer makes every later layer cheaper and their RF growth larger. **Dilation** raises k_eff without extra parameters: four 3×3 convolutions with dilations 1, 2, 4, 8 at stride 1 reach a 31-pixel RF with no downsampling (versus 9 without dilation). Dilated networks for segmentation and audio build on this; see [Dilated, Transposed, Depthwise and Pointwise Convolutions](/learn/m08/conv-variants). Repeating the same dilation (2, 2, 2) leaves holes: the unit sees only every other pixel (“gridding”), as the simulator shows.",
      },
      { t: "h", text: "Effective receptive field" },
      {
        t: "p",
        text: "The theoretical RF says which pixels *can* matter, not how much. To measure influence, take the gradient size |∂y/∂x_i| of a centre output unit y for each input pixel x_i. In a stack of 3×3 convolutions the centre pixel has the most paths to the output, and the count falls toward the edges. (In 1-D: the number of ways to take n steps of −1, 0 or +1 and end at m peaks at m = 0.) With random weights the contributions add up roughly like independent steps, so by the central-limit theorem the profile is close to a Gaussian (bell curve) with standard deviation growing like √n, while the theoretical RF grows like n. Luo et al. (2016) showed that training widens the effective RF somewhat, but it stays a fraction of the theoretical one. Our measurement on random He-initialised conv–ReLU stacks (PyTorch tab):",
      },
      {
        t: "table",
        head: ["Depth (3×3 convs)", "Theoretical RF", "Width holding 50% of |grad|", "Width holding 90%", "Std of profile (px)"],
        rows: [
          ["2", "5", "5", "5", "1.30"],
          ["4", "9", "5", "9", "1.96"],
          ["8", "17", "7", "13", "3.05"],
          ["16", "33", "11", "19", "4.50"],
        ],
      },
      {
        t: "p",
        text: "Doubling the depth from 8 to 16 doubles the theoretical RF (17 → 33) but widens the spread of influence by only 1.48 ≈ √2. In practice: if a task needs context over a region of size R, make the theoretical RF comfortably larger than R, or use downsampling, dilation or attention.",
      },
      { t: "h", text: "Putting it together: a complete small CNN" },
      {
        t: "p",
        text: "A VGG-style network for scikit-learn's 8×8 digits: two stages, each with two same-padded 3×3 convolutions with ReLU and then 2×2 max pooling, followed by global average pooling and one linear layer. The PyTorch program in the framework section printed these shapes and parameter counts.",
      },
      {
        t: "table",
        head: ["Layer", "Output (C×H×W)", "Parameters", "RF r / jump j"],
        rows: [
          ["Input", "1×8×8", "0", "1 / 1"],
          ["Conv 3×3, 1 → 16, p = 1 + ReLU", "16×8×8", "16·1·9 + 16 = 160", "3 / 1"],
          ["Conv 3×3, 16 → 16, p = 1 + ReLU", "16×8×8", "16·16·9 + 16 = 2,320", "5 / 1"],
          ["Max-pool 2×2, s = 2", "16×4×4", "0", "6 / 2"],
          ["Conv 3×3, 16 → 32, p = 1 + ReLU", "32×4×4", "32·16·9 + 32 = 4,640", "10 / 2"],
          ["Conv 3×3, 32 → 32, p = 1 + ReLU", "32×4×4", "32·32·9 + 32 = 9,248", "14 / 2"],
          ["Max-pool 2×2, s = 2", "32×2×2", "0", "16 / 4"],
          ["Global average pool", "32×1×1 → 32", "0", "whole image"],
          ["Linear 32 → 10", "10", "32·10 + 10 = 330", ""],
          ["**Total**", "", "**16,698**", ""],
        ],
      },
      {
        t: "p",
        text: "After the second pool each unit has a 16×16 receptive field, bigger than the 8×8 image (the extra falls on padding), so every unit sees the whole digit before GAP. After 15 epochs of Adam it reaches 97.6% test accuracy. The same body with a flatten + FC(128 → 64 → 10) head has 25,274 parameters and reaches 98.0%; replacing both max-pools with stride-2 convolutions (28,266 parameters) gives 95.1% on this tiny dataset. On 8×8 images the gaps are small and partly noise. The point: every design choice shows up as concrete numbers in shapes, parameters and receptive fields.",
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "Pooling as formulas" },
      {
        t: "p",
        text: String.raw`For one channel $X$ of size $H \times W$, window $k$ and stride $s$, output $(i, j)$ looks at the window $\mathcal{W}_{ij} = \{(s\,i + m,\; s\,j + n) : 0 \le m, n < k\}$. Then:`,
      },
      {
        t: "math",
        tex: String.raw`Y^{\max}_{ij} = \max_{(u,v) \in \mathcal{W}_{ij}} X_{uv}, \qquad Y^{\text{avg}}_{ij} = \frac{1}{k^2}\sum_{(u,v)\in\mathcal{W}_{ij}} X_{uv}, \qquad H_{\text{out}} = \left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1`,
      },
      { t: "h", text: "Backward pass" },
      {
        t: "p",
        text: String.raw`Let $G_{ij} = \partial L/\partial Y_{ij}$ be the upstream gradient and $(u^*_{ij}, v^*_{ij})$ the argmax of window $\mathcal{W}_{ij}$. The chain rule sums over every window that contains $(u, v)$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial L}{\partial X_{uv}}\bigg|_{\max} = \sum_{i,j} G_{ij}\;\mathbf{1}\big[(u,v) = (u^*_{ij}, v^*_{ij})\big], \qquad \frac{\partial L}{\partial X_{uv}}\bigg|_{\text{avg}} = \sum_{i,j:\,(u,v)\in\mathcal{W}_{ij}} \frac{G_{ij}}{k^2}`,
        caption: "Max sends each upstream gradient to its argmax; average spreads it evenly. With non-overlapping windows each input sits in exactly one window, so each sum has one term.",
      },
      {
        t: "p",
        text: "**Worked example** (2×2 windows, stride 2). The input and the upstream gradient:",
      },
      {
        t: "math",
        tex: String.raw`X = \begin{bmatrix}1&3&2&0\\4&2&1&5\\0&1&3&2\\2&6&1&1\end{bmatrix}, \qquad G = \frac{\partial L}{\partial Y} = \begin{bmatrix}0.5&-1\\2&0.25\end{bmatrix}`,
      },
      {
        t: "steps",
        items: [
          { title: "Max pooling forward", text: "The blocks are {1, 3, 4, 2}, {2, 0, 1, 5}, {0, 1, 2, 6}, {3, 2, 1, 1}. Their maxima: 4 at (1,0), 5 at (1,3), 6 at (3,1), 3 at (2,2).", tex: String.raw`Y^{\max} = \begin{bmatrix}4&5\\6&3\end{bmatrix}` },
          { title: "Max pooling backward", text: "Each block's gradient goes to its argmax; the other 12 entries get 0.", tex: String.raw`\frac{\partial L}{\partial X} = \begin{bmatrix}0&0&0&0\\0.5&0&0&-1\\0&0&0.25&0\\0&2&0&0\end{bmatrix}` },
          { title: "Average pooling forward", tex: String.raw`Y^{\text{avg}} = \begin{bmatrix}10/4&8/4\\9/4&7/4\end{bmatrix} = \begin{bmatrix}2.5&2\\2.25&1.75\end{bmatrix}` },
          { title: "Average pooling backward", text: "Divide each block's gradient by k² = 4 and copy it to all four entries.", tex: String.raw`\frac{\partial L}{\partial X} = \begin{bmatrix}0.125&0.125&-0.25&-0.25\\0.125&0.125&-0.25&-0.25\\0.5&0.5&0.0625&0.0625\\0.5&0.5&0.0625&0.0625\end{bmatrix}` },
        ],
      },
      {
        t: "p",
        text: String.raw`Both results match PyTorch autograd exactly (checked in the NumPy code). In both cases the total gradient is kept: the entries of each block of $\partial L/\partial X$ add up to that block's $G_{ij}$. **Overlapping windows:** take $k = 3, s = 2$ on a 5×5 input whose only non-zero entry is a 9 in the centre. All four windows contain the centre, so all four outputs are 9; with $G = \mathbf{1}$ the centre receives $1 + 1 + 1 + 1 = 4$.`,
      },
      { t: "h", text: "Global average and adaptive pooling" },
      {
        t: "math",
        tex: String.raw`\text{GAP}(X)_c = \frac{1}{HW}\sum_{u=0}^{H-1}\sum_{v=0}^{W-1} X_{c,uv}, \qquad \text{adaptive bin } i \text{ of } m:\ \Big[\big\lfloor \tfrac{i\,n}{m}\big\rfloor,\ \big\lceil \tfrac{(i+1)\,n}{m}\big\rceil\Big)`,
      },
      {
        t: "p",
        text: String.raw`Example: $n = 7 \to m = 3$ bins $[0,3), [2,5), [4,7)$ (overlapping by one). For $x = (3,1,4,1,5,9,2)$ the adaptive averages are $8/3 = 2.6667$, $10/3 = 3.3333$, $16/3 = 5.3333$ and the adaptive maxima are $4, 5, 9$. GAP is an average over $HW$ positions, so its gradient is $\partial L/\partial X_{c,uv} = G_c / (HW)$.`,
      },
      { t: "h", text: "What the classifier head costs in parameters" },
      {
        t: "math",
        tex: String.raw`\#\theta_{\text{flatten+FC}} = (C\,H\,W)\,m_1 + m_1 + \dots, \qquad \#\theta_{\text{GAP+FC}} = C \cdot K + K`,
        caption: "K classes, m_1 hidden units. The flatten head grows with the final spatial size H·W; the GAP head does not.",
      },
      {
        t: "p",
        text: String.raw`For VGG-16 ($C = 512$, $H = W = 7$, $K = 1000$): $25{,}088\cdot 4{,}096 + 4{,}096 = 102{,}764{,}544$, plus $4{,}096\cdot4{,}096 + 4{,}096 = 16{,}781{,}312$, plus $4{,}096\cdot 1{,}000 + 1{,}000 = 4{,}097{,}000$, total $123{,}642{,}856$. The GAP head: $512 \cdot 1000 + 1000 = 513{,}000$.`,
      },
      { t: "h", text: "Shift: equivariance and (partial) invariance" },
      {
        t: "p",
        text: String.raw`Let $T_\delta$ shift by $\delta$ pixels. Convolution commutes with shifts: $K \star T_\delta X = T_\delta(K\star X)$. For 2×2/stride-2 pooling $P$: if $\delta$ is a multiple of 2, $P(T_\delta Z) = T_{\delta/2} P(Z)$ — equivariance at the coarser resolution. For odd $\delta$ there is no such rule in general. A 1-D example with one spike: $P(0,9,0,0,0,0) = (9,0,0)$ and $P(9,0,0,0,0,0) = (9,0,0)$, so a shift inside the block changes nothing. But $P(0,0,9,0,0,0) = (0,9,0)$: a 1-pixel shift across a block edge moves the output. Global averaging ignores any shift that keeps the content inside the image: $\text{GAP}(T_\delta Z) = \text{GAP}(Z)$.`,
      },
      { t: "h", text: "Receptive-field formula and closed form" },
      {
        t: "p",
        text: String.raw`Let layer $\ell$ have kernel $k_\ell$, stride $s_\ell$, dilation $d_\ell$ and effective kernel $\hat k_\ell = d_\ell(k_\ell - 1) + 1$. Unit $i$ of layer $\ell$ reads units $s_\ell i - p_\ell + a\,d_\ell$, $a = 0,\dots,k_\ell - 1$, of layer $\ell - 1$: a span of $\hat k_\ell$ neighbouring positions. Neighbouring units of layer $\ell-1$ are $j_{\ell-1}$ input pixels apart and each sees $r_{\ell-1}$ pixels, so together their fields cover`,
      },
      {
        t: "math",
        tex: String.raw`r_\ell = r_{\ell-1} + (\hat k_\ell - 1)\,j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\,s_\ell, \qquad r_0 = j_0 = 1 \quad\Longrightarrow\quad r_L = 1 + \sum_{\ell=1}^{L} (\hat k_\ell - 1)\prod_{i=1}^{\ell-1} s_i`,
      },
      {
        t: "p",
        text: String.raw`To find the actual input interval, project index ranges backwards: if layer $\ell$'s relevant units are $[a, b]$, layer $\ell-1$'s are $[a\,s_\ell - p_\ell,\; b\,s_\ell - p_\ell + d_\ell(k_\ell-1)]$ (indices outside $[0, n_{\ell-1}-1]$ are padding). Unit 0 of layer $\ell$ is centred at input coordinate $c_\ell = c_{\ell-1} + \big(\tfrac{\hat k_\ell - 1}{2} - p_\ell\big) j_{\ell-1}$, $c_0 = 0$, and unit $i$ at $c_\ell + i\,j_\ell$.`,
      },
      {
        t: "table",
        head: ["VGG-16 layer", "k, s", "r", "j", "", "VGG-16 layer", "k, s", "r", "j"],
        rows: [
          ["conv1_1", "3, 1", "3", "1", "", "conv3_3", "3, 1", "40", "4"],
          ["conv1_2", "3, 1", "5", "1", "", "pool3", "2, 2", "44", "8"],
          ["pool1", "2, 2", "6", "2", "", "conv4_1", "3, 1", "60", "8"],
          ["conv2_1", "3, 1", "10", "2", "", "conv4_2", "3, 1", "76", "8"],
          ["conv2_2", "3, 1", "14", "2", "", "conv4_3", "3, 1", "92", "8"],
          ["pool2", "2, 2", "16", "4", "", "pool4", "2, 2", "100", "16"],
          ["conv3_1", "3, 1", "24", "4", "", "conv5_1 … conv5_3", "3, 1", "132, 164, 196", "16"],
          ["conv3_2", "3, 1", "32", "4", "", "pool5", "2, 2", "212", "32"],
        ],
      },
      {
        t: "p",
        text: String.raw`See how the step per layer, $2\,j_{\ell-1}$, grows from 2 (stage 1) to 32 (stage 5). The six convolutions of stages 4–5 add $3\cdot 16 + 3\cdot 32 = 144$ of the 212 pixels, though they are only 6 of the 13 convolutions. With dilations $1, 2, 4, 8$ on four stride-1 3×3 layers, $\hat k = 3, 5, 9, 17$ and $r = 3, 7, 15, 31$.`,
      },
      { t: "h", text: "Effective receptive field" },
      {
        t: "p",
        text: String.raw`Define the influence map of output unit $y$ as $g_i = |\partial y / \partial x_i|$ (averaged over random weights and inputs). Take $n$ stacked linear 1-D convolutions with all weights equal to $1/k$. Then $\partial y/\partial x$ is the box $(1/k,\dots,1/k)$ convolved with itself $n$ times. That is the distribution of a sum of $n$ independent uniform steps, with variance $n\,\sigma_1^2$, where $\sigma_1^2 = (k^2 - 1)/12$ is the variance of one step. For $k = 3$, $\sigma_1^2 = 2/3$, so the standard deviation is $\sqrt{2n/3}$ while the full width is $2n + 1$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\text{effective width}}{\text{theoretical width}} \;\propto\; \frac{\sqrt{n}}{n} = \frac{1}{\sqrt{n}}`,
        caption: "As depth grows, the effective receptive field shrinks compared with the theoretical one.",
      },
      {
        t: "p",
        text: String.raw`Our ReLU networks with random He-initialised weights give standard deviations $1.30, 1.96, 3.05, 4.50$ px for $n = 2, 4, 8, 16$. Each doubling multiplies it by about $\sqrt 2 \approx 1.41$ (1.51, 1.56, 1.48), matching the $\sqrt n$ law, while the theoretical width doubles.`,
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "receptive-field", caption: "Build a layer stack, pick an output unit and trace it back layer by layer; the table tracks output size, receptive field r and jump j." },
      {
        t: "list",
        items: [
          "Start with the **VGG-style** preset (input 48×48). r grows 3, 5, 6, 10, 14, 16, 24, 32, 40, 44 and j doubles at every pool — the first ten rows of the VGG-16 table in the math section. Press **Step** repeatedly: the gold interval widens one layer at a time, and the narration shows the back-projection formula [a·s − p, b·s − p + d(k − 1)] with real numbers.",
          "Watch the *spacing* of units in each row: after each pool they sit twice as far apart. That spacing is the jump — why 3×3 convolutions after the last pool widen the region by 8 pixels each, the first ones by 2.",
          "Move the **Output unit** slider to 0. The interval now starts at a negative index: part of the field lies in the zero padding, so border units see fewer real pixels than r.",
          "Choose **3×3 convs only** and add **+ conv 3×3** layers: r grows by exactly 2 each. Add **+ pool 2×2** (r grows by 1, the jump doubles), then another **+ conv 3×3** — it adds 4. Order matters.",
          "Choose **Dilated (1, 2, 4, 8)**: r = 3, 7, 15, 31 with stride 1 and no shrinking. Then choose **Dilated (2, 2, 2) — gridding** and step to the input: only every other pixel lights up, and the narration counts the holes. This is the gridding artefact of naive dilation.",
          "Tick **Effective RF heatmap**. The input grid is shaded by |∂output/∂input| of a random linear version of the stack (pooling replaced by averaging), averaged over 24 weight draws — real backpropagation in your browser. The side panel reports the central square holding 50% and 90% of the gradient mass: for the VGG-style preset, half the influence comes from a square about half as wide as the 44-pixel theoretical field.",
        ],
      },
      { t: "sim", id: "cnn-pipeline", caption: "Convolution → ReLU → max-pool, twice, on a real image: watch the maps shrink and get coarser at each pooling stage." },
      {
        t: "list",
        items: [
          "Step to the first max-pool stage. The 6 feature maps go from 62×62 to 31×31 — three quarters of the values gone — yet the edges are still clearly visible, because max pooling keeps the strongest responses.",
          "Compare the maps after the second pool (15×15) with the first convolution's maps. Each pixel now summarises a much bigger area (receptive field 3 + 1 + 2·2 + 1·2 = 10 pixels for 3×3 kernels and 2×2 pools), so the maps look blockier and more abstract.",
          "Set **Layer-1 stride** to 2: now the first convolution itself halves the resolution — downsampling by strided convolution instead of pooling.",
        ],
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "Max and average pooling with backward passes (checked against PyTorch autograd), gradient summing for overlapping windows, global average pooling, a receptive-field calculator with dilation (it reproduces the VGG-16 table), exact back-projection of one output unit, and the equivariance-vs-invariance experiment.",
      },
      {
        t: "code",
        lang: "python",
        title: "pooling_and_receptive_field.py",
        code: String.raw`import numpy as np
import torch                      # used only to check our results
import torch.nn.functional as F


# ---------------------------------------------------------------- pooling layers
def pool_forward(x, k=2, s=2, mode="max"):
    """x: (C, H, W). Returns pooled output and a cache for the backward pass."""
    C, H, W = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    out = np.zeros((C, Ho, Wo))
    argmax = np.zeros((C, Ho, Wo, 2), dtype=int)       # where each max came from
    for i in range(Ho):
        for j in range(Wo):
            win = x[:, i * s:i * s + k, j * s:j * s + k].reshape(C, -1)
            if mode == "max":
                a = win.argmax(axis=1)                  # first max wins on ties (like PyTorch)
                out[:, i, j] = win[np.arange(C), a]
                argmax[:, i, j] = np.stack([i * s + a // k, j * s + a % k], axis=1)
            else:
                out[:, i, j] = win.mean(axis=1)
    return out, (x.shape, k, s, mode, argmax)


def pool_backward(g, cache):
    """g = dL/dout, shape (C, Ho, Wo). Returns dL/dx."""
    shape, k, s, mode, argmax = cache
    dx = np.zeros(shape)
    C, Ho, Wo = g.shape
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                if mode == "max":                       # route the whole gradient to the argmax
                    r, q = argmax[c, i, j]
                    dx[c, r, q] += g[c, i, j]           # += : overlapping windows accumulate
                else:                                   # spread it evenly over the window
                    dx[c, i * s:i * s + k, j * s:j * s + k] += g[c, i, j] / (k * k)
    return dx


X = np.array([[1, 3, 2, 0],
              [4, 2, 1, 5],
              [0, 1, 3, 2],
              [2, 6, 1, 1]], dtype=float)[None]          # (1, 4, 4)
G = np.array([[0.5, -1.0],
              [2.0, 0.25]])[None]                        # upstream gradient dL/dY
for mode in ["max", "avg"]:
    y, cache = pool_forward(X, mode=mode)
    dx = pool_backward(G, cache)
    xt = torch.tensor(X[None], requires_grad=True)
    yt = F.max_pool2d(xt, 2) if mode == "max" else F.avg_pool2d(xt, 2)
    (yt * torch.tensor(G[None])).sum().backward()
    print(f"{mode}-pool output:\n{y[0]}\ndL/dX:\n{dx[0]}")
    print("matches torch autograd:", np.allclose(dx, xt.grad.numpy()[0]))

# overlapping windows (k=3, s=2): one pixel can be the max of several windows
Z = np.zeros((1, 5, 5)); Z[0, 2, 2] = 9.0
y, cache = pool_forward(Z, k=3, s=2)
print("3x3/s2 max-pool of a centre spike:", y[0].tolist(),
      "-> gradient at the spike:", pool_backward(np.ones_like(y), cache)[0, 2, 2])

# ---------------------------------------------------------------- global average pooling
fmap = np.random.default_rng(0).standard_normal((4, 5, 5))  # C=4 feature maps
gap = fmap.mean(axis=(1, 2))                                 # (C,) - one number per map
print("GAP output shape:", gap.shape, np.round(gap, 4))


# ---------------------------------------------------------------- receptive field
def receptive_field(layers):
    """layers: list of (name, k, s, d). Returns rows of (name, r, j) using
    r_l = r_{l-1} + (k_eff - 1) * j_{l-1},  j_l = j_{l-1} * s,  k_eff = d*(k-1) + 1."""
    r, j, rows = 1, 1, []
    for name, k, s, d in layers:
        k_eff = d * (k - 1) + 1
        r, j = r + (k_eff - 1) * j, j * s
        rows.append((name, r, j))
    return rows


conv = lambda n: (n, 3, 1, 1)
pool = lambda n: (n, 2, 2, 1)
vgg16 = [conv("conv1_1"), conv("conv1_2"), pool("pool1"),
         conv("conv2_1"), conv("conv2_2"), pool("pool2"),
         conv("conv3_1"), conv("conv3_2"), conv("conv3_3"), pool("pool3"),
         conv("conv4_1"), conv("conv4_2"), conv("conv4_3"), pool("pool4"),
         conv("conv5_1"), conv("conv5_2"), conv("conv5_3"), pool("pool5")]
print("VGG-16 receptive fields (r = width in input pixels, j = jump):")
print("  ".join(f"{n}:{r}/{j}" for n, r, j in receptive_field(vgg16)))
print("four 3x3 convs, dilation 1,2,4,8:",
      [r for _, r, _ in receptive_field([("d", 3, 1, d) for d in (1, 2, 4, 8)])])


def back_project(layers, sizes, unit):
    """Exact input interval seen by output unit number "unit" of the last layer.
    layers: (k, s, p, d); sizes: spatial size of the input of each layer. Walks backwards:
    unit i of a layer covers inputs i*s - p ... i*s - p + d*(k-1) (clipped to the real input)."""
    lo, hi = unit, unit
    for (k, s, p, d), n in zip(reversed(layers), reversed(sizes)):
        lo, hi = max(lo * s - p, 0), min(hi * s - p + d * (k - 1), n - 1)
    return lo, hi


layers = [(3, 1, 1, 1), (3, 1, 1, 1), (2, 2, 0, 1), (3, 1, 1, 1)]   # conv, conv, pool, conv
sizes = [16, 16, 16, 8]
print("unit 3 of the last layer sees input pixels", back_project(layers, sizes, 3),
      "| unit 0 (border, clipped):", back_project(layers, sizes, 0))

# ---------------------------------------------------------------- equivariance vs invariance
rng = np.random.default_rng(1)
img = np.zeros((24, 24)); img[8:14, 9:11] = 1.0; img[8, 7:13] = 1.0     # a small "T" shape
kern = rng.standard_normal((4, 3, 3))                                  # 4 random 3x3 filters


def conv_relu(a):
    ap = np.pad(a, 1)
    out = np.zeros((4, 24, 24))
    for m in range(3):
        for n in range(3):
            out += kern[:, m, n, None, None] * ap[None, m:m + 24, n:n + 24]
    return np.maximum(out, 0)


f0 = conv_relu(img)
print("shift | conv map: equivariance err, change | 2x2 max-pool change | two pools | GAP change")
for dx in [1, 2, 3, 4]:
    f1 = conv_relu(np.roll(img, dx, axis=1))
    eq_err = np.abs(f1 - np.roll(f0, dx, axis=2)).max()
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)
    p1 = lambda f: pool_forward(f)[0]
    print(f"  {dx}   |   {eq_err:.1e}   {rel(f0, f1):.3f}     |   {rel(p1(f0), p1(f1)):.3f}  "
          f"         |  {rel(p1(p1(f0)), p1(p1(f1))):.3f}   | {rel(f0.mean((1, 2)), f1.mean((1, 2))):.1e}")`,
        output: String.raw`max-pool output:
[[4. 5.]
 [6. 3.]]
dL/dX:
[[ 0.    0.    0.    0.  ]
 [ 0.5   0.    0.   -1.  ]
 [ 0.    0.    0.25  0.  ]
 [ 0.    2.    0.    0.  ]]
matches torch autograd: True
avg-pool output:
[[2.5  2.  ]
 [2.25 1.75]]
dL/dX:
[[ 0.125   0.125  -0.25   -0.25  ]
 [ 0.125   0.125  -0.25   -0.25  ]
 [ 0.5     0.5     0.0625  0.0625]
 [ 0.5     0.5     0.0625  0.0625]]
matches torch autograd: True
3x3/s2 max-pool of a centre spike: [[9.0, 9.0], [9.0, 9.0]] -> gradient at the spike: 4.0
GAP output shape: (4,) [-0.0734  0.3314  0.0869 -0.0205]
VGG-16 receptive fields (r = width in input pixels, j = jump):
conv1_1:3/1  conv1_2:5/1  pool1:6/2  conv2_1:10/2  conv2_2:14/2  pool2:16/4  conv3_1:24/4  conv3_2:32/4  conv3_3:40/4  pool3:44/8  conv4_1:60/8  conv4_2:76/8  conv4_3:92/8  pool4:100/16  conv5_1:132/16  conv5_2:164/16  conv5_3:196/16  pool5:212/32
four 3x3 convs, dilation 1,2,4,8: [3, 7, 15, 31]
unit 3 of the last layer sees input pixels (2, 11) | unit 0 (border, clipped): (0, 5)
shift | conv map: equivariance err, change | 2x2 max-pool change | two pools | GAP change
  1   |   0.0e+00   0.859     |   0.761           |  0.476   | 2.5e-17
  2   |   0.0e+00   1.182     |   1.037           |  0.674   | 0.0e+00
  3   |   0.0e+00   1.255     |   1.256           |  0.935   | 2.5e-17
  4   |   0.0e+00   1.322     |   1.275           |  1.096   | 0.0e+00`,
      },
      {
        t: "list",
        items: [
          "Max pooling's backward pass needs the argmax positions from the forward pass — why frameworks store pooling indices (and why `nn.MaxPool2d(return_indices=True)` + `nn.MaxUnpool2d` can undo the routing, as in SegNet-style decoders).",
          "Using `dx[...] += ...` (not `=`) matters for overlapping windows: the centre spike collects gradient from all four windows.",
          "Tracing back unit 3 gives input pixels 2–11, a width of 10, matching r = 10 for conv–conv–pool–conv. Unit 0 is clipped to 0–5 by the border.",
          "The shift experiment uses `np.roll` on an image whose shape stays away from the border. So the conv map is exactly equivariant (error 0) and GAP is exactly invariant up to round-off.",
        ],
      },
    ],

    // -------------------------------------------------------------- framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "**1. A complete CNN with its shape/parameter table, plus three head/downsampling variants trained on the same data.** About 15 seconds on a CPU.",
              },
              {
                t: "code",
                lang: "python",
                title: "small_cnn_digits.py",
                code: String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0   # (1797, 1, 8, 8)
y = torch.tensor(digits.target)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=digits.target)
X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]


def block(c_in, c_out, down):
    """Two 3x3 convs, then downsample by 2 with max-pool or with a stride-2 conv."""
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
              nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU()]
    if down == "maxpool":
        layers.append(nn.MaxPool2d(2))
    else:                                    # learnable downsampling
        layers += [nn.Conv2d(c_out, c_out, 3, stride=2, padding=1), nn.ReLU()]
    return layers


def make(down="maxpool", head="gap"):
    feats = block(1, 16, down) + block(16, 32, down)            # 1x8x8 -> 32x2x2
    if head == "gap":
        tail = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)]
    else:
        tail = [nn.Flatten(), nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Linear(64, 10)]
    return nn.Sequential(*feats, *tail)


def fit(model, epochs=15, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), 64):
            b = order[i:i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(X_tr[b]), y_tr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(X_te).argmax(1) == y_te).float().mean().item()


# ---- shape and parameter table of the main model (max-pool + GAP head) ----
torch.manual_seed(0)
net = make("maxpool", "gap")
h = X_tr[:1]
print(f"{'layer':18s} {'output shape':14s} params")
for layer in net:
    h = layer(h)
    n = sum(p.numel() for p in layer.parameters())
    print(f"{layer.__class__.__name__:18s} {str(tuple(h.shape[1:])):14s} {n}")
print("total parameters:", sum(p.numel() for p in net.parameters()))

# ---- three variants, same seed and schedule ----
for down, head in [("maxpool", "gap"), ("maxpool", "flatten"), ("stride2", "gap")]:
    torch.manual_seed(0)
    model = make(down, head)
    n = sum(p.numel() for p in model.parameters())
    acc = fit(model)
    print(f"{down:8s} + {head:8s} head: {n:6d} params, test accuracy {acc:.4f}")

# ---- the GAP head accepts any input size; the flatten head does not ----
big = nn.functional.interpolate(X_te[:5], size=(12, 12), mode="bilinear", align_corners=False)
print("GAP model on 12x12 input ->", tuple(net(big).shape))
try:
    make("maxpool", "flatten")(big)
except RuntimeError as e:
    print("flatten model on 12x12 input -> RuntimeError:", str(e).split(" (")[0])`,
                output: String.raw`layer              output shape   params
Conv2d             (16, 8, 8)     160
ReLU               (16, 8, 8)     0
Conv2d             (16, 8, 8)     2320
ReLU               (16, 8, 8)     0
MaxPool2d          (16, 4, 4)     0
Conv2d             (32, 4, 4)     4640
ReLU               (32, 4, 4)     0
Conv2d             (32, 4, 4)     9248
ReLU               (32, 4, 4)     0
MaxPool2d          (32, 2, 2)     0
AdaptiveAvgPool2d  (32, 1, 1)     0
Flatten            (32,)          0
Linear             (10,)          330
total parameters: 16698
maxpool  + gap      head:  16698 params, test accuracy 0.9756
maxpool  + flatten  head:  25274 params, test accuracy 0.9800
stride2  + gap      head:  28266 params, test accuracy 0.9511
GAP model on 12x12 input -> (5, 10)
flatten model on 12x12 input -> RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied`,
              },
              {
                t: "list",
                items: [
                  "`nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()` is the standard way to write a GAP head (same as `x.mean(dim=(2, 3))`).",
                  "The GAP model takes a 12×12 input and still outputs 10 logits. The flatten model fails because its first linear layer expects exactly 32·2·2 = 128 inputs.",
                  "On this tiny dataset the three variants land within a few points of each other. Exact accuracies can vary slightly across PyTorch versions and hardware, even with fixed seeds.",
                ],
              },
              {
                t: "p",
                text: "**2. Adaptive pooling and what classifier heads cost.**",
              },
              {
                t: "code",
                lang: "python",
                title: "adaptive_pooling.py",
                code: String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adaptive pooling picks the windows for you: bin i covers [floor(i*n/m), ceil((i+1)*n/m))
x = torch.tensor([[[3., 1., 4., 1., 5., 9., 2.]]])          # length n = 7, want m = 3 outputs
bins = [(math.floor(i * 7 / 3), math.ceil((i + 1) * 7 / 3)) for i in range(3)]
print("bins:", bins)
print("adaptive avg:", [round(v, 4) for v in F.adaptive_avg_pool1d(x, 3).squeeze().tolist()])
print("adaptive max:", F.adaptive_max_pool1d(x, 3).squeeze().tolist())

# Global average pooling = adaptive pooling to 1x1; output size no longer depends on the input
gap = nn.AdaptiveAvgPool2d(1)
pool7 = nn.AdaptiveAvgPool2d((7, 7))                          # what torchvision's VGG uses
for hw in [(7, 7), (14, 14), (10, 13), (56, 56)]:
    fmap = torch.randn(2, 512, *hw)
    print(f"input {str(tuple(fmap.shape)):20s} -> GAP {tuple(gap(fmap).shape)}"
          f"  adaptive 7x7 {tuple(pool7(fmap).shape)}")
fmap = torch.randn(1, 4, 6, 6)
print("GAP equals mean over H, W:", torch.allclose(gap(fmap).flatten(1), fmap.mean(dim=(2, 3))))

# Parameter cost of the classifier head on a 512x7x7 feature map (VGG-16 sizes)
fc_head = nn.Sequential(nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(),
                        nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
gap_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000))
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"flatten + 3 FC head: {count(fc_head):,} params")
print(f"GAP + 1 FC head:     {count(gap_head):,} params  ({count(fc_head) / count(gap_head):.0f}x fewer)")`,
                output: String.raw`bins: [(0, 3), (2, 5), (4, 7)]
adaptive avg: [2.6667, 3.3333, 5.3333]
adaptive max: [4.0, 5.0, 9.0]
input (2, 512, 7, 7)       -> GAP (2, 512, 1, 1)  adaptive 7x7 (2, 512, 7, 7)
input (2, 512, 14, 14)     -> GAP (2, 512, 1, 1)  adaptive 7x7 (2, 512, 7, 7)
input (2, 512, 10, 13)     -> GAP (2, 512, 1, 1)  adaptive 7x7 (2, 512, 7, 7)
input (2, 512, 56, 56)     -> GAP (2, 512, 1, 1)  adaptive 7x7 (2, 512, 7, 7)
GAP equals mean over H, W: True
flatten + 3 FC head: 123,642,856 params
GAP + 1 FC head:     513,000 params  (241x fewer)`,
              },
              {
                t: "p",
                text: "**3. Measuring the effective receptive field with autograd.** Backpropagate from the centre unit of random conv–ReLU stacks and see how the gradient mass spreads over the input.",
              },
              {
                t: "code",
                lang: "python",
                title: "effective_receptive_field.py",
                code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
SIZE = 65                  # odd input size, so there is a single centre pixel
c = SIZE // 2


def erf_profile(depth, trials=20, relu=True):
    """Average |d out_centre / d input| over random 3x3 conv stacks (stride 1, 'same' padding)."""
    acc = torch.zeros(SIZE, SIZE)
    for _ in range(trials):
        layers = []
        for i in range(depth):
            conv = nn.Conv2d(1 if i == 0 else 8, 8, 3, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight)          # He init: keeps signal scale stable
            layers += [conv, nn.ReLU()] if relu else [conv]
        net = nn.Sequential(*layers)
        x = torch.randn(1, 1, SIZE, SIZE, requires_grad=True)
        out = net(x)
        out[0, :, c, c].sum().backward()                  # centre unit, summed over channels
        acc += x.grad[0, 0].abs()
    return acc / trials


print("depth | theoretical RF | width holding 50% / 90% of |grad| mass | std of profile (px)")
for depth in [2, 4, 8, 16]:
    g = erf_profile(depth)
    g = g / g.sum()
    theo = 2 * depth + 1
    # smallest centred square containing a given share of the total gradient magnitude
    widths = {}
    for share in (0.5, 0.9):
        for half in range(SIZE // 2 + 1):
            if g[c - half:c + half + 1, c - half:c + half + 1].sum() >= share:
                widths[share] = 2 * half + 1
                break
    coords = torch.arange(SIZE, dtype=torch.float32) - c
    std = (g.sum(0) * coords**2).sum().sqrt().item()  # std of the column marginal
    print(f"{depth:5d} | {theo:14d} | {widths[0.5]:16d} / {widths[0.9]:<20d} | {std:.2f}")`,
                output: String.raw`depth | theoretical RF | width holding 50% / 90% of |grad| mass | std of profile (px)
    2 |              5 |                5 / 5                    | 1.30
    4 |              9 |                5 / 9                    | 1.96
    8 |             17 |                7 / 13                   | 3.05
   16 |             33 |               11 / 19                   | 4.50`,
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "The same model in Keras (channels-last). `MaxPooling2D`, `AveragePooling2D` and `GlobalAveragePooling2D` match the PyTorch layers. `GlobalAveragePooling2D` already returns a flat (N, C) tensor, so no `Flatten` is needed.",
              },
              {
                t: "code",
                lang: "python",
                title: "small_cnn_keras.py (not executed in CI)",
                code: String.raw`import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(0)
digits = load_digits()
X = (digits.images.astype("float32") / 16.0)[..., None]       # (1797, 8, 8, 1) NHWC
X_tr, X_te, y_tr, y_te = train_test_split(X, digits.target, test_size=0.25,
                                          random_state=0, stratify=digits.target)

def conv(c):
    return tf.keras.layers.Conv2D(c, 3, padding="same", activation="relu")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(None, None, 1)),        # any spatial size, thanks to GAP
    conv(16), conv(16), tf.keras.layers.MaxPooling2D(2),
    conv(32), conv(32), tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.GlobalAveragePooling2D(),      # (N, 32)
    tf.keras.layers.Dense(10),
])
model.summary()                                    # 16,698 trainable parameters
model.compile(optimizer=tf.keras.optimizers.Adam(5e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(X_tr, y_tr, batch_size=64, epochs=15, verbose=0)
print("test accuracy:", model.evaluate(X_te, y_te, verbose=0)[1])
print("12x12 input ->", model(tf.image.resize(X_te[:5], (12, 12))).shape)`,
                output: "Prints the layer summary (16,698 trainable parameters, the same count as the PyTorch model), a test accuracy in the high 0.9s, and the output shape (5, 10) for 12×12 inputs.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      { t: "h", text: "Ways to shrink and summarise feature maps" },
      {
        t: "table",
        head: ["Operator", "What it does", "Params", "Pros", "Cons", "Typical use"],
        rows: [
          ["Max pooling 2×2 / s2", "Max of each block", "0", "Keeps strongest values; cheap; some shift tolerance", "Throws away 75% of values; aliasing; sparse gradients", "VGG, classic CNNs"],
          ["Overlapping max pool 3×3 / s2", "Max over overlapping windows", "0", "A bit smoother, less aliasing", "More compute than 2×2", "AlexNet, ResNet stem"],
          ["Average pooling", "Mean of each block", "0", "Blurs (low-pass), dense gradients", "Weakens strong, rare features", "Transition layers (DenseNet), LeNet-style"],
          ["Stride-2 convolution", "Convolution at every other position", "C_out·C_in·k² + C_out", "Learned downsampling", "Extra compute; aliasing", "ResNet, MobileNet, most modern CNNs"],
          ["Blur + subsample (anti-aliased)", "Low-pass filter, then stride", "0 (fixed blur)", "Better shift stability", "Small extra cost", "Shift-robust CNNs (Zhang 2019)"],
          ["Global average pooling", "Mean of each whole map", "0", "Tiny head, any input size, invariance, CAM", "Discards spatial layout", "ResNet, Inception, MobileNet, EfficientNet heads"],
          ["Global max pooling", "Max of each whole map", "0", "Detects presence anywhere", "Gradient to one position only", "Multi-instance tasks, some text CNNs"],
          ["Adaptive pooling", "Fixed output size from any input", "0", "Decouples head from input size", "Bins of unequal size", "torchvision VGG (7×7), SPP-style heads"],
          ["Flatten + FC", "Reshape and dense layers", "C·H·W·m + …", "Can use absolute position", "Huge parameter count; fixed input size", "LeNet, AlexNet, VGG"],
          ["Attention pooling", "Learned weighted average", "Small", "Content-dependent aggregation", "More complex", "CLIP's ResNet variant, ViT class token"],
        ],
      },
      { t: "h", text: "Ways to grow the receptive field" },
      {
        t: "table",
        head: ["Strategy", "RF growth per layer", "Cost", "Side effects"],
        rows: [
          ["Stack 3×3 convs (stride 1)", "+2·j", "Linear in depth", "Slow growth; ERF grows only as √depth"],
          ["Downsample (pool / stride)", "Multiplies future growth by s", "Reduces later compute", "Loses resolution (bad for dense prediction)"],
          ["Larger kernels (7×7, 31×31)", "+(k − 1)·j", "Quadratic in k", "ConvNeXt (7×7) and other large-kernel designs revisit this"],
          ["Dilated convs", "+d(k − 1)·j", "Same as undilated", "Gridding if dilations share a factor"],
          ["Global pooling / attention", "Whole image at once", "Cheap (GAP) or quadratic (attention)", "Loses or must re-learn spatial layout"],
        ],
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Image classification backbones** — VGG puts 2×2 max pooling between stages. ResNet uses a 3×3/s2 max pool in the stem, stride-2 convolutions between stages and global average pooling before the classifier. MobileNet and EfficientNet end in GAP + a linear layer.",
          "**Inputs of different sizes** — GAP and adaptive pooling let one trained classifier handle different resolutions (e.g. testing at a higher resolution than training).",
          "**Weakly supervised localisation and explanation** — Class Activation Mapping (Zhou et al., 2016) uses the GAP + linear head to draw heatmaps of where each class's evidence lies; Grad-CAM extends it using gradients.",
          "**Segmentation and dense prediction** — must balance receptive field and resolution: dilated convolutions (DeepLab) or encoder–decoder designs with skip connections (U-Net) give large receptive fields without losing detail.",
          "**Object detection** — feature maps at strides of 8, 16 and 32 pixels have receptive fields suited to small, medium and large objects; feature pyramid networks combine them. RoI pooling/RoIAlign are adaptive pooling over proposed boxes.",
          "**Audio and time series** — 1-D max/average pooling and global pooling over time summarise spectrogram or sensor features. WaveNet-style dilated stacks reach receptive fields of thousands of samples.",
        ],
      },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Pooling" },
      {
        t: "list",
        items: [
          "**Strengths**: no parameters, cheap, cuts compute 4× per 2×2 stage, enlarges receptive fields, gives some tolerance to small shifts and distortions.",
          "**Weaknesses**: throws information away for good (bad for tasks needing exact positions — segmentation, keypoints); max pooling's gradient is sparse; subsampling aliases, so outputs can change noticeably under 1-pixel shifts; its invariance is only partial and local.",
          "**Strided convolution vs pooling**: learnable and often slightly better at scale, but costs parameters and compute. Neither is shift-invariant without explicit anti-aliasing.",
        ],
      },
      { t: "h", text: "Classifier heads" },
      {
        t: "table",
        head: ["", "Flatten + FC", "GAP + linear"],
        rows: [
          ["Parameters (VGG-16 sizes)", "123,642,856", "513,000"],
          ["Input size", "Fixed", "Any"],
          ["Uses absolute position", "Yes", "No (translation invariant)"],
          ["Overfitting risk", "High (needs dropout)", "Low (built-in regulariser)"],
          ["Interpretability", "Low", "Class activation maps for free"],
        ],
      },
      { t: "h", text: "Receptive field" },
      {
        t: "list",
        items: [
          "A theoretical RF smaller than the objects you care about is a hard limit: one unit literally cannot see the whole object.",
          "A theoretical RF much larger than needed is not wasted: the effective RF is much smaller, so a safety margin is healthy.",
          "Growing the RF fast with early downsampling is cheap but loses resolution. So dense-prediction networks downsample less, use dilation, or upsample back with skip connections.",
        ],
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Saying pooling makes CNNs translation invariant.** Local pooling only tolerates small shifts within a window; in our experiment a 1-pixel shift still changes a 2×2-pooled map by 76% (relative L2 change). Output invariance comes mainly from global pooling and data augmentation.",
          "**Computing pooling output size as n/k.** The formula is ⌊(n + 2p − k)/s⌋ + 1. A 7×7 map with 2×2/s2 pooling becomes 3×3, not 3.5; a 3×3/s2 pool with p = 1 on 112 gives 56.",
          "**Forgetting that pooling works per channel.** Max pooling a 64×32×32 tensor gives 64×16×16 — never fewer channels. To reduce channels, use a 1×1 convolution.",
          "**Wrong flatten size in the first linear layer.** Hard-coding `nn.Linear(32*7*7, …)` breaks when the input size or padding changes. Fix: compute it with a dummy tensor, use `nn.LazyLinear`, or use GAP/adaptive pooling so the size is fixed.",
          "**Assigning instead of adding in a hand-written max-pool backward.** With overlapping windows one input can be the max of several windows; use `+=`.",
          "**Counting padding or element-wise layers in the receptive field.** ReLU, BatchNorm and dropout don't change r; padding moves the field but doesn't enlarge it. But leaving pooling layers out underestimates it.",
          "**Multiplying r by the stride.** The stride changes the *jump* for later layers, not the current layer's growth: r_l = r_{l−1} + (k_l − 1)·j_{l−1} uses the previous jump.",
          "**Ignoring dilation in RF arithmetic.** A 3×3 kernel with dilation 4 spans 9 pixels: use k_eff = d(k − 1) + 1.",
          "**Treating theoretical and effective receptive fields as the same.** A unit whose theoretical RF covers the image may still be driven mostly by a small central region; check with gradients when long-range context matters.",
          "**Using max pooling where averaging is meant (or the reverse) in the head.** Global max pooling sends gradient to one position per map and can train slowly; GAP is the standard choice for classification.",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What does 2×2 max pooling with stride 2 do, and how many parameters does it have?", a: "It splits each channel into non-overlapping 2×2 blocks and keeps each block's largest value: height and width halve (⌊n/2⌋), channels stay. Zero parameters. It cuts later compute by about 4×, enlarges later receptive fields and gives some tolerance to small shifts." },
      { t: "qa", level: "beginner", q: "What is the difference between max pooling and average pooling?", a: "Max pooling keeps each window's largest value (the strongest detection); its gradient goes only to the argmax. Average pooling outputs the mean, like a blur then subsampling; its gradient is shared equally (1/k² each). Max pooling is common inside classic CNNs on ReLU maps; averaging appears mostly as global average pooling at the end." },
      { t: "qa", level: "beginner", q: "What is a receptive field?", a: "The region of the input image that can influence a given unit. One 3×3 convolution sees 3×3; two stacked see 5×5, three 7×7, and downsampling makes later layers grow it faster. It limits context: a unit cannot recognise an object larger than its receptive field." },
      { t: "qa", level: "intermediate", q: "Derive the backward pass of max pooling and average pooling.", a: "Max pooling: Y_ij equals X at the argmax (u*, v*) of window W_ij, so ∂Y_ij/∂X_uv is 1 at the argmax and 0 elsewhere. Hence ∂L/∂X_uv = Σ_ij G_ij·1[(u,v) = argmax_ij]: the gradient goes to the winners, summed if an input wins several overlapping windows. Average pooling: ∂Y_ij/∂X_uv = 1/k² for every (u,v) in the window, so ∂L/∂X_uv = Σ over windows containing (u,v) of G_ij/k². Example: with G = 0.8 on one 2×2 window, max sends 0.8 to the argmax; average sends 0.2 to each of the four inputs." },
      { t: "qa", level: "intermediate", q: "Why did modern CNNs replace flatten + fully connected heads with global average pooling?", a: "Parameters and generalisation. Flattening a 512×7×7 map into 4,096 units needs 102.8 M weights (VGG-16's three FC layers hold 123.6 M of its 138.4 M parameters); GAP + linear needs 513 K for 1,000 classes. GAP has no parameters, regularises, accepts any input size, makes the prediction translation invariant and enables class activation maps. Proposed in Network In Network (Lin et al., 2013); used in GoogLeNet, ResNet and later models." },
      { t: "qa", level: "intermediate", q: "Compute the receptive field of the ResNet stem: conv 7×7/s2, max-pool 3×3/s2, then two 3×3/s1 convs. What are the output sizes for a 224×224 input?", a: "r: conv7/s2 → 1 + 6·1 = 7, j = 2; pool3/s2 → 7 + 2·2 = 11, j = 4; conv3 → 11 + 2·4 = 19; conv3 → 27, j = 4. Sizes: conv (p = 3): ⌊(224 + 6 − 7)/2⌋ + 1 = 112; pool (p = 1): ⌊(112 + 2 − 3)/2⌋ + 1 = 56; the 3×3 convs with p = 1 keep 56×56. So after four layers each unit sees 27×27 pixels, and neighbouring units are 4 pixels apart." },
      { t: "qa", level: "advanced", q: "Is a CNN translation invariant? Discuss equivariance, pooling and aliasing.", a: "Stride-1 convolution is translation equivariant (away from borders). Subsampling (pooling or striding) breaks exact equivariance: it holds only for shifts that are multiples of the total stride. Other shifts can change the representation a lot, because subsampling without blurring aliases high frequencies (Zhang, 2019, showed classifier outputs wobble under 1-pixel shifts and proposed blurring first). Local max pooling gives only partial tolerance (in our experiment a 1-pixel shift still changes a pooled map by 76%). Approximate invariance of the prediction comes from global average pooling (exactly invariant to shifts that keep content inside the image, apart from border and subsampling effects earlier on) and from training on shifted/cropped data." },
      { t: "qa", level: "advanced", q: "What is the effective receptive field and why is it smaller than the theoretical one?", a: "The input region that really influences a unit, measured by |∂y/∂x_i|. In a conv stack the centre pixel has the most paths to the output, edge pixels the fewest. With random weights contributions add up like independent steps, so the profile is roughly Gaussian with std ∝ √n for n layers, while the theoretical RF grows ∝ n (Luo et al., 2016). Our measurement: 16 stacked 3×3 conv–ReLU layers have theoretical RF 33, but half the gradient mass lies in the central 11×11 and the std is 4.5 px. Downsampling, dilation, larger kernels and attention enlarge it; training also tends to widen it somewhat." },
      { t: "qa", level: "advanced", q: "How does dilation affect the receptive field, and what is the gridding problem?", a: "A k×k kernel with dilation d spaces its taps d apart: effective size d(k − 1) + 1 with the same k² parameters, so r grows by d(k − 1)·j per layer without downsampling. Dilations 1, 2, 4, 8 give r = 3, 7, 15, 31 for four 3×3 layers and cover every pixel. But if all layers share a common factor (2, 2, 2), every tap lands on pixels of the same parity: the unit sees only every other pixel in each direction (a checkerboard of holes) — gridding artefacts. Fixes: mix dilation rates with no common factor (hybrid dilated convolution) or add undilated layers." },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            kind: "numeric",
            q: "A 64×31×31 tensor goes through 2×2 max pooling with stride 2 (no padding). How many values does the output contain?",
            answer: 14400,
            tolerance: 0,
            explain: "⌊(31 − 2)/2⌋ + 1 = 15, channels unchanged: 64·15·15 = 14,400.",
          },
          {
            kind: "numeric",
            q: "An upstream gradient of 0.8 arrives at one output of a 2×2 average-pooling layer. What gradient does each of the four inputs receive?",
            answer: 0.2,
            tolerance: 0.001,
            explain: "Average pooling shares the gradient equally: 0.8/4 = 0.2.",
          },
          {
            q: "In the backward pass of max pooling, the inputs that were not the maximum of their window receive…",
            options: ["the average of the upstream gradient", "zero gradient", "the same gradient as the maximum", "a gradient proportional to their value"],
            answer: 1,
            explain: "Locally the max just picks one input, so only the argmax has a non-zero derivative; all others get 0.",
          },
          {
            kind: "numeric",
            q: "Layers: conv 3×3 (s1) → max-pool 2×2 (s2) → conv 3×3 (s1) → conv 3×3 (s1). What is the receptive field width of the last layer?",
            answer: 12,
            tolerance: 0,
            explain: "r: 3 (j = 1) → 3 + 1·1 = 4 (j = 2) → 4 + 2·2 = 8 → 8 + 2·2 = 12.",
          },
          {
            kind: "numeric",
            q: "How many parameters has a GAP + linear head that maps 2,048 feature maps to 1,000 classes (ResNet-50)?",
            answer: 2049000,
            tolerance: 0,
            explain: "GAP has none; the linear layer has 2,048·1,000 weights + 1,000 biases = 2,049,000.",
          },
          {
            kind: "numeric",
            q: "What is the effective kernel size of a 3×3 convolution with dilation 4?",
            answer: 9,
            tolerance: 0,
            explain: "k_eff = d(k − 1) + 1 = 4·2 + 1 = 9.",
          },
          {
            q: "Which component is mainly responsible for the (approximate) translation invariance of a modern CNN classifier's prediction?",
            options: ["Each 3×3 convolution", "Local 2×2 max pooling", "Global average pooling (plus augmentation)", "The ReLU non-linearity"],
            answer: 2,
            explain: "Convolutions are equivariant and local pooling only tolerates small shifts; averaging over all positions removes any dependence on where features are.",
          },
          {
            q: "Doubling the number of stacked 3×3 stride-1 convolutions approximately…",
            options: [
              "doubles both the theoretical and effective receptive field",
              "doubles the theoretical RF but multiplies the effective RF width by about √2",
              "leaves the effective RF unchanged",
              "quadruples the effective RF",
            ],
            answer: 1,
            explain: "Theoretical r = 2n + 1 grows linearly; the std of the gradient profile grows like √n (measured: 3.05 → 4.50 px from 8 to 16 layers).",
          },
        ],
      },
      {
        t: "exercise",
        title: "Receptive field and shapes of the ResNet stem",
        prompt: "A 224×224 image passes through: conv 7×7, stride 2, padding 3 → max-pool 3×3, stride 2, padding 1 → conv 3×3, stride 1, padding 1 → conv 3×3, stride 1, padding 1. (a) Give the spatial output size after each layer. (b) Fill in r and j after each layer. (c) Which input columns does output unit 10 of the last layer see? (d) If the second conv had dilation 2 (padding 2), what would the final r be?",
        hint: "Sizes: ⌊(n + 2p − k_eff)/s⌋ + 1. RF: r_l = r_{l−1} + (k_eff − 1)·j_{l−1}, j_l = j_{l−1}·s_l. Back-project [a, b] → [a·s − p, b·s − p + d(k − 1)], going from the last layer to the first.",
        solution: [
          {
            t: "table",
            head: ["Layer", "Output size", "r", "j"],
            rows: [
              ["Input", "224", "1", "1"],
              ["Conv 7×7 / s2, p3", "⌊(224 + 6 − 7)/2⌋ + 1 = 112", "1 + 6·1 = 7", "2"],
              ["Max-pool 3×3 / s2, p1", "⌊(112 + 2 − 3)/2⌋ + 1 = 56", "7 + 2·2 = 11", "4"],
              ["Conv 3×3 / s1, p1", "56", "11 + 2·4 = 19", "4"],
              ["Conv 3×3 / s1, p1", "56", "19 + 2·4 = 27", "4"],
            ],
          },
          {
            t: "p",
            text: "(c) Tracing unit 10 back: last conv [10·1 − 1, 10·1 − 1 + 2] = [9, 11]; previous conv [8, 12]; max-pool [8·2 − 1, 12·2 − 1 + 2] = [15, 25]; first conv [15·2 − 3, 25·2 − 3 + 6] = [27, 53]. Width 53 − 27 + 1 = 27 = r. (d) The dilated conv has k_eff = 2·2 + 1 = 5, so it adds (5 − 1)·4 = 16 instead of 8, giving r = 19 + 16 = 35.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Verify the receptive-field formula with autograd",
        prompt: "Write a function that computes (k_eff, r, j) for a stack of (type, k, s, p, d) tuples, and a back-projection function giving the exact input interval of one output unit. Build the same stack in PyTorch with single-channel convolutions (strictly positive weights, so nothing cancels) and average pooling. Backpropagate from the centre output unit of a 64×64 input and check that the pixels with non-zero gradient span exactly the predicted interval. Stack: conv 3×3/s1/p1 → conv 3×3/s2/p1 → pool 2×2/s2 → conv 3×3/s1/p2/d2 → conv 3×3/s1/p1.",
        hint: "Use average pooling: it gives every tap a gradient. Max pooling would send it to one pixel and make the non-zero region look smaller than the receptive field.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "verify_receptive_field.py",
            code: String.raw`import torch
import torch.nn as nn

# (type, k, s, p, d): a stack mixing stride, pooling and dilation
spec = [("conv", 3, 1, 1, 1), ("conv", 3, 2, 1, 1), ("pool", 2, 2, 0, 1),
        ("conv", 3, 1, 2, 2), ("conv", 3, 1, 1, 1)]


def rf_table(spec):
    r, j, rows = 1, 1, []
    for kind, k, s, p, d in spec:
        k_eff = d * (k - 1) + 1
        r, j = r + (k_eff - 1) * j, j * s
        rows.append((kind, k_eff, r, j))
    return rows


for kind, k_eff, r, j in rf_table(spec):
    print(f"{kind:4s} k_eff={k_eff}  r={r:3d}  j={j}")

def back_project(spec, unit):
    """Exact input interval of one output unit: walk backwards, unit i -> [i*s - p, i*s - p + d*(k-1)]."""
    lo, hi = unit, unit
    for kind, k, s, p, d in reversed(spec):
        lo, hi = lo * s - p, hi * s - p + d * (k - 1)
    return lo, hi


# Empirical check: which input pixels receive a non-zero gradient from one output unit?
torch.manual_seed(0)
layers = []
for kind, k, s, p, d in spec:
    if kind == "conv":
        conv = nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d, bias=False)
        nn.init.uniform_(conv.weight, 0.5, 1.5)       # strictly positive: no accidental cancellation
        layers.append(conv)
    else:
        layers.append(nn.AvgPool2d(k, s))   # average pool: every tap gets gradient (max would pick one)
net = nn.Sequential(*layers)
x = torch.rand(1, 1, 64, 64, requires_grad=True)
out = net(x)
i = out.shape[-1] // 2
out[0, 0, i, i].backward()
nz = x.grad[0, 0].nonzero()
rows, cols = nz[:, 0], nz[:, 1]
print("output size:", tuple(out.shape[-2:]), " unit:", (i, i))
print("back-projected interval:", back_project(spec, i))
print("input rows", rows.min().item(), "to", rows.max().item(),
      "-> width", (rows.max() - rows.min() + 1).item())
print("input cols", cols.min().item(), "to", cols.max().item(),
      "-> width", (cols.max() - cols.min() + 1).item())`,
            output: String.raw`conv k_eff=3  r=  3  j=1
conv k_eff=3  r=  5  j=2
pool k_eff=2  r=  7  j=4
conv k_eff=5  r= 23  j=4
conv k_eff=3  r= 31  j=4
output size: (16, 16)  unit: (8, 8)
back-projected interval: (18, 48)
input rows 18 to 48 -> width 31
input cols 18 to 48 -> width 31`,
          },
          {
            t: "p",
            text: "The formula predicts r = 31, back-projection predicts input columns 18–48, and autograd finds non-zero gradient exactly on rows and columns 18–48. Notice how the dilated layer (k_eff = 5), coming after a total stride of 4, adds 16 pixels at once.",
          },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "Pooling summarises each window of each channel with a fixed rule: max (strongest response) or average (mean). No parameters; output size ⌊(n + 2p − k)/s⌋ + 1.",
          "Backward: max pooling sends each upstream gradient to its argmax (adding up over overlapping windows); average pooling gives G/k² to every input in the window.",
          "Stride-2 convolutions downsample with learned weights and are the norm in modern CNNs; any subsampling aliases unless you blur first.",
          "Flatten + FC heads are huge (VGG-16: 123.6 M of 138.4 M parameters) and lock the input size. GAP + linear (Network In Network, 2013) is tiny (513 K), works with any size and is translation invariant. Adaptive pooling fixes the output size for any input.",
          "Convolution is equivariant; local pooling only tolerates small shifts (more with depth); global pooling gives invariance.",
          "Receptive field: r_l = r_{l−1} + (k_eff − 1)·j_{l−1}, j_l = j_{l−1}·s_l, k_eff = d(k − 1) + 1. Growth speeds up after downsampling; VGG-16's pool5 sees 212×212 pixels.",
          "Effective RF ≈ Gaussian, width ∝ √depth versus theoretical ∝ depth (16 stacked 3×3 convs: 33 theoretical, half the influence within 11×11).",
          "A full CNN = [conv–ReLU ×2 → downsample] × stages → GAP → linear. Track shape, parameters and receptive field layer by layer.",
        ],
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial L}{\partial X_{uv}}\Big|_{\max} = \sum_{ij} G_{ij}\mathbf{1}[(u,v)=\arg\max \mathcal{W}_{ij}],\quad \frac{\partial L}{\partial X_{uv}}\Big|_{\text{avg}} = \sum_{ij:(u,v)\in\mathcal{W}_{ij}}\frac{G_{ij}}{k^2},\quad r_\ell = r_{\ell-1} + \big(d_\ell(k_\ell-1)\big)j_{\ell-1},\quad j_\ell = j_{\ell-1}s_\ell`,
        caption: "The lesson's key formulas.",
      },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Dilated, Transposed, Depthwise and Pointwise Convolutions](/learn/m08/conv-variants) — dilation in full, transposed convolutions for upsampling, and the depthwise-separable layers of efficient networks.",
          "[Classic CNN Architectures: LeNet to ConvNeXt](/learn/m08/cnn-architectures) — each architecture is a choice of this lesson's blocks: where to downsample, which pooling, which head, how large a receptive field.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — the layer between convolution and ReLU in almost every modern CNN.",
          "[Residual and Skip Connections](/learn/m07/residual-connections) — how very deep stacks (and so large receptive fields) became trainable.",
        ],
      },
    ],
  },
};

export default lesson;
