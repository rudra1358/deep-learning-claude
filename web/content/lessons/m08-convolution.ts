import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m08/convolution",
  title: "The Convolution Operation",
  summary:
    "Why fully connected layers fail on images, and how sliding a small kernel across the pixels fixes it: cross-correlation vs convolution, a 5×5 example with every multiply-add, feature maps, stride, padding, the output-size formula, multi-channel kernels and their parameter count, parameter sharing, translation equivariance, receptive fields, pooling, and the backward pass.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m08/images-as-tensors", "m02/vectors-matrices", "m03/activation-functions", "m04/backpropagation"],
  next: ["m08/pooling-receptive-field", "m08/conv-variants", "m08/cnn-architectures"],
  terms: ["cnn", "convolution", "kernel", "feature-map", "stride", "padding"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A **convolutional layer** computes each output from a small local patch of its input, using a small grid of weights — the **kernel** or **filter** — reused at every position. Slide one 3×3 kernel over an image, write down a weighted sum at each spot, and you get a new 2-D array called a **feature map**. It shows where in the image the kernel's pattern appears. Stack such layers, with non-linearities and downsampling in between, and you have a **convolutional neural network (CNN)**.",
      },
      {
        t: "p",
        text: "Why was it needed? Images are large, and fully connected layers treat every pixel as an unrelated input. A 224×224 RGB photo has 150,528 numbers. Connecting them to just 1,000 hidden units needs over 150 million weights in the first layer alone. Such a layer ignores that neighbouring pixels are related, and must re-learn the same detector at every position. Convolution builds two assumptions into the architecture — **locality** (useful patterns are small) and **stationarity** (a pattern means the same thing wherever it appears) — and cuts the parameter count by several orders of magnitude.",
      },
      {
        t: "p",
        text: "A short history. Kunihiko Fukushima's Neocognitron (1980) used layers of local feature detectors, inspired by Hubel and Wiesel's studies of the visual cortex. Yann LeCun and colleagues trained convolutional networks end-to-end with backpropagation in the late 1980s; their 1998 paper “Gradient-Based Learning Applied to Document Recognition” (LeNet-5) read handwritten digits and cheques. In 2012 AlexNet (Krizhevsky, Sutskever and Hinton) won the ImageNet challenge by a wide margin with a deep CNN trained on GPUs, and CNNs became the default for computer vision.",
      },
      {
        t: "p",
        text: "Today convolutions power classification, detection, segmentation, medical imaging, video, audio spectrograms and 1-D signals such as ECG. They also sit inside larger models: multimodal image encoders and the U-Nets in diffusion models. Even Vision Transformers usually start with a strided convolution that cuts the image into patches.",
      },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Images as Tensors and Why CNNs](/learn/m08/images-as-tensors) — an image as an array of pixel intensities with shape H×W×C (or C×H×W), what a channel is, and PyTorch's NCHW layout. We recap it briefly.",
          String.raw`[Vectors and Matrices](/learn/m02/vectors-matrices) — the dot product. Every convolution output is a dot product between a flattened image patch and a flattened kernel, so you should be comfortable with $\sum_i a_i b_i$.`,
          "[Activation Functions](/learn/m03/activation-functions) — especially ReLU, applied after almost every convolution.",
          "[Backpropagation](/learn/m04/backpropagation) — kernels are learned by gradient descent; the backward-pass section assumes you know how gradients flow through a weighted sum.",
          "Basic NumPy slicing (`x[r:r+k, c:c+k]`) and array shapes — the from-scratch code is built from these.",
        ],
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A stencil and a magnifying glass",
        text: "Imagine searching a big printed page for a small symbol. You do not take in the whole page at once. You slide a small cardboard window across it, row by row, and at each spot ask the same question: “does this look like the symbol?” You write a score on a grid of sticky notes laid out like the positions you checked. The window is the **kernel**, asking the same question everywhere is **weight sharing**, and the grid of notes is the **feature map**. Use several stencils (vertical stroke, horizontal stroke, curve) and you get several feature maps — one per pattern.",
      },
      {
        t: "p",
        text: "A kernel is a small grid of numbers. At each position, lay it over the image, multiply each kernel number by the pixel underneath, and add the nine products. If the pixels look like the kernel — large where it is positive, small where it is negative — the sum is large. If they look like the opposite, the sum is very negative. On a flat patch, a kernel whose numbers add to zero gives exactly zero. That is all a convolutional layer does. The power comes from many kernels, stacked layers, and learning the kernel numbers from data.",
      },
      {
        t: "p",
        text: "A concrete case: an image whose left half has brightness 10 and right half 0 (a vertical edge), and a kernel with columns of +1, 0, −1. Wherever the window sits in one flat region, left column minus right column is 0. Where the window straddles the edge, the left column sees 10s and the right sees 0s, giving 30. The feature map is a stripe of 30s exactly at the edge and 0 elsewhere: the kernel found the edge and located it.",
      },
      {
        t: "math",
        tex: String.raw`\underbrace{\begin{bmatrix}10&10&10&0&0&0\\10&10&10&0&0&0\\ \vdots&&&&&\vdots\\10&10&10&0&0&0\end{bmatrix}}_{6\times 6\ \text{image}}\ \star\ \begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix} = \begin{bmatrix}0&30&30&0\\0&30&30&0\\0&30&30&0\\0&30&30&0\end{bmatrix}`,
        caption: "A vertical-edge kernel responds only where the window straddles the edge (verified with NumPy).",
      },
      {
        t: "p",
        text: "Two bonuses come almost for free. First, the same kernel finds the edge wherever it is: shift the image and the stripe of 30s shifts by the same amount (**translation equivariance**). Second, the layer needs only 9 weights for an image of any size, instead of one weight per pixel per output. A deeper layer slides kernels over these feature maps, so it detects patterns of patterns — corners made of two edges, eyes made of curves — over ever larger regions of the original image.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Images as tensors (recap)" },
      {
        t: "p",
        text: "A grayscale image is an H×W array of intensities (0–255 as bytes, usually rescaled to [0, 1]). A colour image has three **channels** (red, green, blue): H×W×3 (**HWC**, channels-last: PIL, NumPy, Keras) or 3×H×W (**CHW**, channels-first: PyTorch). A batch adds a leading axis: N×C×H×W in PyTorch (**NCHW**). Inside a CNN, a hidden tensor's “channels” are feature maps, one per kernel of the previous layer.",
      },
      { t: "h", text: "Why fully connected layers fail on images" },
      {
        t: "list",
        items: [
          "**Parameter explosion.** A flattened 224×224×3 input has 224·224·3 = 150,528 entries. A dense layer with 1,000 units needs 150,528 × 1,000 = 150,528,000 weights plus 1,000 biases = 150,529,000 parameters — about 600 MB in float32, for one layer. A convolutional layer with 64 kernels of size 3×3×3 has 64·27 + 64 = 1,792 parameters, whatever the image size.",
          "**No notion of locality.** Flattening destroys the 2-D neighbourhoods: pixels (10, 10) and (10, 11) become entries 2,250 and 2,251 of a vector, no more related than pixel (200, 3). The network must rediscover from data that nearby pixels matter together.",
          "**No sharing across positions.** A dense layer that learns a cat's eye in the top-left corner knows nothing about an eye in the bottom-right; it needs separate weights and separate examples for every position.",
          "**Fixed input size.** A dense layer's weight matrix fixes the number of input pixels. A convolutional layer works on any H×W.",
        ],
      },
      { t: "h", text: "The convolution operation, step by step" },
      {
        t: "steps",
        items: [
          { title: "Choose a kernel", text: "A small k×k grid of weights K (usually k = 3, sometimes 1, 5 or 7) and a scalar bias b. A CNN learns these; classical image processing designs them by hand." },
          { title: "(Optional) pad the input", text: "Surround the H×W input with p rings of zeros, giving (H + 2p)×(W + 2p). Padding lets the kernel centre on border pixels and controls the output size." },
          { title: "Place the window", text: "Put the kernel's top-left corner at (0, 0) of the padded input. The k×k block under it is the current **window** (or patch)." },
          { title: "Multiply element-wise", text: "Multiply each window value by the kernel weight on top of it: k² products." },
          { title: "Sum and add bias", text: "Add the k² products and the bias. This one number is the output at this position.", tex: String.raw`Y[i,j] = b + \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_p[s\,i+m,\; s\,j+n]\,K[m,n]` },
          { title: "Slide by the stride", text: "Move the window s columns right (s = **stride**) and repeat. At the end of a row, go back to the left edge and move s rows down." },
          { title: "Stop when the kernel no longer fits", text: "The visited positions form the output grid — the **feature map** — of size H_out × W_out from the output-size formula. Positions where the kernel would hang off the padded input are skipped." },
          { title: "Apply a non-linearity", text: "Usually ReLU is applied to every value of the feature map. Without it, a stack of convolutions would collapse into one (larger) convolution, because linear maps composed are still linear." },
        ],
      },
      { t: "h", text: "Convolution or cross-correlation?" },
      {
        t: "p",
        text: "What deep-learning libraries call “convolution” is strictly **cross-correlation**: the kernel is laid over the window as it is. The convolution of signal processing and mathematics first **flips** the kernel (rotates it 180°), then slides it. The flip makes true convolution commutative and associative, which matters in signal-processing theory. For a CNN it makes no difference: the kernel is learned, so a cross-correlation network just learns the flipped kernel. PyTorch's `nn.Conv2d`, TensorFlow's `Conv2D` and cuDNN all compute cross-correlation. Symmetric kernels (blur, Laplacian) give the same result either way; antisymmetric ones (edge detectors) change sign.",
      },
      { t: "h", text: "Stride and padding" },
      {
        t: "list",
        items: [
          "**Stride s** is the step between window positions. s = 1 visits every position; s = 2 visits every other one and roughly halves height and width — a cheap, learnable way to downsample.",
          "**Valid padding** (p = 0): no padding, so the kernel only visits positions where it fits fully. A 3×3 kernel shrinks each side by 2. After many layers the image would shrink away, and border pixels are covered by fewer windows than central ones.",
          "**Same padding**: pick p so that, at stride 1, the output is the same size as the input. For an odd kernel size, p = (k − 1)/2 — p = 1 for 3×3, p = 2 for 5×5. Most modern CNNs use same-padded 3×3 convolutions and downsample only with stride or pooling.",
          "Padding values are usually zeros (**zero padding**). Reflect or replicate padding copy border pixels outward instead and avoid the artificial dark frame zeros create.",
        ],
      },
      { t: "h", text: "Multi-channel inputs and multiple kernels" },
      {
        t: "p",
        text: "An RGB image has three channels, and a hidden layer may have 64 or 512. A kernel then has the **same depth as the input**: for C_in input channels its shape is C_in×k×k. At each position it multiplies all C_in·k·k window values by its weights and sums them into **one** number (plus one bias). So one kernel always makes one 2-D feature map, whatever the input depth. A layer has C_out kernels, giving C_out feature maps stacked into an output of shape C_out×H_out×W_out. The full weight tensor is **C_out × C_in × k × k** (PyTorch order): C_out·C_in·k² weights plus C_out biases.",
      },
      { t: "h", text: "Parameter sharing and translation equivariance" },
      {
        t: "p",
        text: "**Parameter sharing** means the same kernel weights are used at every position. That is why the parameter count does not depend on image size, and it acts as a strong regulariser: a detector learned from an edge in one place works everywhere at once. It also gives **translation equivariance**: shift the input by some pixels and the feature map shifts by the same amount (away from borders, and for shifts that are multiples of the stride). Equivariance is not invariance — the output moves with the input instead of staying the same. Some **invariance** (the same final answer wherever the object is) comes later, from pooling and global averaging at the end of the network.",
      },
      { t: "h", text: "Receptive field" },
      {
        t: "p",
        text: "The **receptive field** of an activation is the region of the original input that can affect it. One 3×3 convolution sees 3×3 pixels. A second 3×3 convolution on top sees 3×3 positions of the first map, each of which saw 3×3 pixels, so it sees 5×5 pixels; a third sees 7×7. Stride and pooling make it grow faster. That is how a CNN with only small kernels gets neurons that respond to whole objects: early layers detect edges, middle layers textures and parts, late layers object-level patterns.",
      },
      { t: "h", text: "Pooling" },
      {
        t: "p",
        text: "A **pooling** layer shrinks each feature map separately with a fixed (not learned) summary of each window. **Max pooling** with a 2×2 window and stride 2 keeps the largest value in each non-overlapping 2×2 block, halving height and width while keeping the strongest response; **average pooling** keeps the mean. Pooling has no parameters, saves computation in later layers, enlarges the receptive field, and tolerates small shifts (a feature that moves one pixel within a block gives the same max). **Global average pooling** averages each whole feature map to one number and ends most modern CNNs.",
      },
      { t: "h", text: "Putting it together: a small CNN" },
      {
        t: "table",
        head: ["Layer", "Output shape (C×H×W)", "Parameters", "What it does"],
        rows: [
          ["Input (8×8 grayscale digit)", "1×8×8", "0", "Pixel intensities in [0, 1]"],
          ["Conv 3×3, 16 kernels, p = 1", "16×8×8", "16·1·9 + 16 = 160", "16 local pattern detectors (edges, strokes)"],
          ["ReLU", "16×8×8", "0", "Keep positive evidence only"],
          ["Max-pool 2×2, s = 2", "16×4×4", "0", "Downsample, tolerate small shifts"],
          ["Conv 3×3, 32 kernels, p = 1", "32×4×4", "32·16·9 + 32 = 4,640", "Combine all 16 maps into 32 higher-level features"],
          ["ReLU", "32×4×4", "0", ""],
          ["Max-pool 2×2, s = 2", "32×2×2", "0", "Each value now sees a 10×10 receptive field (clipped by the 8×8 image)"],
          ["Flatten + Linear 128 → 10", "10", "128·10 + 10 = 1,290", "Class scores (logits)"],
        ],
      },
      {
        t: "p",
        text: "This is the network trained in the PyTorch tab below: 6,090 parameters in total, about 98% test accuracy on scikit-learn's digits after ten epochs on a CPU.",
      },
      { t: "h", text: "Hand-designed kernels vs learned kernels" },
      {
        t: "p",
        text: "Before deep learning, kernels were designed by hand. **Sobel** and **Prewitt** kernels approximate horizontal and vertical derivatives (edge detection). **Box** and **Gaussian** kernels average a neighbourhood (blur, denoise). The **Laplacian** approximates the second derivative (edges and spots), and **sharpen** adds Laplacian detail back to the image. A CNN starts from random kernels and learns them by gradient descent. Trained first-layer kernels often look like oriented edge and colour-blob detectors — close to the hand-designed ones — while deeper layers learn detectors nobody would design.",
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "Definitions" },
      {
        t: "p",
        text: String.raw`Let $X$ be a single-channel $H \times W$ input, $K$ a $k \times k$ kernel, $b$ a scalar bias, $s \ge 1$ the stride and $p \ge 0$ the padding. $X_p$ is $X$ surrounded by $p$ rings of zeros: $X_p[u,v] = X[u-p, v-p]$ inside the image and $0$ outside. Indices start at 0. The deep-learning “convolution” (cross-correlation) gives the feature map $Y$:`,
      },
      {
        t: "math",
        tex: String.raw`Y[i,j] \;=\; b + \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_p[\,s\,i+m,\; s\,j+n\,]\;K[m,n], \qquad 0 \le i < H_{\text{out}},\; 0 \le j < W_{\text{out}}`,
        caption: "Cross-correlation: the window starting at (s·i, s·j) is multiplied element-wise by K and summed.",
      },
      {
        t: "p",
        text: String.raw`True (signal-processing) convolution flips the kernel. Indexing the kernel from its centre, $(X * K)[i,j] = \sum_{m}\sum_{n} X[i-m,\, j-n]\,K[m,n]$; equivalently, it is cross-correlation with the kernel rotated 180°, $K^{\text{rot}}[m,n] = K[k-1-m,\; k-1-n]$:`,
      },
      {
        t: "math",
        tex: String.raw`X * K \;=\; X \star K^{\text{rot}}, \qquad K^{\text{rot}}[m,n] = K[k-1-m,\,k-1-n]`,
      },
      { t: "h", text: "Output size" },
      {
        t: "p",
        text: String.raw`The padded input has $H + 2p$ rows. The window's top row can start at rows $0, s, 2s, \dots$ as long as its last row $s\,i + k - 1$ stays inside, i.e. $s\,i \le H + 2p - k$. So the number of valid $i$ is`,
      },
      {
        t: "math",
        tex: String.raw`H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1, \qquad W_{\text{out}} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1`,
        caption: "The floor appears because a partial final step is dropped: leftover rows/columns are never visited.",
      },
      {
        t: "table",
        head: ["n", "k", "p", "s", "Computation", "Output"],
        rows: [
          ["5", "3", "0", "1", "⌊(5 + 0 − 3)/1⌋ + 1", "3 (valid)"],
          ["5", "3", "1", "1", "⌊(5 + 2 − 3)/1⌋ + 1", "5 (same)"],
          ["5", "3", "1", "2", "⌊(5 + 2 − 3)/2⌋ + 1", "3"],
          ["7", "3", "0", "2", "⌊(7 − 3)/2⌋ + 1", "3"],
          ["6", "3", "0", "2", "⌊(6 − 3)/2⌋ + 1 = ⌊1.5⌋ + 1", "2 (last column never visited)"],
          ["32", "5", "0", "1", "⌊(32 − 5)/1⌋ + 1", "28 (LeNet-5's first layer)"],
          ["224", "7", "3", "2", "⌊(224 + 6 − 7)/2⌋ + 1 = ⌊111.5⌋ + 1", "112 (ResNet's first layer)"],
          ["64", "3", "1", "2", "⌊(64 + 2 − 3)/2⌋ + 1 = ⌊31.5⌋ + 1", "32"],
        ],
      },
      {
        t: "p",
        text: String.raw`Setting $H_{\text{out}} = H$ with $s = 1$ gives $H + 2p - k + 1 = H$, so “same” padding needs $p = (k-1)/2$ — a whole number only for odd $k$, one reason odd kernel sizes are standard. Pooling uses the same formula (usually $p = 0$): a 2×2 max-pool with stride 2 maps $n \mapsto \lfloor (n-2)/2 \rfloor + 1 = \lfloor n/2 \rfloor$.`,
      },
      { t: "h", text: "Worked example: 5×5 input, 3×3 kernel, every multiply-add" },
      {
        t: "math",
        tex: String.raw`X = \begin{bmatrix}1&2&0&1&3\\0&1&3&2&1\\2&1&0&1&0\\1&3&1&0&2\\0&2&1&3&1\end{bmatrix},\qquad K = \begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix},\qquad b = 0,\; s = 1,\; p = 0`,
      },
      {
        t: "p",
        text: String.raw`The output size is $\lfloor (5 + 0 - 3)/1 \rfloor + 1 = 3$, so $Y$ is 3×3: 9 positions × 9 products = 81 multiplications. Products are listed row by row through the window (top row first), matching the kernel entries $1, 0, -1, 1, 0, -1, 1, 0, -1$. A handy check: with this kernel, each output equals (sum of the window's left column) − (sum of its right column).`,
      },
      {
        t: "table",
        head: ["Output", "Window position", "Window (row by row)", "Nine products", "Sum"],
        rows: [
          ["$Y_{00}$", "rows 0–2, cols 0–2", "1 2 0 / 0 1 3 / 2 1 0", "1 + 0 + 0 + 0 + 0 − 3 + 2 + 0 + 0", "**0**"],
          ["$Y_{01}$", "rows 0–2, cols 1–3", "2 0 1 / 1 3 2 / 1 0 1", "2 + 0 − 1 + 1 + 0 − 2 + 1 + 0 − 1", "**0**"],
          ["$Y_{02}$", "rows 0–2, cols 2–4", "0 1 3 / 3 2 1 / 0 1 0", "0 + 0 − 3 + 3 + 0 − 1 + 0 + 0 + 0", "**−1**"],
          ["$Y_{10}$", "rows 1–3, cols 0–2", "0 1 3 / 2 1 0 / 1 3 1", "0 + 0 − 3 + 2 + 0 + 0 + 1 + 0 − 1", "**−1**"],
          ["$Y_{11}$", "rows 1–3, cols 1–3", "1 3 2 / 1 0 1 / 3 1 0", "1 + 0 − 2 + 1 + 0 − 1 + 3 + 0 + 0", "**2**"],
          ["$Y_{12}$", "rows 1–3, cols 2–4", "3 2 1 / 0 1 0 / 1 0 2", "3 + 0 − 1 + 0 + 0 + 0 + 1 + 0 − 2", "**1**"],
          ["$Y_{20}$", "rows 2–4, cols 0–2", "2 1 0 / 1 3 1 / 0 2 1", "2 + 0 + 0 + 1 + 0 − 1 + 0 + 0 − 1", "**1**"],
          ["$Y_{21}$", "rows 2–4, cols 1–3", "1 0 1 / 3 1 0 / 2 1 3", "1 + 0 − 1 + 3 + 0 + 0 + 2 + 0 − 3", "**2**"],
          ["$Y_{22}$", "rows 2–4, cols 2–4", "0 1 0 / 1 0 2 / 1 3 1", "0 + 0 + 0 + 1 + 0 − 2 + 1 + 0 − 1", "**−1**"],
        ],
      },
      {
        t: "math",
        tex: String.raw`Y = X \star K = \begin{bmatrix}0&0&-1\\-1&2&1\\1&2&-1\end{bmatrix} \qquad\text{and}\qquad X * K = X \star K^{\text{rot}} = \begin{bmatrix}0&0&1\\1&-2&-1\\-1&-2&1\end{bmatrix}`,
        caption: "Cross-correlation (what PyTorch computes) vs true convolution. This kernel is antisymmetric left–right, so flipping it just negates the output.",
      },
      {
        t: "p",
        text: String.raw`After **ReLU** the negative entries become 0: $\mathrm{ReLU}(Y) = \begin{bmatrix}0&0&0\\0&2&1\\1&2&0\end{bmatrix}$. A bias $b = 1$ would add 1 to every entry of $Y$ before the ReLU.`,
      },
      { t: "h", text: "Same padding and stride on the same example" },
      {
        t: "p",
        text: String.raw`With $p = 1$ the input becomes 7×7 (a ring of zeros) and the output 5×5. The corner output, for instance, sees the padded window $\begin{bmatrix}0&0&0\\0&1&2\\0&0&1\end{bmatrix}$, giving $0 + 0 + 0 + 0 + 0 - 2 + 0 + 0 - 1 = -3$. With $p = 1, s = 2$ the window visits only rows and columns 0, 2, 4 of the padded input — every other entry of the same-padded result:`,
      },
      {
        t: "math",
        tex: String.raw`p=1,s=1:\ \begin{bmatrix}-3&-2&0&-1&3\\-4&0&0&-1&4\\-5&-1&2&1&3\\-6&1&2&-1&4\\-5&-1&2&-1&3\end{bmatrix}\qquad p=1,s=2:\ \begin{bmatrix}-3&0&3\\-5&2&3\\-5&2&3\end{bmatrix}`,
        caption: "Strided convolution = full convolution, then keep every s-th row and column.",
      },
      {
        t: "p",
        text: String.raw`ReLU on the 5×5 map, then 2×2 max-pooling with stride 2, gives a $\lfloor 5/2 \rfloor = 2$-wide output; the last row and column are dropped. The four 2×2 blocks of the ReLU map have maxima $\begin{bmatrix}0&0\\1&2\end{bmatrix}$ (2×2 average pooling would give $\begin{bmatrix}0&0\\0.25&1.25\end{bmatrix}$).`,
      },
      { t: "h", text: "Multi-channel convolution" },
      {
        t: "p",
        text: String.raw`For an input with $C_{\text{in}}$ channels $X[c, u, v]$ and a layer with $C_{\text{out}}$ kernels, the weights are $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ and the bias is $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$. Output channel $o$ is`,
      },
      {
        t: "math",
        tex: String.raw`Y[o,i,j] = b_o + \sum_{c=0}^{C_{\text{in}}-1}\;\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_p[c,\; s\,i+m,\; s\,j+n]\; W[o,c,m,n]`,
        caption: "Each output channel sums correlations over ALL input channels, then adds one bias.",
      },
      {
        t: "math",
        tex: String.raw`\#\text{params} = C_{\text{out}}\,C_{\text{in}}\,k^2 + C_{\text{out}}, \qquad \#\text{multiply-adds} = C_{\text{out}}\,C_{\text{in}}\,k^2\,H_{\text{out}}\,W_{\text{out}}`,
      },
      {
        t: "p",
        text: String.raw`Examples: 3 → 64 channels with 3×3 kernels: $64\cdot3\cdot9 + 64 = 1{,}792$. 64 → 128 with 3×3: $128\cdot64\cdot9 + 128 = 73{,}856$. 256 → 256 with 3×3: $590{,}080$. The multiply-add count depends on the spatial size even though the parameter count does not.`,
      },
      {
        t: "p",
        text: String.raw`**Worked multi-channel example.** Two 3×3 input channels, one 2×2×2 kernel, bias $b = 1$, $s = 1$, $p = 0$, so the output is $\lfloor(3-2)/1\rfloor + 1 = 2$ wide:`,
      },
      {
        t: "math",
        tex: String.raw`X_0 = \begin{bmatrix}1&0&2\\2&1&0\\0&1&3\end{bmatrix},\; X_1 = \begin{bmatrix}0&1&1\\1&0&2\\2&1&0\end{bmatrix},\quad W_0 = \begin{bmatrix}1&-1\\0&2\end{bmatrix},\; W_1 = \begin{bmatrix}2&0\\-1&1\end{bmatrix}`,
      },
      {
        t: "steps",
        items: [
          { title: "Position (0,0), channel 0", tex: String.raw`1\cdot1 + 0\cdot(-1) + 2\cdot0 + 1\cdot2 = 3` },
          { title: "Position (0,0), channel 1", tex: String.raw`0\cdot2 + 1\cdot0 + 1\cdot(-1) + 0\cdot1 = -1` },
          { title: "Combine with bias", tex: String.raw`Y[0,0] = 3 + (-1) + 1 = 3` },
          { title: "Remaining positions", text: "(0,1): channel sums −2 and 4 → −2 + 4 + 1 = 3.  (1,0): 3 and 1 → 5.  (1,1): 7 and −1 → 7.", tex: String.raw`Y = \begin{bmatrix}3&3\\5&7\end{bmatrix}` },
        ],
      },
      { t: "h", text: "Translation equivariance" },
      {
        t: "p",
        text: String.raw`Let $T_\delta$ shift an image by $\delta = (\delta_1, \delta_2)$ pixels: $(T_\delta X)[u,v] = X[u-\delta_1, v-\delta_2]$. For stride 1, ignoring borders, plugging into the definition gives $(T_\delta X \star K)[i,j] = \sum_{m,n} X[i+m-\delta_1,\, j+n-\delta_2]K[m,n] = (X\star K)[i-\delta_1, j-\delta_2]$, i.e.`,
      },
      {
        t: "math",
        tex: String.raw`(T_\delta X) \star K \;=\; T_\delta\,(X \star K)`,
        caption: "Shifting then convolving equals convolving then shifting. With stride s this holds for shifts that are multiples of s.",
      },
      { t: "h", text: "Receptive field recursion" },
      {
        t: "p",
        text: String.raw`Let $r_\ell$ be the receptive-field width of layer $\ell$ in input pixels, and $j_\ell$ the **jump** — the distance in input pixels between neighbouring positions of layer $\ell$. Starting from $r_0 = 1, j_0 = 1$, a layer with kernel $k_\ell$ and stride $s_\ell$ gives`,
      },
      {
        t: "math",
        tex: String.raw`r_\ell = r_{\ell-1} + (k_\ell - 1)\,j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\,s_\ell`,
      },
      {
        t: "table",
        head: ["Layer", "k", "s", "r (receptive field)", "j (jump)"],
        rows: [
          ["conv 3×3", "3", "1", "1 + 2·1 = 3", "1"],
          ["conv 3×3", "3", "1", "3 + 2·1 = 5", "1"],
          ["max-pool 2×2", "2", "2", "5 + 1·1 = 6", "2"],
          ["conv 3×3", "3", "1", "6 + 2·2 = 10", "2"],
          ["max-pool 2×2", "2", "2", "10 + 1·2 = 12", "4"],
          ["conv 3×3", "3", "1", "12 + 2·4 = 20", "4"],
        ],
      },
      {
        t: "p",
        text: String.raw`Three stacked 3×3 convolutions (stride 1) see 7×7, like one 7×7 convolution, but with $3\cdot 9 C^2 = 27C^2$ weights instead of $49C^2$ (for $C$ channels in and out) and three non-linearities instead of one. This is VGG's design principle (Simonyan and Zisserman, 2014).`,
      },
      { t: "h", text: "The backward pass" },
      {
        t: "p",
        text: String.raw`Take stride 1, no padding, one channel, and let $G[i,j] = \partial L / \partial Y[i,j]$ be the upstream gradient. Since $Y[i,j] = b + \sum_{m,n} X[i+m, j+n]K[m,n]$, each weight $K[m,n]$ appears in every output, times the input pixel at offset $(m,n)$. By the chain rule:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial L}{\partial K[m,n]} = \sum_{i,j} G[i,j]\,X[i+m,\,j+n] = (X \star G)[m,n], \qquad \frac{\partial L}{\partial b} = \sum_{i,j} G[i,j]`,
        caption: "The kernel gradient is a cross-correlation of the input with the upstream gradient (same shape as K).",
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial L}{\partial X[u,v]} = \sum_{m,n} G[u-m,\,v-n]\,K[m,n] = \big(\mathrm{pad}_{k-1}(G) \star K^{\text{rot}}\big)[u,v]`,
        caption: "The input gradient is a “full” convolution: zero-pad G by k − 1 and correlate with the 180°-rotated kernel (G is 0 outside its range).",
      },
      {
        t: "p",
        text: "So a convolution's backward pass is made of convolutions too, which is why the same optimised kernels (e.g. in cuDNN) serve forward and backward. The weight gradient adds up contributions from every position — parameter sharing seen from the gradient side. The NumPy code below checks both formulas against PyTorch autograd. For stride s > 1, the upstream gradient is first “dilated” by inserting s − 1 zeros between its entries.",
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "convolution", caption: "The kernel window (gold) slides over the padded input; the nine products, their sum and the bias are shown for every output cell." },
      {
        t: "list",
        items: [
          "The simulator opens on the lesson's 5×5 example with the [1 0 −1] kernel. Press **Step** nine times and check each output against the worked table in the math section.",
          "Tick **Flip kernel (true convolution)**: every output changes sign, as in the math section. Switch to the Gaussian blur and flip again — nothing changes, because the kernel is symmetric.",
          "Set **Padding p = 1**. Dashed cells are padding zeros; the output becomes 5×5 and the corner cell −3. The narration counts how many of the nine window cells are padding.",
          "Set **Stride s = 2** and watch the window jump two cells. Then choose 6×6 with p = 0 and s = 2: the formula gives ⌊3/2⌋ + 1 = 2, and the narration points out the column that is never visited.",
          "Choose the **Vertical edge** input with **Sobel — vertical edges**: the feature map lights up only on the edge columns. Switch to **Sobel — horizontal edges**: the map goes to zero. Then try the **Brightness ramp** with the **Laplacian**: zero inside, because a straight ramp has no curvature.",
          "Click input cells to change them (+1; right-click −1) and watch the output update instantly. Paint a dot on a blank grid and apply the identity, blur and sharpen kernels in turn.",
          "Enable **ReLU** and **2×2 max-pool**. After the convolution steps, the pool stage steps through the feature map, showing max(a, b, c, d) for each block.",
        ],
      },
      { t: "sim", id: "cnn-pipeline", caption: "A two-layer CNN with hand-designed kernels applied to a real image: convolution → ReLU → max-pool → convolution → ReLU → max-pool." },
      {
        t: "list",
        items: [
          "Step through the seven stages and read each shape label: 1×64×64 → 6×62×62 → 6×62×62 → 6×31×31 → 4×31×31 → 4×31×31 → 4×15×15 with default settings.",
          "Set **Layer-1 padding** to 1: conv-1 output becomes 64×64 and every later shape changes. Set **stride** to 2: conv 1 jumps straight to 31×31 (or 32×32 with padding), doing the same downsampling with a quarter of the multiply-adds.",
          "Look at stage 1 on the **Shapes** image: Sobel-x is teal on one side of each shape and clay on the other. After ReLU (stage 2) only one side survives — ReLU drops one edge polarity, which is why learned CNNs often contain kernel pairs of opposite sign.",
          "On **Rings & stripes**, compare the two diagonal kernels on the striped half, then upload your own photo. The kernels are hand-designed, not learned. The point: real feature extraction is just the first simulator's arithmetic, repeated over every pixel and channel.",
        ],
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "A full NumPy multi-channel 2-D convolution with stride, padding and bias, plus max pooling. It reproduces the worked example, checks itself against `torch.nn.functional.conv2d` and `max_pool2d` on random data for several stride/padding settings, and checks the backward-pass formulas against PyTorch autograd. Plain loops keep it readable; the coding exercise has a vectorised (im2col) version.",
      },
      {
        t: "code",
        lang: "python",
        title: "conv2d_from_scratch.py",
        code: String.raw`import numpy as np
import torch                       # used only to check our results
import torch.nn.functional as F


def output_size(n, k, p, s):
    """Spatial output size of a convolution / pooling layer: floor((n + 2p - k) / s) + 1."""
    return (n + 2 * p - k) // s + 1


def conv2d(x, w, b=None, stride=1, padding=0):
    """Deep-learning 'convolution' (cross-correlation), written with explicit loops.

    x: input,   shape (N, C_in, H, W)
    w: kernels, shape (C_out, C_in, k, k)
    b: biases,  shape (C_out,) or None
    returns:    shape (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, C_in_w, k, _ = w.shape
    assert C_in == C_in_w, "kernel depth must equal number of input channels"
    # zero-pad only the two spatial axes
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = output_size(H, k, padding, stride)
    W_out = output_size(W, k, padding, stride)
    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    r, c = i * stride, j * stride           # top-left corner of the window
                    window = xp[n, :, r:r + k, c:c + k]     # (C_in, k, k)
                    out[n, co, i, j] = np.sum(window * w[co])  # multiply-accumulate over C_in*k*k
            if b is not None:
                out[n, co] += b[co]
    return out


def maxpool2d(x, size=2, stride=2):
    """Max pooling over non-overlapping (by default) size x size windows, per channel."""
    N, C, H, W = x.shape
    H_out, W_out = output_size(H, size, 0, stride), output_size(W, size, 0, stride)
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            r, c = i * stride, j * stride
            out[:, :, i, j] = x[:, :, r:r + size, c:c + size].max(axis=(2, 3))
    return out


# ---- 1) The worked example from the lesson: 5x5 input, 3x3 vertical-edge kernel ----
X = np.array([[1, 2, 0, 1, 3],
              [0, 1, 3, 2, 1],
              [2, 1, 0, 1, 0],
              [1, 3, 1, 0, 2],
              [0, 2, 1, 3, 1]], dtype=float)
K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]], dtype=float)
x1, w1 = X[None, None], K[None, None]      # add batch and channel axes -> (1,1,5,5), (1,1,3,3)
print("valid (p=0, s=1):\n", conv2d(x1, w1)[0, 0])
print("same  (p=1, s=1):\n", conv2d(x1, w1, padding=1)[0, 0])
print("p=1, s=2:\n", conv2d(x1, w1, padding=1, stride=2)[0, 0])
same = conv2d(x1, w1, padding=1)
print("ReLU -> 2x2 max-pool:\n", maxpool2d(np.maximum(same, 0))[0, 0])

# ---- 2) Random multi-channel test against PyTorch ----
rng = np.random.default_rng(0)
N, C_in, H, W, C_out, k = 2, 3, 9, 9, 4, 3
x = rng.standard_normal((N, C_in, H, W))
w = rng.standard_normal((C_out, C_in, k, k))
b = rng.standard_normal(C_out)

xt, wt, bt = (torch.tensor(a) for a in (x, w, b))
for stride, padding in [(1, 0), (1, 1), (2, 1), (3, 2)]:
    ours = conv2d(x, w, b, stride=stride, padding=padding)
    ref = F.conv2d(xt, wt, bt, stride=stride, padding=padding).numpy()
    print(f"stride={stride} padding={padding}: shape {ours.shape}, "
          f"max |ours - torch| = {np.abs(ours - ref).max():.2e}")
mp = maxpool2d(x)
print("maxpool shape", mp.shape, "max diff", np.abs(mp - F.max_pool2d(xt, 2).numpy()).max())
print("parameters in this layer:", w.size + b.size, "= C_out*C_in*k*k + C_out =",
      C_out * C_in * k * k + C_out)

# ---- 3) Backward pass: dL/dW is a correlation of the input with the upstream gradient ----
# Single image, single channel, stride 1, no padding: Y = X (corr) K, take L = sum(G * Y)
G = rng.standard_normal((3, 3))                       # upstream gradient dL/dY
dK = conv2d(x1, G[None, None])[0, 0]                  # correlate X with G -> 3x3
# dL/dX is a "full" convolution: pad G by k-1 and correlate with the 180-degree-rotated kernel
dX = conv2d(np.pad(G, 2)[None, None], K[::-1, ::-1][None, None].copy())[0, 0]

Xt = torch.tensor(X[None, None], requires_grad=True)
Kt = torch.tensor(K[None, None], requires_grad=True)
(F.conv2d(Xt, Kt) * torch.tensor(G)).sum().backward()
print("dL/dK matches autograd:", np.allclose(dK, Kt.grad.numpy()[0, 0]))
print("dL/dX matches autograd:", np.allclose(dX, Xt.grad.numpy()[0, 0]))`,
        output: String.raw`valid (p=0, s=1):
 [[ 0.  0. -1.]
 [-1.  2.  1.]
 [ 1.  2. -1.]]
same  (p=1, s=1):
 [[-3. -2.  0. -1.  3.]
 [-4.  0.  0. -1.  4.]
 [-5. -1.  2.  1.  3.]
 [-6.  1.  2. -1.  4.]
 [-5. -1.  2. -1.  3.]]
p=1, s=2:
 [[-3.  0.  3.]
 [-5.  2.  3.]
 [-5.  2.  3.]]
ReLU -> 2x2 max-pool:
 [[0. 0.]
 [1. 2.]]
stride=1 padding=0: shape (2, 4, 7, 7), max |ours - torch| = 3.55e-15
stride=1 padding=1: shape (2, 4, 9, 9), max |ours - torch| = 3.55e-15
stride=2 padding=1: shape (2, 4, 5, 5), max |ours - torch| = 3.55e-15
stride=3 padding=2: shape (2, 4, 4, 4), max |ours - torch| = 3.55e-15
maxpool shape (2, 3, 4, 4) max diff 0.0
parameters in this layer: 112 = C_out*C_in*k*k + C_out = 112
dL/dK matches autograd: True
dL/dX matches autograd: True`,
      },
      {
        t: "list",
        items: [
          "Differences of about 3.6e-15 are round-off from summing in a different order, not errors.",
          "Only the spatial axes are padded; padding batch or channel axes would be a bug.",
          "`K[::-1, ::-1]` is the kernel rotated 180° (a reversed view; `.copy()` makes it a normal contiguous array).",
          "The four nested loops show the cost: N·C_out·H_out·W_out windows, each with C_in·k·k multiply-adds.",
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
                text: "A small CNN trained on scikit-learn's `load_digits`: 1,797 grayscale 8×8 handwritten digits, about ten seconds on a CPU. The script prints each parameter tensor's shape (note the C_out×C_in×k×k layout), traces activation shapes through the network, and reports test accuracy after each epoch.",
              },
              {
                t: "code",
                lang: "python",
                title: "cnn_digits.py",
                code: String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# 1797 grayscale 8x8 handwritten digits, pixel values 0..16
digits = load_digits()
X = digits.images.astype(np.float32) / 16.0          # scale to [0, 1]
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
# PyTorch expects NCHW: (batch, channels, height, width)
X_tr = torch.tensor(X_tr).unsqueeze(1)
X_te = torch.tensor(X_te).unsqueeze(1)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)
print("train", tuple(X_tr.shape), "test", tuple(X_te.shape))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 1x8x8  -> 16x8x8  ("same")
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x8x8 -> 16x4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16x4x4 -> 32x4x4
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x4x4 -> 32x2x2
        )
        self.classifier = nn.Linear(32 * 2 * 2, 10)       # 128 features -> 10 class scores

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


model = SmallCNN()
for name, p in model.named_parameters():
    print(f"{name:22s} {str(tuple(p.shape)):16s} {p.numel():5d}")
print("total parameters:", sum(p.numel() for p in model.parameters()))

# trace the shapes through the network for one image
h = X_tr[:1]
for layer in model.features:
    h = layer(h)
    print(f"{layer.__class__.__name__:10s} -> {tuple(h.shape)}")

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
batch = 64
for epoch in range(1, 11):
    model.train()
    perm = torch.randperm(len(X_tr))
    total = 0.0
    for i in range(0, len(X_tr), batch):
        idx = perm[i:i + batch]
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    print(f"epoch {epoch:2d}  train loss {total / len(X_tr):.4f}  test accuracy {acc:.4f}")`,
                output: String.raw`train (1347, 1, 8, 8) test (450, 1, 8, 8)
features.0.weight      (16, 1, 3, 3)      144
features.0.bias        (16,)               16
features.3.weight      (32, 16, 3, 3)    4608
features.3.bias        (32,)               32
classifier.weight      (10, 128)         1280
classifier.bias        (10,)               10
total parameters: 6090
Conv2d     -> (1, 16, 8, 8)
ReLU       -> (1, 16, 8, 8)
MaxPool2d  -> (1, 16, 4, 4)
Conv2d     -> (1, 32, 4, 4)
ReLU       -> (1, 32, 4, 4)
MaxPool2d  -> (1, 32, 2, 2)
epoch  1  train loss 1.6963  test accuracy 0.8200
epoch  2  train loss 0.3804  test accuracy 0.9000
epoch  3  train loss 0.1817  test accuracy 0.9156
epoch  4  train loss 0.1845  test accuracy 0.9600
epoch  5  train loss 0.1218  test accuracy 0.9511
epoch  6  train loss 0.0760  test accuracy 0.9533
epoch  7  train loss 0.1142  test accuracy 0.9556
epoch  8  train loss 0.0464  test accuracy 0.9778
epoch  9  train loss 0.0280  test accuracy 0.9711
epoch 10  train loss 0.0214  test accuracy 0.9822`,
              },
              {
                t: "list",
                items: [
                  "`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` — the weight has shape (out, in, k, k): 16·1·9 = 144 and 32·16·9 = 4,608, matching C_out·C_in·k².",
                  "`padding=1` with a 3×3 kernel keeps 8×8 → 8×8; recent PyTorch versions also accept `padding=\"same\"` for stride 1.",
                  "98.2% test accuracy after 10 epochs, from 6,090 parameters. Exact numbers vary slightly across PyTorch versions and hardware, even with a fixed seed.",
                  "`F.conv2d(x, weight, bias, stride, padding)` is the functional form used in the NumPy check; `nn.Conv2d` just owns the weight and bias as parameters.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "The same architecture in Keras. Keras is channels-last by default (N×H×W×C), so the digits get a trailing channel axis, and `Conv2D` stores its kernel as (k, k, C_in, C_out). `padding=\"same\"` and `padding=\"valid\"` replace explicit padding amounts.",
              },
              {
                t: "code",
                lang: "python",
                title: "cnn_digits_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(0)

digits = load_digits()
X = (digits.images.astype("float32") / 16.0)[..., None]   # (1797, 8, 8, 1)  NHWC
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),  # 8x8x16
    tf.keras.layers.MaxPooling2D(pool_size=2),                                     # 4x4x16
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),  # 4x4x32
    tf.keras.layers.MaxPooling2D(pool_size=2),                                     # 2x2x32
    tf.keras.layers.Flatten(),                                                     # 128
    tf.keras.layers.Dense(10),                                                     # logits
])
model.summary()
print("first conv kernel shape:", model.layers[0].kernel.shape)   # (3, 3, 1, 16)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(X_tr, y_tr, batch_size=64, epochs=10, validation_data=(X_te, y_te), verbose=2)
print("test accuracy:", model.evaluate(X_te, y_te, verbose=0)[1])`,
                output: "Prints the model summary (6,090 trainable parameters, the same as the PyTorch model), the first kernel's shape (3, 3, 1, 16), one line per epoch with training loss/accuracy and validation loss/accuracy, and finally the test accuracy, which is typically in the high 0.9s.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      {
        t: "p",
        text: "The basic operation has many variants, most covered in [Dilated, Transposed, Depthwise and Pointwise Convolutions](/learn/m08/conv-variants). Here is the map.",
      },
      {
        t: "table",
        head: ["Variant", "What changes", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Standard 2-D convolution", "C_out kernels, each C_in×k×k", "Local, shared, equivariant; very fast libraries", "Cost grows with C_in·C_out·k²", "Almost every CNN layer"],
          ["Strided convolution", "s > 1", "Shrinks the map in a learned way; less compute", "Can skip details (aliasing); shift-matching only for shifts that are multiples of s", "Instead of pooling (ResNet stems, many modern nets)"],
          ["1×1 (pointwise) convolution", "k = 1", "Cheaply mixes channels; changes the channel count", "Sees no neighbours", "Bottlenecks, Inception, MobileNet"],
          ["Depthwise convolution", "One k×k kernel per input channel; channels never mix", "About C_out times fewer weights and FLOPs", "Needs a 1×1 conv afterwards to mix channels", "MobileNet, Xception, ConvNeXt"],
          ["Dilated (atrous) convolution", "Gaps of d − 1 between kernel taps", "Sees a wider area with no extra weights and no shrinking", "Grid-shaped blind spots if stacked carelessly", "Segmentation (DeepLab), audio (WaveNet)"],
          ["Transposed convolution", "Spreads each input value over a k×k output patch (the backward pass of a strided conv)", "Learned upsampling", "Checkerboard patterns", "Decoders, GAN generators, U-Net"],
          ["1-D / 3-D convolution", "Kernel slides along 1 or 3 axes", "Same benefits for sequences or 3-D volumes", "3-D is expensive", "Audio, text, ECG (1-D); video, CT/MRI (3-D)"],
          ["Grouped convolution", "Channels split into g groups, each convolved on its own", "g times fewer weights", "Channels mix less", "AlexNet (2 GPUs), ResNeXt"],
        ],
      },
      { t: "h", text: "Pooling variants" },
      {
        t: "table",
        head: ["Pooling", "Operation", "Notes"],
        rows: [
          ["Max pooling", "Largest value in each window", "Keeps the strongest response; most common in classic CNNs"],
          ["Average pooling", "Mean of each window", "Smoother; used in some older nets (LeNet used a learned, scaled version)"],
          ["Global average pooling", "Mean of each whole feature map → C numbers", "Replaces big dense layers at the end of modern CNNs; works for any input size"],
          ["Strided convolution (instead of pooling)", "Learned shrinking", "Common in recent models"],
        ],
      },
      { t: "h", text: "Classic hand-designed kernels" },
      {
        t: "table",
        head: ["Kernel", "Weights", "Sum of weights", "Effect"],
        rows: [
          ["Identity", "[0 0 0; 0 1 0; 0 0 0]", "1", "Copies the image"],
          ["Box blur", "all 1/9", "1", "Local mean; smooths noise"],
          ["Gaussian blur", "[1 2 1; 2 4 2; 1 2 1]/16", "1", "Local mean that weights the centre most; softer blur"],
          ["Sobel-x", "[−1 0 1; −2 0 2; −1 0 1]", "0", "Change left-to-right → finds vertical edges"],
          ["Sobel-y", "[−1 −2 −1; 0 0 0; 1 2 1]", "0", "Change top-to-bottom → finds horizontal edges"],
          ["Laplacian", "[0 1 0; 1 −4 1; 0 1 0]", "0", "Change of the change: edges and spots; 0 on smooth ramps"],
          ["Sharpen", "[0 −1 0; −1 5 −1; 0 −1 0]", "1", "Identity minus Laplacian: makes edges stand out"],
        ],
      },
      {
        t: "p",
        text: "Kernels whose weights sum to 0 ignore constant brightness (they measure change); kernels summing to 1 keep the brightness of flat regions. A learned first-layer kernel can be any mix of these.",
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Image classification** — LeNet-5 read handwritten digits on bank cheques in the 1990s. AlexNet, VGG, ResNet and EfficientNet are ImageNet-trained CNN classifiers, also used as general feature extractors (backbones) for other tasks.",
          "**Object detection and segmentation** — Faster R-CNN, YOLO, Mask R-CNN and U-Net are built on convolutional backbones. U-Net (Ronneberger et al., 2015) was made for biomedical segmentation and is now also the denoising network in many diffusion models.",
          "**Medical imaging** — CNNs classify and segment X-rays, CT and MRI scans, retinal photos and histopathology slides; 3-D convolutions process volumes directly.",
          "**Autonomous driving and robotics** — lane, pedestrian and vehicle detection from cameras, usually with CNN backbones tuned for embedded GPUs.",
          "**Audio and signals** — 1-D convolutions over raw waveforms (WaveNet, van den Oord et al., 2016, used dilated causal convolutions), 2-D convolutions over spectrograms for speech and sound, and 1-D CNNs for ECG and sensor time series.",
          "**On-device vision** — MobileNet-style depthwise separable convolutions run real-time camera features on phones, where parameters and multiply-adds matter.",
          "**Classical image processing** — this lesson's hand-designed kernels (blur, sharpen, edge detection) are what image editors and libraries such as OpenCV apply under the hood.",
        ],
      },
    ],
    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Parameter efficiency**: weights scale with C_in·C_out·k², not image size (1,792 vs 150 million in the intro comparison).",
          "**Built-in prior for images**: locality and weight sharing match natural images, so CNNs learn well from fairly little data.",
          "**Translation equivariance** comes free, and pooling adds some local invariance.",
          "**Any input size**: a fully convolutional network handles different resolutions.",
          "**Hardware friendly**: dense, regular multiply-adds that suit GPUs and accelerators (cuDNN, im2col + matrix multiply, Winograd, FFT-based methods).",
          "**Hierarchical features**: stacking gives edges → textures → parts → objects as receptive fields grow.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Local by design**: relating distant image parts needs many layers or downsampling; attention models capture long-range links in one layer.",
          "**Not invariant to rotation or scale**: a CNN must learn rotated and rescaled patterns from data (hence data augmentation).",
          "**Imperfect equivariance**: striding and pooling break exact shift equivariance; small input shifts can noticeably change a deep CNN's output.",
          "**Compute grows with resolution**: multiply-adds scale with H_out·W_out, so high-resolution inputs are expensive even though parameters are not.",
          "**Strong inductive bias cuts both ways**: with very large datasets, less constrained models (Vision Transformers) can match or beat CNNs; CNNs tend to win on smaller data.",
        ],
      },
      { t: "h", text: "Practical trade-offs" },
      {
        t: "table",
        head: ["Choice", "Option A", "Option B"],
        rows: [
          ["Kernel size", "Small (3×3) stacked: fewer parameters, more non-linearity", "Large (7×7): bigger receptive field in one layer, more parameters"],
          ["Downsampling", "Max-pool: no parameters, robust", "Strided conv: learnable, fewer FLOPs, can alias"],
          ["Padding", "Same: keeps sizes aligned, needed for deep stacks and skip connections", "Valid: no artificial border values, but shrinks maps"],
          ["Width (C_out)", "More channels: more features, cost ∝ C_in·C_out", "Fewer channels: faster, risk of underfitting"],
        ],
      },
    ],
    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Getting the output size wrong.** Forgetting the floor, the +1 or the 2p causes a shape mismatch at the first linear layer. Fix: apply ⌊(n + 2p − k)/s⌋ + 1 layer by layer, or pass a dummy tensor through the feature extractor and print its shape (as the PyTorch tab does).",
          "**Mixing up NCHW and NHWC.** Give PyTorch an (N, H, W, C) array and it treats width as channels. Fix: `x.permute(0, 3, 1, 2)` for PyTorch; keep Keras data channels-last. Check that the channel size equals the layer's `in_channels`.",
          "**Thinking a kernel is 2-D in a multi-channel layer.** A 3×3 kernel on a 64-channel input has 64·3·3 = 576 weights and makes one map. Counts that ignore C_in are off by a factor of C_in.",
          "**Forgetting the bias in the parameter count.** A Conv2d has C_out biases unless `bias=False` (common before batch normalisation, whose shift makes the bias redundant).",
          "**Expecting “same” padding with stride > 1 to keep the size.** Same padding keeps the size only at stride 1; at stride 2 the output is about half (⌈n/2⌉ in TensorFlow's convention).",
          "**Implementing true convolution and comparing with PyTorch.** Flipping the kernel in your own code makes results disagree with `F.conv2d` for any non-symmetric kernel. Deep-learning convolution is cross-correlation: no flip.",
          "**Padding the wrong axes in NumPy.** `np.pad(x, p)` on an (N, C, H, W) array pads batch and channel axes too. Fix: per-axis widths like `((0,0),(0,0),(p,p),(p,p))`.",
          "**Confusing equivariance with invariance.** Convolution alone does not make the network position-independent; the feature map moves with the object. Invariance comes from pooling/global pooling and training data.",
          "**Forgetting the non-linearity.** Two convolutions with no activation between equal one convolution with a bigger kernel: more parameters, no extra power.",
          "**Assuming more layers alone give a big receptive field.** At stride 1 with 3×3 kernels it grows only 2 per layer; 10 layers see just 21×21 pixels. Downsampling (stride, pooling) or dilation is needed to see whole objects.",
        ],
      },
    ],
    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What is a convolutional kernel and what does a feature map represent?", a: "A kernel (filter) is a small array of learnable weights, e.g. 3×3×C_in. It slides over the input; at each position the window is multiplied element-wise by the kernel and summed (plus a bias). The grid of these sums is the feature map: how strongly the kernel's pattern appears at each location. One kernel makes one feature map; a layer with C_out kernels makes C_out maps." },
      { t: "qa", level: "beginner", q: "Why not just use fully connected layers on images?", a: "Too many parameters and the wrong structure. A 224×224×3 image has 150,528 inputs, so a 1,000-unit dense layer needs about 150.5 million weights. It also ignores spatial locality and must re-learn each detector at every position. Convolution shares small local kernels across positions: orders of magnitude fewer parameters, any image size, and translation equivariance." },
      { t: "qa", level: "beginner", q: "What do stride and padding do? Give the output-size formula.", a: "Stride is the step between window positions (s = 2 roughly halves height and width). Padding adds p rings (usually zeros) around the input so the kernel can centre on border pixels and to control output size. Output size per axis = ⌊(n + 2p − k)/s⌋ + 1. Example: n = 32, k = 5, p = 2, s = 2 → ⌊31/2⌋ + 1 = 16." },
      { t: "qa", level: "beginner", q: "What is the difference between valid and same padding?", a: "Valid padding uses p = 0: the kernel only visits positions where it fits, so a k×k kernel shrinks each side by k − 1. Same padding picks p so the output size equals the input size at stride 1; for odd k, p = (k − 1)/2 (p = 1 for 3×3)." },
      { t: "qa", level: "intermediate", q: "How many parameters does a Conv2d(64, 128, kernel_size=3) layer have, and how many multiply-adds on a 56×56 output?", a: "Weights: 128·64·3·3 = 73,728; biases: 128; total 73,856. Each output value needs 64·9 = 576 multiply-adds, and there are 128·56·56 = 401,408 outputs, so about 231 million multiply-adds per image. Parameters do not depend on spatial size; compute does." },
      { t: "qa", level: "intermediate", q: "Is deep-learning convolution really convolution? Does it matter?", a: "No — libraries compute cross-correlation, without flipping the kernel. True convolution flips the kernel 180° first, which makes it commutative and associative. For a CNN it does not matter: the kernel is learned, so the network learns the flipped version. It matters only when importing hand-designed kernels or comparing with signal-processing code, where non-symmetric kernels give sign-flipped or mirrored results." },
      { t: "qa", level: "intermediate", q: "Explain parameter sharing and translation equivariance, and how they differ from invariance.", a: "Parameter sharing: the same kernel weights are used at every position. The result is equivariance: shift the input and the feature map shifts by the same amount, (T X)⋆K = T(X⋆K) (away from borders, for shifts that are multiples of the stride). Invariance would mean the output does not change at all. CNNs get approximate invariance from pooling and global average pooling at the end, not from convolution itself." },
      { t: "qa", level: "intermediate", q: "Why do modern CNNs prefer stacks of 3×3 convolutions over larger kernels?", a: "Two stacked 3×3 layers see 5×5 and three see 7×7, like larger kernels, but with fewer weights (27C² for three 3×3 vs 49C² for one 7×7, with C channels in and out) and a non-linearity after each layer for more expressive power. This was VGG's key design idea." },
      { t: "qa", level: "advanced", q: "Derive the gradients of a convolution layer with respect to the kernel and the input.", a: "For Y[i,j] = b + Σ_{m,n} X[i+m, j+n] K[m,n] with upstream G = ∂L/∂Y: ∂L/∂K[m,n] = Σ_{i,j} G[i,j] X[i+m, j+n], the cross-correlation X ⋆ G (shape k×k). ∂L/∂b = Σ G. ∂L/∂X[u,v] = Σ_{m,n} G[u−m, v−n] K[m,n] (G = 0 outside its range): the full convolution of G with K, done by correlating the (k − 1)-zero-padded G with the 180°-rotated kernel. With stride s, G is first dilated with s − 1 zeros between entries. Multi-channel versions sum over the right channel axes." },
      { t: "qa", level: "advanced", q: "How is convolution implemented efficiently on GPUs?", a: "The most general method is im2col (or implicit GEMM). Every receptive-field patch is unrolled into a column of length C_in·k², giving a (C_in·k²)×(H_out·W_out) matrix; the weights form a C_out×(C_in·k²) matrix, and the layer becomes one matrix multiplication, which GPUs run near peak speed. The cost is memory (each input pixel copied up to k² times); implicit-GEMM kernels avoid it by building patches on the fly. For small kernels, Winograd transforms cut the number of multiplications; for large kernels, FFT-based convolution turns the operation into element-wise products in the frequency domain. Libraries such as cuDNN pick an algorithm per layer shape." },
      { t: "qa", level: "advanced", q: "Compute the receptive field of conv3–conv3–pool2–conv3–pool2–conv3 (stride 1 convolutions) and explain the recursion.", a: "Track the receptive field r and the jump j (input pixels between neighbouring positions): r_l = r_{l−1} + (k_l − 1) j_{l−1}, j_l = j_{l−1} s_l, starting at r = 1, j = 1. conv3: r = 3, j = 1; conv3: 5, 1; pool2 (s = 2): 6, 2; conv3: 6 + 2·2 = 10, 2; pool2: 10 + 1·2 = 12, 4; conv3: 12 + 2·4 = 20, 4. So each final activation sees a 20×20 patch. That is the theoretical receptive field; the effective one (where gradients are actually large) is usually smaller and roughly Gaussian-shaped." },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            kind: "numeric",
            q: "An input of width 32 is convolved with a 5×5 kernel, padding 2, stride 2. What is the output width?",
            answer: 16,
            tolerance: 0,
            explain: "⌊(32 + 2·2 − 5)/2⌋ + 1 = ⌊31/2⌋ + 1 = 15 + 1 = 16.",
          },
          {
            kind: "numeric",
            q: "How many learnable parameters (weights + biases) does nn.Conv2d(64, 128, kernel_size=3) have?",
            answer: 73856,
            tolerance: 0,
            explain: "128·64·3·3 = 73,728 weights plus 128 biases = 73,856.",
          },
          {
            kind: "numeric",
            q: "In the lesson's 5×5 worked example (kernel [1 0 −1] in every row, no padding, stride 1), what is the centre output Y[1,1]?",
            answer: 2,
            tolerance: 0,
            explain: "Window rows 1–3, cols 1–3 = [1 3 2 / 1 0 1 / 3 1 0]. Left column sum 1 + 1 + 3 = 5, right column sum 2 + 1 + 0 = 3, so Y[1,1] = 5 − 3 = 2.",
          },
          {
            q: "A conv layer receives a 3×32×32 tensor and has 10 kernels of size 3×3. What is the shape of ONE kernel?",
            options: ["3×3", "10×3×3", "3×3×3", "10×32×32"],
            answer: 2,
            explain: "Each kernel covers all input channels: C_in×k×k = 3×3×3. The whole layer's weights are 10×3×3×3, and each kernel makes one feature map.",
          },
          {
            q: "Which statement about PyTorch's nn.Conv2d is correct?",
            options: [
              "It flips the kernel before sliding (true convolution)",
              "It computes cross-correlation; the kernel is not flipped",
              "It computes convolution in the frequency domain, so flipping doesn't matter",
              "It flips the kernel only when padding > 0",
            ],
            answer: 1,
            explain: "Deep-learning libraries compute cross-correlation. The kernels are learned, so the network just learns them flipped; it can represent exactly the same things.",
          },
          {
            kind: "numeric",
            q: "You stack three 3×3 convolutions with stride 1. How many input pixels wide is the receptive field of one output of the third layer?",
            answer: 7,
            tolerance: 0,
            explain: "r grows by k − 1 = 2 per layer at stride 1: 1 → 3 → 5 → 7.",
          },
          {
            q: "You shift an input image 4 pixels to the right and pass it through one stride-1 convolution layer. Ignoring borders, the feature map…",
            options: ["is unchanged (invariance)", "shifts 4 pixels to the right (equivariance)", "shifts 4 pixels to the left", "changes unpredictably"],
            answer: 1,
            explain: "Convolution is translation equivariant: (T X)⋆K = T(X⋆K). The output moves with the input. To get invariance (no change at all) you need pooling or a global summary.",
          },
          {
            kind: "numeric",
            q: "A 28×28 image goes through a 3×3 convolution (no padding, stride 1), then 2×2 max-pooling with stride 2. What width comes out?",
            answer: 13,
            tolerance: 0,
            explain: "Convolution: ⌊(28 − 3)/1⌋ + 1 = 26. Pooling: ⌊(26 − 2)/2⌋ + 1 = 13.",
          },
          {
            q: "The gradient of the loss with respect to a convolution kernel is…",
            options: [
              "the kernel rotated by 180°",
              "the layer's input cross-correlated with the gradient arriving from above",
              "the gradient from above summed over all positions",
              "zero, because weights are shared",
            ],
            answer: 1,
            explain: "∂L/∂K[m,n] = Σ_{i,j} G[i,j] X[i+m, j+n] = (X ⋆ G)[m,n]. Because the kernel is shared, the gradients from every position are added up; they don't cancel to zero. (The sum of G alone is the bias gradient.)",
          },
        ],
      },
      {
        t: "exercise",
        title: "Shapes, parameters and receptive field of a LeNet-style network",
        prompt: "A 3×32×32 colour image passes through: conv 5×5 with 6 kernels (p = 0, s = 1) → ReLU → max-pool 2×2 (s = 2) → conv 5×5 with 16 kernels (p = 0, s = 1) → ReLU → max-pool 2×2 (s = 2) → flatten. (a) Give the tensor shape after every layer. (b) Count each convolution's parameters (with biases). (c) How many numbers come out of the flatten? (d) How wide is the receptive field of one value after the second pooling layer?",
        hint: "Use ⌊(n + 2p − k)/s⌋ + 1 for each layer. Each kernel is as deep as the number of incoming channels. For the receptive field use r_l = r_{l−1} + (k_l − 1)·j_{l−1}, j_l = j_{l−1}·s_l.",
        solution: [
          {
            t: "table",
            head: ["Layer", "Shape", "Parameters"],
            rows: [
              ["Input", "3×32×32", "0"],
              ["Conv 5×5, 6 kernels", "6×28×28  (⌊(32 − 5)/1⌋ + 1 = 28)", "6·3·25 + 6 = 456"],
              ["ReLU", "6×28×28", "0"],
              ["Max-pool 2×2", "6×14×14", "0"],
              ["Conv 5×5, 16 kernels", "16×10×10  (⌊(14 − 5)/1⌋ + 1 = 10)", "16·6·25 + 16 = 2,416"],
              ["ReLU", "16×10×10", "0"],
              ["Max-pool 2×2", "16×5×5", "0"],
              ["Flatten", "400", "0"],
            ],
          },
          { t: "p", text: "(c) 16·5·5 = 400 numbers. (d) Receptive field: conv5 → r = 5, j = 1; pool2 → r = 5 + 1·1 = 6, j = 2; conv5 → r = 6 + 4·2 = 14, j = 2; pool2 → r = 14 + 1·2 = 16, j = 4. Each flattened value depends on a 16×16 patch of the 32×32 image." },
        ],
      },
      {
        t: "exercise",
        title: "Vectorise conv2d with im2col",
        prompt: "The from-scratch conv2d uses four nested Python loops. Write `conv2d_im2col(x, w, b, stride, padding)` that copies every patch the kernel sees into one matrix of shape (N, C_in·k·k, H_out·W_out), using only k² slices (one per kernel offset), and then does the whole layer with a single matrix multiplication. Check it against `torch.nn.functional.conv2d` on 8 random 3×32×32 inputs with 16 kernels of size 3×3, for (s, p) = (1, 1), (2, 0) and (2, 1).",
        hint: "For kernel offset (a, b), the values it touches at every output position are the strided slice xp[:, :, a : a + s·H_out : s, b : b + s·W_out : s]. Put them in an array of shape (N, C, k, k, H_out, W_out), reshape, then multiply by w.reshape(C_out, -1).",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "im2col_conv.py",
            code: String.raw`import numpy as np
import torch
import torch.nn.functional as F


def conv2d_im2col(x, w, b, stride=1, padding=0):
    """Vectorised convolution: gather every receptive-field patch into a column, then one matmul."""
    N, C, H, W = x.shape
    C_out, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    # cols[n, c, a, b, i, j] = xp[n, c, i*stride + a, j*stride + b]
    cols = np.empty((N, C, k, k, H_out, W_out))
    for a in range(k):
        for bb in range(k):
            cols[:, :, a, bb] = xp[:, :, a:a + stride * H_out:stride, bb:bb + stride * W_out:stride]
    cols = cols.reshape(N, C * k * k, H_out * W_out)        # (N, C*k*k, L)
    out = w.reshape(C_out, -1) @ cols + b[:, None]           # (C_out, C*k*k) @ (N, C*k*k, L)
    return out.reshape(N, C_out, H_out, W_out)


rng = np.random.default_rng(1)
x = rng.standard_normal((8, 3, 32, 32))
w = rng.standard_normal((16, 3, 3, 3))
b = rng.standard_normal(16)
for s, p in [(1, 1), (2, 0), (2, 1)]:
    ours = conv2d_im2col(x, w, b, stride=s, padding=p)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=s, padding=p).numpy()
    print(f"s={s} p={p} shape={ours.shape} max diff={np.abs(ours - ref).max():.1e}")`,
            output: String.raw`s=1 p=1 shape=(8, 16, 32, 32) max diff=0.0e+00
s=2 p=0 shape=(8, 16, 15, 15) max diff=3.6e-15
s=2 p=1 shape=(8, 16, 16, 16) max diff=0.0e+00`,
          },
          {
            t: "p",
            text: "The weights reshaped to (C_out, C_in·k·k) and the patches to (C_in·k·k, H_out·W_out) use the same (c, a, b) order, so each output is exactly a flattened kernel dotted with a flattened patch. NumPy's `@` broadcasts the weight matrix over the batch axis. The price is memory: `cols` holds k² = 9 copies of the input.",
          },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "Images are tensors of shape C×H×W (PyTorch) or H×W×C (Keras); a dense layer on a 224×224×3 image would need ~150.5 million weights per 1,000 units.",
          "A convolution slides a k×k (×C_in) kernel over the input. Each output = multiply window and kernel value by value, add it all up, add a bias. The grid of outputs is a feature map.",
          "Deep-learning convolution is really cross-correlation (no flip). True convolution = cross-correlation with the kernel turned 180°; for learned kernels it makes no difference.",
          "Stride s sets the step; padding p adds zeros around the edge. Valid = no padding; same = p = (k − 1)/2 at stride 1.",
          "A layer's weights have shape C_out×C_in×k×k: one kernel covers all input channels and makes one map.",
          "Sharing one kernel everywhere keeps the parameter count independent of image size and makes outputs shift with inputs (equivariance); pooling adds a little shift tolerance and shrinks the map.",
          "Receptive fields grow by (k − 1)·jump per layer; a stack of 3×3 convolutions costs less than one big kernel that sees the same area.",
          "Backward pass: ∂L/∂K = X ⋆ G, ∂L/∂b = ΣG, ∂L/∂X = full convolution of G with K.",
          "Hand-designed kernels (Sobel, blur, Laplacian, sharpen) show what single kernels can extract; CNNs learn their kernels and stack them into feature hierarchies.",
        ],
      },
      {
        t: "math",
        tex: String.raw`Y[o,i,j] = b_o + \sum_{c,m,n} X_p[c, s i+m, s j+n]\,W[o,c,m,n],\quad H_{\text{out}} = \left\lfloor\tfrac{H+2p-k}{s}\right\rfloor + 1,\quad \#\theta = C_{\text{out}}C_{\text{in}}k^2 + C_{\text{out}},\quad r_\ell = r_{\ell-1} + (k_\ell-1)j_{\ell-1}`,
        caption: "Key formulas of the lesson.",
      },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Pooling, Receptive Fields and CNN Architecture](/learn/m08/pooling-receptive-field) — global pooling, flattening, effective receptive fields and how a complete classifier is assembled.",
          "[Dilated, Transposed, Depthwise and Pointwise Convolutions](/learn/m08/conv-variants) — the variants in the table, with their shape arithmetic: how segmentation networks upsample and mobile networks cut cost tenfold.",
          "[Classic CNN Architectures: LeNet to ConvNeXt](/learn/m08/cnn-architectures) — how these building blocks became the networks that defined computer vision, and why residual connections made very deep CNNs trainable.",
          "Revisit [Backpropagation](/learn/m04/backpropagation) with this lesson's gradients in mind: the weight gradient summing over all positions is parameter sharing seen from the gradient side.",
        ],
      },
    ],
  },
};

export default lesson;
