import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m08/images-as-tensors",
  title: "Images as Tensors and Why CNNs",
  summary:
    "How a photo becomes numbers: pixels, bit depth, channels and colour spaces, H×W×C vs C×H×W and NCHW vs NHWC layouts, strides and channels_last memory format, conversion and per-channel normalisation, resizing and interpolation — and a real experiment showing why fully connected networks ignore image structure while CNNs use it.",
  minutes: 55,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices", "m03/multilayer-networks", "m02/probability"],
  next: ["m08/convolution", "m08/pooling-receptive-field", "m07/regularization"],
  terms: ["pixel", "channel", "image-tensor", "nchw", "channels-last", "bit-depth", "inductive-bias"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "To a neural network an image is not a picture. It is a **tensor**: a rectangular block of numbers. A digital photo is a grid of **pixels** (picture elements). Each pixel stores one brightness value (grayscale) or several, such as red, green and blue. So a 640×427 colour photo is 427 rows × 640 columns × 3 colour values = 819,840 numbers, usually stored as 8-bit integers from 0 to 255.",
      },
      {
        t: "p",
        text: "First we cover the ground every vision model stands on: how pixels are encoded, how image tensors sit in memory, how raw bytes are scaled and normalised, and how images are resized without distortion. Getting these wrong does not crash your program — it quietly ruins accuracy.",
      },
      {
        t: "p",
        text: "Then we answer the question behind this whole module: **why not just flatten the image and use a multilayer perceptron?** We count the parameter explosion, then run a real experiment: shuffle the pixels of every digit image with one fixed permutation. The fully connected network does not notice — its accuracy is identical to four decimal places — while convolutional networks get worse. That shows what an MLP throws away (where pixels sit relative to each other) and what a CNN builds in (locality, weight sharing, translation equivariance). These built-in assumptions are called **inductive biases**, and they are why convolution exists. [The Convolution Operation](/learn/m08/convolution) builds the operation itself.",
      },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — a tensor's **shape**, indexing, transposition and broadcasting. Image tensors are 3-D (one image) or 4-D (a batch), and normalisation is a broadcasting operation.",
          String.raw`[Multilayer Networks](/learn/m03/multilayer-networks) — a dense (fully connected) layer computes $W\mathbf{x} + \mathbf{b}$ on a vector. We count its parameters for image inputs and show why it ignores pixel arrangement.`,
          String.raw`[Probability, Random Variables and Distributions](/learn/m02/probability) — mean and standard deviation. Per-channel normalisation is standardisation $(x - \mu)/\sigma$, done channel by channel.`,
          "Basic NumPy: array shapes, `astype`, slicing and `transpose`. Helpful but optional: having opened an image file in Python before.",
        ],
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A mosaic and a bag of tiles",
        text: "A digital image is a mosaic: a grid of small tiles, each painted by mixing red, green and blue paint in amounts from 0 to 255. Write down the three amounts for every tile, row by row, and you have the image tensor. Now tip the tiles into a bag. Every tile is still there with the same colour, but the picture is gone: what made it a face or a digit was the **arrangement**. A fully connected network gets the tiles as a numbered list and treats tile #2,250 and tile #2,251 as unrelated, just as if they came out of the bag. A convolutional network looks at small neighbourhoods of tiles — it reads the mosaic, not the bag.",
      },
      {
        t: "p",
        text: "Three numbers per pixel on a grid gives a 3-D block: height × width × channels. There are two equally valid ways to stack it. Keep each pixel's three colour values together (H×W×C, “channels-last”, how image files and NumPy store it). Or keep each colour plane together, like three transparent sheets — red, green, blue — laid on top of each other (C×H×W, “channels-first”, what PyTorch layers expect). Same numbers, different order.",
      },
      {
        t: "p",
        text: "Before the network sees the numbers we rescale them. Bytes 0–255 are divided by 255 to land in [0, 1]. Then each channel is shifted and scaled so that, over the training data, it has roughly mean 0 and standard deviation 1. It is the same reason you standardise any input: gradient descent works best when inputs are centred and on a similar scale.",
      },
      {
        t: "p",
        text: "Finally, a batch needs one fixed input size, so images are resized. Squashing a 640×427 photo into 224×224 turns circles into ellipses. The standard recipe instead scales the shorter side to 256 and cuts out the central 224×224 square, keeping the aspect ratio.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Pixels and bit depth" },
      {
        t: "p",
        text: "A sensor measures light at each pixel, and the camera **quantises** it (rounds to a whole number). With **bit depth** b there are 2^b levels. 8 bits gives 256 levels (0 = black, 255 = full intensity), the standard for JPEG and PNG, stored as `uint8`. Medical images (CT, MRI in DICOM files) usually use 12 or 16 bits (up to 65,536 levels), camera RAW 12–14 bits, HDR floating point. A network only cares about consistent scaling, so the first step converts integers to floats with a known scale.",
      },
      { t: "h", text: "Grayscale and colour channels" },
      {
        t: "list",
        items: [
          "**Grayscale**: one value per pixel — an H×W matrix. MNIST digits, X-rays and many industrial images are grayscale.",
          "**RGB**: three values per pixel — how much red, green and blue light — so an H×W×3 array. Each 2-D slice (all red values, all green, all blue) is a **channel**. Screens add the three lights together; (255, 255, 255) is white, (255, 0, 0) pure red.",
          "**RGBA**: a fourth **alpha** channel stores opacity (common in PNG files). Most models expect 3 channels, so alpha is dropped or blended onto a background (`img.convert(\"RGB\")` in Pillow).",
          "**Grayscale from RGB** is a weighted sum, not a plain average, because our eyes are most sensitive to green. Pillow's `convert(\"L\")` uses the ITU-R BT.601 luma weights 0.299 R + 0.587 G + 0.114 B.",
          "**Beyond three channels**: satellite images have many spectral bands (e.g. near-infrared), depth cameras add a D channel (RGB-D), and inside a CNN the hidden tensors have 64, 256 or 2,048 “channels” that are learned feature maps, not colours. The tensor machinery is the same.",
        ],
      },
      { t: "h", text: "Other colour spaces (briefly)" },
      {
        t: "p",
        text: "**HSV/HSL** separate hue (which colour), saturation and value/lightness — handy for colour augmentation (“jitter the hue by ±0.05”). **YCbCr** splits brightness (luma) from two colour (chroma) channels; JPEG compresses in this space, storing chroma at lower resolution since the eye is less sensitive to it. **CIE Lab** makes straight-line distances roughly match how different colours look. **BGR** is RGB reversed — OpenCV's `cv2.imread` returns it, a classic bug source. Networks are almost always trained on RGB; the key rule is to feed a pretrained model the colour space and channel order it was trained with.",
      },
      { t: "h", text: "Image tensors: H×W×C vs C×H×W" },
      {
        t: "p",
        text: "One image is a rank-3 tensor (three axes). Image libraries (Pillow, NumPy, OpenCV, matplotlib) and TensorFlow/Keras use **channels-last** H×W×C: `img[r, c]` is one pixel's RGB triple. PyTorch convolution layers use **channels-first** C×H×W: `x[0]` is the whole red plane. Switching between them only reorders axes — `transpose(2, 0, 1)` in NumPy or `permute(2, 0, 1)` in PyTorch — and never changes a value. One more trap: Pillow reports `img.size` as (width, height), while array shapes are (height, width, …).",
      },
      { t: "h", text: "Batches: NCHW vs NHWC" },
      {
        t: "p",
        text: "Networks process **mini-batches**, so a fourth axis N (the image index) goes in front. PyTorch's `nn.Conv2d` expects **N×C×H×W** (NCHW); TensorFlow/Keras defaults to **N×H×W×C** (NHWC). A batch of 32 RGB images of 224×224 is a tensor of shape (32, 3, 224, 224) in PyTorch: 4,816,896 numbers, 18.4 MiB in float32. Video adds a time axis (N×C×T×H×W for PyTorch's 3-D convolutions).",
      },
      { t: "h", text: "Memory layout, strides, permute and channels_last" },
      {
        t: "p",
        text: "A tensor's shape says how to *index* it. Its **strides** say how far apart in memory neighbouring elements along each axis are. A contiguous NCHW tensor of shape (2, 3, 224, 224) has strides (150528, 50176, 224, 1) elements: one column over is 1 element, one row down is 224, one channel over is 50,176 (a whole plane). `permute(0, 2, 3, 1)` returns an NHWC-shaped **view** with permuted strides (150528, 224, 1, 50176). No data moves, and the result is not contiguous; `.contiguous()` would copy it into a new buffer. `view`/`reshape` read memory in order, so using them instead of `permute` to “change the layout” scrambles the image.",
      },
      {
        t: "p",
        text: "PyTorch's **channels_last memory format**, `x.contiguous(memory_format=torch.channels_last)`, keeps the logical shape N×C×H×W (so code still works) but stores data in NHWC order — strides (150528, 1, 672, 3), a pixel's C values side by side. Convolutions on modern NVIDIA GPUs (Tensor Cores, especially with mixed precision) and many CPU libraries run faster this way, so converting the model (`model.to(memory_format=torch.channels_last)`) and inputs is a standard optimisation. It changes speed, never results.",
      },
      { t: "h", text: "From file to network input: the preprocessing pipeline" },
      {
        t: "steps",
        items: [
          { title: "Decode", text: "Read the compressed file (JPEG/PNG) into a uint8 array of shape H×W×3, e.g. `np.asarray(Image.open(path).convert(\"RGB\"))`. `convert(\"RGB\")` handles grayscale, palette and RGBA files the same way." },
          { title: "Resize (and crop)", text: "Bring every image to the model's size while keeping the aspect ratio: resize the shorter side, then centre-crop (evaluation) or random-crop (training)." },
          { title: "Convert to float in [0, 1]", text: "Divide by 255 (by 2^b − 1 for other bit depths). torchvision's `ToTensor` does this and the HWC → CHW permutation in one step." },
          { title: "Normalise per channel", text: "Subtract each channel's mean and divide by its standard deviation, both computed on the training set. For ImageNet-pretrained models the usual values are mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) for (R, G, B)." },
          { title: "Batch", text: "Stack N preprocessed images along a new first axis to get N×C×H×W. A DataLoader does this for you." },
          { title: "(Training only) augment", text: "Apply random changes — flips, crops, colour jitter — before conversion, so every epoch sees slightly different images." },
        ],
      },
      { t: "h", text: "Normalisation: what and why" },
      {
        t: "p",
        text: "After ÷255 every pixel is still positive, with a mean around 0.4–0.5. Per-channel standardisation centres each channel near 0 with unit spread over the dataset. That keeps first-layer pre-activations well scaled at initialisation (He/Xavier assume roughly zero-mean, unit-variance inputs) and helps gradient descent. The statistics are **dataset-level constants**, not per image: the sample photo below has ImageNet-normalised channel means 0.36, 0.51 and 0.65, not 0, because it is brighter and bluer than the average ImageNet image — expected and correct. Fine-tuning? Use the pretrained model's statistics. Training from scratch? Compute them on your training set (an exercise does this).",
      },
      { t: "h", text: "Resizing, interpolation and aspect ratio" },
      {
        t: "list",
        items: [
          "**Nearest neighbour** copies the closest source pixel. Fast, and keeps exact values — essential for segmentation label masks, where averaging class ids 3 and 7 into 5 would be nonsense. But blocky when enlarging and aliased (jagged, with false patterns) when shrinking.",
          "**Bilinear** blends the four surrounding source pixels, weighted by closeness (worked example in the math section). The default for photos.",
          "**Bicubic** fits a cubic curve through 4×4 neighbours: sharper, a bit slower, and can overshoot (values slightly outside the original range).",
          "**Area / antialiased** resampling averages all source pixels covered by an output pixel. When shrinking a lot (e.g. 4000 px → 224 px), plain bilinear samples only a few source pixels per output pixel and causes aliasing; Pillow's resize and torchvision's `Resize(antialias=True)` filter properly.",
          "**Aspect ratio**: forcing a 640×427 image into 224×224 stretches it 1.5× more vertically than horizontally. Three standard alternatives: resize the shorter side and centre-crop (ImageNet classification); resize the longer side and pad (“letterboxing”, common in detection so no object is cut off); or train at the native aspect ratio with a fully convolutional network.",
        ],
      },
      { t: "h", text: "Data augmentation (pointer)" },
      {
        t: "p",
        text: "Image tensors are just arrays, so changes that keep the label — horizontal flips, random crops and rescaling, small rotations, colour jitter, blur, Cutout/Mixup-style mixing — can make endless believable new training images. Augmentation is one of the strongest regularisers in vision; see [Regularization](/learn/m07/regularization). Two rules: augment only the training set (evaluation uses a fixed resize/crop), and make sure the change really keeps the label (a horizontal flip turns a “b” into a “d”).",
      },
      { t: "h", text: "Why fully connected layers fail on images" },
      {
        t: "p",
        text: "An MLP needs a vector, so the image is flattened: (C, H, W) → C·H·W numbers. Problems follow.",
      },
      {
        t: "table",
        head: ["Input image", "Flattened inputs", "Params of one dense layer with 1,000 units", "float32 memory"],
        rows: [
          ["8×8×1 (sklearn digits)", "64", "65,000", "0.2 MiB"],
          ["28×28×1 (MNIST)", "784", "785,000", "3.0 MiB"],
          ["224×224×3 (ImageNet crop)", "150,528", "150,529,000", "574 MiB"],
          ["427×640×3 (the sample photo)", "819,840", "819,841,000", "3,127 MiB"],
        ],
      },
      {
        t: "list",
        items: [
          "**Parameter explosion.** The first layer has (inputs × units) weights. At ImageNet size one 1,000-unit layer has over 150 million parameters — more than all of ResNet-50 (25.6 million) — and would need huge amounts of data not to overfit. A 3×3 convolution with 64 output channels on the same image has 64·3·9 + 64 = 1,792 parameters, whatever the image size.",
          String.raw`**Loss of spatial structure.** A dense layer computes $\sum_i w_i x_i$. The sum is the same in any input order, as long as the weights are reordered too. Neighbouring pixels get no special treatment, so the network must learn from data that pixels (10, 10) and (10, 11) are related. The experiment below makes this concrete.`,
          "**No sharing across positions.** A vertical-stroke detector learned at the left of the image is separate weights from the same detector at the right. Each position must be learned on its own, from examples where the pattern appeared there.",
          "**Fixed input size.** The weight matrix fixes the number of inputs: a model trained on 224×224 cannot even run on a 256×256 image.",
        ],
      },
      { t: "h", text: "The shuffled-pixels experiment" },
      {
        t: "p",
        text: "Take scikit-learn's 8×8 digits. Draw one random permutation of the 64 pixel positions and apply it to every image, training and test alike. A human can no longer read the images, but no information is lost — the shuffle can be undone. Then train three models on original and shuffled data with the same seeds (full code in the PyTorch tab):",
      },
      {
        t: "table",
        head: ["Model", "Original pixels (3 seeds, mean)", "Permuted pixels (mean)", "Change"],
        rows: [
          ["MLP 64 → 64 → 10", "0.9659", "0.9659", "none — identical run for run"],
          ["CNN, conv-pool ×2 + flatten + linear", "0.9793", "0.9644", "−1.5 points"],
          ["CNN, three convs + global average pooling", "0.9689", "0.8296", "−13.9 points"],
        ],
      },
      {
        t: "p",
        text: "The MLP's accuracy does not change at all. Shuffling the inputs and the columns of the first weight matrix the same way gives the same function, so (with matching initialisation) training follows the same path. The MLP is **invariant to any fixed pixel permutation**: it makes no use of which pixels are neighbours. The CNNs get worse because their kernels assume neighbouring inputs belong together; on shuffled pixels a 3×3 window holds 9 unrelated pixels. The drop is small for the flatten + linear CNN (on a tiny 8×8 image its dense layer can still memorise position-specific evidence) and large for the all-convolutional network with global average pooling, which relies entirely on local structure. On real photos the gap is far larger.",
      },
      { t: "h", text: "What CNNs build in: inductive biases" },
      {
        t: "p",
        text: "An **inductive bias** is an assumption built into a model's architecture that favours some solutions before any data is seen. Convolutional layers encode these assumptions about natural images:",
      },
      {
        t: "list",
        items: [
          "**Locality** — useful features come from nearby pixels, so each output looks only at a small window (3×3, 5×5). Long-range interactions are built up by stacking layers.",
          "**Weight sharing (stationarity)** — a pattern means the same thing anywhere in the image, so one small kernel is reused at every position. This makes the parameter count independent of image size.",
          "**Translation equivariance** — follows from the first two: shift the input and the feature map shifts by the same amount. Pooling and global averaging later add some translation *invariance*; see [Pooling, Receptive Fields and CNN Architecture](/learn/m08/pooling-receptive-field).",
          "**Hierarchy** — stacking local layers builds larger, more abstract features (edges → textures → parts → objects) as the receptive field grows layer by layer.",
        ],
      },
      {
        t: "p",
        text: "When the assumptions match the data, a strong bias lets a model learn from less data with far fewer parameters. When they do not — the shuffled pixels above — the bias hurts. [The Convolution Operation](/learn/m08/convolution) builds the operation that implements these assumptions.",
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "Notation" },
      {
        t: "p",
        text: String.raw`An RGB image is a tensor $X \in \{0,\dots,255\}^{H \times W \times 3}$ (channels-last) with entries $X[r, c, k]$: row $r \in \{0,\dots,H-1\}$, column $c \in \{0,\dots,W-1\}$, channel $k \in \{0,1,2\}$ (R, G, B). A batch is $\mathcal{X} \in \mathbb{R}^{N \times C \times H \times W}$ (NCHW). With bit depth $b$, an integer level $q \in \{0, \dots, 2^b - 1\}$ maps to a float in $[0, 1]$ by`,
      },
      {
        t: "math",
        tex: String.raw`x = \frac{q}{2^b - 1} \qquad (b = 8:\; x = q / 255)`,
      },
      { t: "h", text: "Grayscale conversion" },
      {
        t: "math",
        tex: String.raw`Y = 0.299\,R + 0.587\,G + 0.114\,B`,
        caption: "ITU-R BT.601 luma weights (they sum to 1, so white stays white).",
      },
      {
        t: "p",
        text: String.raw`The sample photo's top-left pixel is $(R, G, B) = (174, 201, 231)$, so $Y = 0.299 \cdot 174 + 0.587 \cdot 201 + 0.114 \cdot 231 = 52.026 + 117.987 + 26.334 = 196.347$ — the value the NumPy code prints.`,
      },
      { t: "h", text: "Per-channel normalisation" },
      {
        t: "p",
        text: String.raw`For channel $k$ with dataset mean $\mu_k$ and standard deviation $\sigma_k$ (computed on $[0,1]$-scaled values over all training images and pixels), each value is standardised:`,
      },
      {
        t: "math",
        tex: String.raw`z[k, r, c] = \frac{x[k, r, c] - \mu_k}{\sigma_k}, \qquad \mu_k = \frac{1}{NHW}\sum_{n,r,c} x_n[k,r,c], \quad \sigma_k^2 = \frac{1}{NHW}\sum_{n,r,c}\big(x_n[k,r,c] - \mu_k\big)^2`,
        caption: "Over the training set, every normalised channel then has mean 0 and standard deviation 1.",
      },
      {
        t: "steps",
        items: [
          { title: "Scale the pixel (174, 201, 231) to [0, 1]", tex: String.raw`(174, 201, 231)/255 = (0.682353,\; 0.788235,\; 0.905882)` },
          { title: "Subtract the ImageNet means (0.485, 0.456, 0.406)", tex: String.raw`(0.197353,\; 0.332235,\; 0.499882)` },
          { title: "Divide by the ImageNet stds (0.229, 0.224, 0.225)", tex: String.raw`(0.8618,\; 1.4832,\; 2.2217)` },
        ],
      },
      {
        t: "p",
        text: String.raw`These are exactly the three numbers the NumPy program prints for pixel (0, 0). Normalisation is a scale-and-shift per channel, so $x = \sigma_k z + \mu_k$ undoes it (needed to display a normalised image). It can also be folded into the first convolution: since $\sum w\,(x-\mu)/\sigma = \sum (w/\sigma)\,x - \sum w\mu/\sigma$, scaling the weights by $1/\sigma_k$ and adjusting the bias gives the same outputs.`,
      },
      { t: "h", text: "Memory layout and strides" },
      {
        t: "p",
        text: String.raw`In a contiguous array, element $(n, c, h, w)$ sits at a position (offset) that depends on the layout:`,
      },
      {
        t: "math",
        tex: String.raw`\text{NCHW: } \; \big((n\,C + c)\,H + h\big)\,W + w \qquad\qquad \text{NHWC: } \; \big((n\,H + h)\,W + w\big)\,C + c`,
      },
      {
        t: "p",
        text: String.raw`The coefficients are the strides: NCHW has $(CHW, HW, W, 1)$, NHWC has $(HWC, 1, WC, C)$ when listed in N, C, H, W order. For $C = 3, H = W = 224$: NCHW $(150528, 50176, 224, 1)$ and channels-last $(150528, 1, 672, 3)$ — the numbers PyTorch prints in the framework section. Permuting axes permutes the strides without moving data.`,
      },
      { t: "h", text: "Bilinear interpolation" },
      {
        t: "p",
        text: String.raw`To resize an $H \times W$ image to $H' \times W'$, output pixel $(i, j)$ is mapped back to source coordinates using pixel centres (the “half-pixel” rule of PyTorch's align_corners=False and of OpenCV):`,
      },
      {
        t: "math",
        tex: String.raw`y = (i + 0.5)\frac{H}{H'} - 0.5, \qquad x = (j + 0.5)\frac{W}{W'} - 0.5 \qquad(\text{clamped to } [0, H-1],\,[0, W-1])`,
      },
      {
        t: "p",
        text: String.raw`With $y_0 = \lfloor y \rfloor$, $x_0 = \lfloor x \rfloor$, $\alpha = y - y_0$, $\beta = x - x_0$, the value is a weighted average of the four neighbours — closer neighbours get more weight:`,
      },
      {
        t: "math",
        tex: String.raw`I'(i,j) = (1-\alpha)(1-\beta)\,I(y_0,x_0) + (1-\alpha)\,\beta\,I(y_0,x_0+1) + \alpha(1-\beta)\,I(y_0+1,x_0) + \alpha\,\beta\,I(y_0+1,x_0+1)`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Enlarge $I = \begin{bmatrix}0&1\\1&0\end{bmatrix}$ to $4\times4$, so $H/H' = 0.5$. Output $(1,1)$: $y = x = 1.5 \cdot 0.5 - 0.5 = 0.25$, so $y_0 = x_0 = 0$, $\alpha = \beta = 0.25$, and $I'(1,1) = 0.5625\cdot 0 + 0.1875\cdot 1 + 0.1875 \cdot 1 + 0.0625\cdot 0 = 0.375$. Output $(0,1)$: $y = 0.5\cdot0.5 - 0.5 = -0.25 \to 0$ (clamped), $x = 0.25$, so $I'(0,1) = 0.75\cdot 0 + 0.25 \cdot 1 = 0.25$. The full result, printed by the NumPy code, is`,
      },
      {
        t: "math",
        tex: String.raw`\begin{bmatrix}0&0.25&0.75&1\\0.25&0.375&0.625&0.75\\0.75&0.625&0.375&0.25\\1&0.75&0.25&0\end{bmatrix}`,
      },
      { t: "h", text: "Parameter counts: dense vs convolutional" },
      {
        t: "math",
        tex: String.raw`\#\theta_{\text{dense}} = (C\,H\,W)\cdot m + m \qquad\qquad \#\theta_{\text{conv}} = C_{\text{out}}\,C\,k^2 + C_{\text{out}}`,
        caption: "m hidden units vs C_out kernels of size k×k: the dense count grows with the image area, the convolutional count does not.",
      },
      {
        t: "p",
        text: String.raw`For $224\times224\times3$ and $m = 1000$: $150{,}528 \cdot 1000 + 1000 = 150{,}529{,}000$. For $m = 4096$ (VGG's dense-layer width): $616{,}566{,}784$. For a 3×3 convolution with $C_{\text{out}} = 64$: $64\cdot 3 \cdot 9 + 64 = 1{,}792$ — about 84,000 times fewer than the 1,000-unit dense layer.`,
      },
      { t: "h", text: "Why an MLP is permutation invariant" },
      {
        t: "p",
        text: String.raw`Let $P$ be a permutation matrix (one 1 in each row and column), so $P\mathbf{x}$ reorders the pixels of a flattened image $\mathbf{x}$, and $P^\top P = I$. For a first layer with weights $W$, define $W' = W P^\top$ (the same shuffle applied to the columns). Then`,
      },
      {
        t: "math",
        tex: String.raw`W'(P\mathbf{x}) + \mathbf{b} \;=\; W P^\top P\,\mathbf{x} + \mathbf{b} \;=\; W\mathbf{x} + \mathbf{b}`,
        caption: "Every function an MLP can compute on the original images, it can compute on the permuted images, with equally many parameters.",
      },
      {
        t: "p",
        text: String.raw`If the starting weights are related by $W'_0 = W_0 P^\top$, the gradients stay related the same way at every step ($\partial L / \partial W' = (\partial L/\partial W)P^\top$), and element-wise optimisers like SGD and Adam keep the relation. So the two training runs are the same run, which is why the MLP accuracies match to every printed digit. A convolution has no such freedom. Its kernel $K$ is applied to every window of neighbours, and for an arbitrary pixel permutation $P$ there is in general no kernel $K'$ with $K' \star (P\mathbf{x}) = P(K \star \mathbf{x})$. Its only symmetries are shifts: $K \star (T\mathbf{x}) = T(K \star \mathbf{x})$ (equivariance). The architecture “knows” which pixels are neighbours.`,
      },
      { t: "h", text: "Memory of a batch" },
      {
        t: "p",
        text: String.raw`A batch of $N = 32$ images of $3\times224\times224$ holds $32 \cdot 150{,}528 = 4{,}816{,}896$ values: $4{,}816{,}896$ bytes ($4.59$ MiB) as uint8, $19{,}267{,}584$ bytes ($18.375$ MiB) as float32. That is why datasets are stored as compressed uint8 files and converted to float on the fly, batch by batch.`,
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "cnn-pipeline", caption: "A real image goes through this lesson's preprocessing (centre-crop, resize to 64×64, BT.601 grayscale, scale to [0, 1]) and then through two convolution layers." },
      {
        t: "list",
        items: [
          "Look at the first stage (the input): its shape is 1 × 64 × 64 = 4,096 numbers. The pipeline did exactly the mechanism-section steps: centre-crop to a square (no stretching), resize, grayscale with 0.299R + 0.587G + 0.114B, and divide by 255.",
          "Upload your own photo. Any size or aspect ratio becomes a 1×64×64 tensor; a wide panorama loses its edges to the centre crop. A 1,000-unit dense layer on this input would need 4,096,000 weights; the simulator's first convolution uses a few 3×3 kernels — a few dozen weights.",
          "Step to the first feature maps (6 channels). The tensor is now 6×62×62: the “channels” are no longer colours but the responses of 6 kernels. This is the C axis of NCHW inside a network.",
          "Change the layer-1 stride and padding and watch the shapes in the stage labels change; the formula behind them is the next lesson's topic.",
        ],
      },
      { t: "sim", id: "convolution", caption: "Preview of the next lesson: one 3×3 kernel sliding over a small grid of pixel values." },
      {
        t: "p",
        text: "In the convolution simulator, choose the **Digit “3”** input. It really is a small grayscale image tensor: a grid of intensities. Apply the vertical-edge kernel and notice that each output depends only on a 3×3 neighbourhood — locality in action. Shuffle the input cells and the kernel would combine unrelated pixels, which is what happens to the CNNs in the shuffled-pixels experiment.",
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "The whole lesson in NumPy: load a real JPEG (one of scikit-learn's two sample photos, decoded with Pillow), check shape, dtype and memory, make grayscale, convert uint8 → float, reorder HWC → CHW and inspect strides, normalise with ImageNet statistics, form a batch, write nearest-neighbour and bilinear resizing from scratch, and tabulate the dense-layer parameter explosion.",
      },
      {
        t: "code",
        lang: "python",
        title: "image_tensors_numpy.py",
        code: String.raw`import numpy as np
from sklearn.datasets import load_sample_image

# ---- 1) Load a real photo: sklearn ships two JPEGs (decoded with Pillow) ----
img = load_sample_image("china.jpg")            # NumPy array, channels-last (H, W, C)
H, W, C = img.shape
print("shape (H, W, C):", img.shape, " dtype:", img.dtype)
print("min / max:", img.min(), img.max(), " bytes:", img.nbytes)
print("pixel (row 0, col 0) RGB:", img[0, 0])
print("per-channel mean (0-255):", np.round(img.reshape(-1, 3).mean(axis=0), 2))

# ---- 2) Grayscale: weighted sum of R, G, B (ITU-R BT.601 luma weights) ----
luma = np.array([0.299, 0.587, 0.114])
gray = img.astype(np.float64) @ luma            # (H, W, 3) @ (3,) -> (H, W)
print("gray shape:", gray.shape, " gray[0,0] = %.3f" % gray[0, 0])

# ---- 3) uint8 -> float in [0, 1], then HWC -> CHW (what PyTorch wants) ----
x = img.astype(np.float32) / 255.0
chw = x.transpose(2, 0, 1)                      # axes (H, W, C) -> (C, H, W): a view, no copy
print("CHW shape:", chw.shape, " C-contiguous:", chw.flags["C_CONTIGUOUS"])
print("strides HWC (bytes):", x.strides, " strides CHW view:", chw.strides)

# ---- 4) Per-channel normalisation with the ImageNet statistics used by torchvision ----
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
z = (chw - mean[:, None, None]) / std[:, None, None]   # broadcast over H and W
print("normalised pixel (0,0):", np.round(z[:, 0, 0], 4))
print("normalised per-channel mean:", np.round(z.mean(axis=(1, 2)), 4))
print("normalised per-channel std: ", np.round(z.std(axis=(1, 2)), 4))

# ---- 5) A batch of N images: stack along a new leading axis -> NCHW ----
batch = np.stack([z, z[:, :, ::-1]])            # original + horizontally flipped copy
print("batch NCHW:", batch.shape, " as NHWC:", batch.transpose(0, 2, 3, 1).shape)


# ---- 6) Resizing: nearest-neighbour and bilinear interpolation from scratch ----
def resize_nearest(a, out_h, out_w):
    h, w = a.shape[:2]
    # map output pixel centres back to input coordinates (half-pixel convention)
    rows = np.minimum(((np.arange(out_h) + 0.5) * h / out_h).astype(int), h - 1)
    cols = np.minimum(((np.arange(out_w) + 0.5) * w / out_w).astype(int), w - 1)
    return a[rows][:, cols]


def resize_bilinear(a, out_h, out_w):
    h, w = a.shape[:2]
    y = np.clip((np.arange(out_h) + 0.5) * h / out_h - 0.5, 0, h - 1)
    xx = np.clip((np.arange(out_w) + 0.5) * w / out_w - 0.5, 0, w - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (y - y0)[:, None, None], (xx - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx       # blend along x on row y0
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx       # ... and on row y1
    return top * (1 - wy) + bot * wy                        # blend the two rows along y


small_nn = resize_nearest(x, 107, 160)          # keep the 427:640 aspect ratio (~2:3)
small_bl = resize_bilinear(x, 107, 160)
print("nearest:", small_nn.shape, " bilinear:", small_bl.shape)
tiny = np.array([[0.0, 1.0], [1.0, 0.0]])[:, :, None]
print("2x2 -> 4x4 bilinear:\n", np.round(resize_bilinear(tiny, 4, 4)[:, :, 0], 3))

# ---- 7) Why dense layers do not scale: first-layer parameters for 1000 hidden units ----
for h, w, c in [(8, 8, 1), (28, 28, 1), (224, 224, 3), (H, W, C)]:
    n_in = h * w * c
    params = n_in * 1000 + 1000
    print(f"{h}x{w}x{c}: {n_in:>7d} inputs -> {params:>11,d} params ({params * 4 / 2**20:8.1f} MiB float32)")`,
        output: String.raw`shape (H, W, C): (427, 640, 3)  dtype: uint8
min / max: 0 255  bytes: 819840
pixel (row 0, col 0) RGB: [174 201 231]
per-channel mean (0-255): [144.72 145.47 140.92]
gray shape: (427, 640)  gray[0,0] = 196.347
CHW shape: (3, 427, 640)  C-contiguous: False
strides HWC (bytes): (7680, 12, 4)  strides CHW view: (4, 7680, 12)
normalised pixel (0,0): [0.8618 1.4832 2.2217]
normalised per-channel mean: [0.3604 0.511  0.6518]
normalised per-channel std:  [1.3432 1.4664 1.6703]
batch NCHW: (2, 3, 427, 640)  as NHWC: (2, 427, 640, 3)
nearest: (107, 160, 3)  bilinear: (107, 160, 3)
2x2 -> 4x4 bilinear:
 [[0.    0.25  0.75  1.   ]
 [0.25  0.375 0.625 0.75 ]
 [0.75  0.625 0.375 0.25 ]
 [1.    0.75  0.25  0.   ]]
8x8x1:      64 inputs ->      65,000 params (     0.2 MiB float32)
28x28x1:     784 inputs ->     785,000 params (     3.0 MiB float32)
224x224x3:  150528 inputs -> 150,529,000 params (   574.2 MiB float32)
427x640x3:  819840 inputs -> 819,841,000 params (  3127.4 MiB float32)`,
      },
      {
        t: "list",
        items: [
          "`img.nbytes` = 427·640·3 = 819,840: one byte per value in uint8. The float32 copy `x` is four times larger.",
          "The CHW array is a **view**: its strides (4, 7680, 12) bytes show a step along C moves 4 bytes (one float32) while a step along W moves 12 (a whole pixel). It is not C-contiguous; `np.ascontiguousarray` would copy it into true CHW order.",
          "After ImageNet normalisation this one image's channel means are 0.36–0.65, not 0: the statistics belong to the dataset, not each image.",
          "The resized images keep the 427:640 ≈ 2:3 aspect ratio (107×160).",
          "`resize_bilinear` reproduces PyTorch's `F.interpolate(..., mode=\"bilinear\", align_corners=False)` to within 1e-13 — the coding exercise checks this.",
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
                text: "**1. Loading and preprocessing with Pillow and torchvision.** `ToTensor` turns a Pillow image or HWC uint8 array into a CHW float tensor in [0, 1]; `Normalize` standardises per channel; `Resize(256)` scales the shorter side to 256, keeping the aspect ratio; `CenterCrop(224)` cuts the central square. The script also shows NCHW vs NHWC strides and channels_last.",
              },
              {
                t: "code",
                lang: "python",
                title: "torchvision_preprocess.py",
                code: String.raw`import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_sample_image
from torchvision import transforms as T

arr = load_sample_image("china.jpg")              # (427, 640, 3) uint8, HWC
img = Image.fromarray(arr)                         # a Pillow image, as Image.open("x.jpg") would give
print("PIL size (W, H):", img.size, " mode:", img.mode)

# ---- ToTensor: HWC uint8 [0, 255]  ->  CHW float32 [0, 1] ----
t = T.ToTensor()(img)
print("ToTensor:", tuple(t.shape), t.dtype, f"range [{t.min():.3f}, {t.max():.3f}]")
print("same as manual /255 + permute:",
      torch.equal(t, torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255))

# ---- The standard ImageNet evaluation pipeline used with torchvision models ----
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
preprocess = T.Compose([
    T.Resize(256),            # shorter side -> 256, aspect ratio kept: 427x640 -> 256x383
    T.CenterCrop(224),        # central 224x224 crop
    T.ToTensor(),
    T.Normalize(MEAN, STD),   # (x - mean[c]) / std[c] for each channel c
])
x = preprocess(img)
print("after Resize(256):", T.Resize(256)(img).size, "-> final tensor", tuple(x.shape))
print("per-channel mean:", [round(v, 3) for v in x.mean(dim=(1, 2)).tolist()])
print("per-channel std: ", [round(v, 3) for v in x.std(dim=(1, 2)).tolist()])

# ---- Undo the normalisation (e.g. to display an image) ----
m, s = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
restored = x * s + m
print("restored range:", f"[{restored.min():.3f}, {restored.max():.3f}]")

# ---- Batches, layouts and memory format ----
batch = torch.stack([x, x.flip(-1)])              # N x C x H x W  (PyTorch convention)
nhwc = batch.permute(0, 2, 3, 1)                  # N x H x W x C  (TensorFlow convention)
print("NCHW", tuple(batch.shape), "strides", batch.stride())
print("NHWC view", tuple(nhwc.shape), "strides", nhwc.stride(), "contiguous:", nhwc.is_contiguous())
cl = batch.contiguous(memory_format=torch.channels_last)
print("channels_last: shape", tuple(cl.shape), "strides", cl.stride(), "same values:", torch.equal(cl, batch))`,
                output: String.raw`PIL size (W, H): (640, 427)  mode: RGB
ToTensor: (3, 427, 640) torch.float32 range [0.000, 1.000]
same as manual /255 + permute: True
after Resize(256): (383, 256) -> final tensor (3, 224, 224)
per-channel mean: [0.424, 0.503, 0.654]
per-channel std:  [1.226, 1.39, 1.563]
restored range: [0.000, 1.000]
NCHW (2, 3, 224, 224) strides (150528, 50176, 224, 1)
NHWC view (2, 224, 224, 3) strides (150528, 224, 1, 50176) contiguous: False
channels_last: shape (2, 3, 224, 224) strides (150528, 1, 672, 3) same values: True`,
              },
              {
                t: "list",
                items: [
                  "`ToTensor()(img)` equals `torch.from_numpy(arr).permute(2, 0, 1).float() / 255` exactly.",
                  "`Resize(256)` with an int resizes the **shorter** side (427 → 256, 640 → 383); `Resize((256, 256))` with a tuple would ignore the aspect ratio.",
                  "Current torchvision also offers the `torchvision.transforms.v2` API, which handles boxes and masks too; for pretrained models, `weights.transforms()` returns their exact training preprocessing.",
                  "The channels_last tensor equals the NCHW tensor: same values, different memory order (strides).",
                ],
              },
              {
                t: "p",
                text: "**2. The shuffled-pixels experiment.** Three architectures, each trained on original digits and on digits with one fixed pixel shuffle, 3 seeds each. The permuted MLP starts from identically permuted initial weights, so both runs compute the same function from step one. About 25 seconds on a CPU.",
              },
              {
                t: "code",
                lang: "python",
                title: "shuffled_pixels.py",
                code: String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0            # (1797, 8, 8) in [0, 1]
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# One fixed random permutation of the 64 pixel positions, applied to EVERY image
perm = np.random.default_rng(0).permutation(64)
def shuffle_pixels(imgs):
    return imgs.reshape(len(imgs), 64)[:, perm].reshape(-1, 8, 8)

data = {
    "original": (X_tr, X_te),
    "permuted": (shuffle_pixels(X_tr), shuffle_pixels(X_te)),
}
y_tr_t, y_te_t = torch.tensor(y_tr), torch.tensor(y_te)


def make_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x4x4
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x2x2
        nn.Flatten(), nn.Linear(128, 10))


def make_gap():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def train_eval(model, Xtr, Xte, epochs=15, seed=0):
    g = torch.Generator().manual_seed(seed)          # same mini-batch order for every run
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xtr, xte = torch.tensor(Xtr).unsqueeze(1), torch.tensor(Xte).unsqueeze(1)
    for _ in range(epochs):
        order = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(xtr[idx]), y_tr_t[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (model(xte).argmax(1) == y_te_t).float().mean().item()


for name, make in [("MLP", make_mlp), ("CNN+flatten", make_cnn), ("CNN+GAP", make_gap)]:
    accs = []
    for seed in range(3):
        row = []
        for key in ["original", "permuted"]:
            torch.manual_seed(seed)
            model = make()
            if name == "MLP" and key == "permuted":
                # Give the MLP the SAME initial function: permute the columns of the first
                # weight matrix exactly like the pixels. Then the two runs are the same network.
                with torch.no_grad():
                    model[1].weight[:] = model[1].weight[:, perm]
            row.append(train_eval(model, *data[key], seed=seed))
        accs.append(row)
    accs = np.array(accs)
    print(f"{name}: test accuracy original {np.round(accs[:, 0], 4)}  mean {accs[:, 0].mean():.4f}")
    print(f"{name}: test accuracy permuted {np.round(accs[:, 1], 4)}  mean {accs[:, 1].mean():.4f}")`,
                output: String.raw`MLP: test accuracy original [0.9778 0.9644 0.9556]  mean 0.9659
MLP: test accuracy permuted [0.9778 0.9644 0.9556]  mean 0.9659
CNN+flatten: test accuracy original [0.9778 0.98   0.98  ]  mean 0.9793
CNN+flatten: test accuracy permuted [0.9689 0.9533 0.9711]  mean 0.9644
CNN+GAP: test accuracy original [0.98   0.9578 0.9689]  mean 0.9689
CNN+GAP: test accuracy permuted [0.94   0.6733 0.8756]  mean 0.8296`,
              },
              {
                t: "list",
                items: [
                  "The MLP rows match seed by seed: shuffling pixels changes nothing for a network with no idea of neighbourhood.",
                  "The flatten + linear CNN drops 1.5 points on average; the all-convolutional CNN with global average pooling drops 13.9 points and varies wildly across seeds (0.94, 0.67, 0.88).",
                  "Without the weight-permutation trick, the two MLP runs would start from different random weights and differ slightly by chance — but not systematically.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras is channels-last (NHWC) throughout. Preprocessing can live inside the model as Keras layers: `Resizing`, `CenterCrop`, `Rescaling(1/255)` and `Normalization(mean=…, variance=…)`. Note that `Normalization` takes the **variance** (std²), not the standard deviation.",
              },
              {
                t: "code",
                lang: "python",
                title: "keras_preprocess.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_image

arr = load_sample_image("china.jpg")                    # (427, 640, 3) uint8, HWC
batch = tf.convert_to_tensor(arr[None])                 # (1, 427, 640, 3) NHWC
print("input:", batch.shape, batch.dtype)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 384, crop_to_aspect_ratio=True),  # keep aspect ratio
    tf.keras.layers.CenterCrop(224, 224),
    tf.keras.layers.Rescaling(1.0 / 255),                            # uint8 -> [0, 1]
    tf.keras.layers.Normalization(mean=mean, variance=std**2),       # variance, not std!
])
x = preprocess(batch)
print("output:", x.shape, "per-channel mean:", tf.reduce_mean(x, axis=(0, 1, 2)).numpy())

# Converting to PyTorch's layout is a transpose of axes
nchw = tf.transpose(x, perm=[0, 3, 1, 2])
print("NCHW:", nchw.shape)

# Loading from disk: tf.keras.utils.load_img("photo.jpg") returns a PIL image;
# tf.keras.utils.img_to_array(...) turns it into an HWC float32 array (values still 0..255).`,
                output: "Prints the input shape (1, 427, 640, 3) with dtype uint8, the preprocessed shape (1, 224, 224, 3) with its per-channel means (close to, but not exactly, the PyTorch values because the resize and crop differ slightly), and the transposed NCHW shape (1, 3, 224, 224).",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      { t: "h", text: "Pixel encodings" },
      {
        t: "table",
        head: ["Encoding", "Channels", "Typical dtype / range", "Where it appears", "Notes"],
        rows: [
          ["Grayscale", "1", "uint8 0–255", "Digits, documents, X-rays", "Smallest; no colour cues"],
          ["RGB", "3", "uint8 0–255", "Photos (JPEG, PNG), most datasets", "Default for pretrained models"],
          ["BGR", "3", "uint8 0–255", "OpenCV (`cv2.imread`), old Caffe models", "Same data, reversed channel order"],
          ["RGBA", "4", "uint8 0–255", "PNG with transparency", "Drop the alpha (see-through) channel, or paste onto a background, before a 3-channel model"],
          ["YCbCr", "3", "uint8", "Inside JPEG and video files", "Keeps brightness apart from colour"],
          ["HSV / Lab", "3", "float", "Colour augmentation, colour analysis", "Axes closer to how people see colour"],
          ["16-bit / float", "1+", "uint16, float32", "Medical (DICOM), scientific, HDR, RAW", "Don't ÷255; use windowing or dataset statistics"],
          ["Multispectral", "4–13+", "uint16/float", "Satellite images", "The first conv layer must take C input channels"],
        ],
      },
      { t: "h", text: "Tensor layouts" },
      {
        t: "table",
        head: ["Layout", "Shape order", "Used by", "Advantages", "Disadvantages"],
        rows: [
          ["HWC", "H, W, C", "Pillow/NumPy/OpenCV/matplotlib", "Natural for files and showing images", "Not what PyTorch layers expect"],
          ["CHW / NCHW", "N, C, H, W", "PyTorch (default), older cuDNN kernels", "Each channel's plane sits in one block of memory", "Can be slower on Tensor-Core GPUs"],
          ["NHWC", "N, H, W, C", "TensorFlow/Keras (default), TFLite", "A pixel's channels sit side by side; fast on modern GPUs/CPUs", "Needs a transpose for PyTorch code"],
          ["channels_last (PyTorch)", "shape NCHW, stored NHWC", "PyTorch speed mode", "NHWC speed with no code changes", "Convert both the model and the inputs"],
          ["NCTHW", "N, C, T, H, W", "Video / 3-D medical scans in PyTorch", "Works with Conv3d", "Uses a lot of memory"],
        ],
      },
      { t: "h", text: "Normalisation schemes" },
      {
        t: "table",
        head: ["Scheme", "Formula", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Scale to [0, 1]", "x / 255", "Simple, no statistics needed", "Not centred on 0", "Small models trained from scratch, toy data"],
          ["Scale to [−1, 1]", "x / 127.5 − 1", "Centred on 0, bounded", "Treats all channels the same", "GANs, diffusion models (tanh outputs)"],
          ["Dataset mean/std per channel", "(x − μ_k)/σ_k", "Mean 0, std 1: easy for training", "You must save and reuse the statistics", "Most classifiers; ImageNet-pretrained models"],
          ["Per-image standardisation", "(x − mean(x))/std(x)", "Not fooled by too-bright or too-dark photos", "Throws away how bright the image really was", "Some medical and satellite pipelines"],
          ["Learned (BatchNorm on input)", "BN layer first", "Statistics are tracked for you", "Extra layer; depends on batch size", "Now and then in custom models"],
        ],
      },
      { t: "h", text: "Resizing strategies" },
      {
        t: "table",
        head: ["Strategy", "Aspect ratio", "Information loss", "Typical use"],
        rows: [
          ["Direct resize to H×W", "Distorted", "Nothing cut off, but shapes get stretched", "When stretching does no harm (some textures, medical patches)"],
          ["Shorter side + centre crop", "Preserved", "Edges cropped", "Classification evaluation (Resize 256, CenterCrop 224)"],
          ["Random resized crop", "Varied on purpose", "Random region", "Classification training (augmentation)"],
          ["Longer side + pad (letterbox)", "Preserved", "None; padding added", "Object detection"],
          ["Native size, fully convolutional", "Preserved", "None", "Segmentation, detection backbones"],
        ],
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Transfer learning with pretrained models** — every torchvision, timm or Keras Applications model documents its expected input (RGB, size, mean/std). Wrong normalisation or BGR images is one of the commonest reasons a fine-tuned model underperforms.",
          "**Medical imaging** — CT scans store Hounsfield units in 12–16 bits. Pipelines use “windowing” (clip to a range such as soft tissue, then rescale) instead of ÷255, and stack slices into 3-D tensors (N×C×D×H×W).",
          "**Remote sensing** — Sentinel-2 satellite products have 13 spectral bands; models use C = 13 input channels with per-band statistics.",
          "**Object detection and segmentation** — letterboxing keeps geometry; masks are resized with nearest neighbour so class ids stay valid.",
          "**Video understanding** — frames stack along a time axis; memory grows with frame count, so layout and dtype choices matter.",
          "**High-throughput pipelines** — store uint8, decode on the CPU, normalise on the GPU, and use channels_last with mixed precision: routine production optimisations.",
        ],
      },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Representation choices" },
      {
        t: "list",
        items: [
          "**uint8 vs float32 storage**: uint8 is 4× smaller and is what files hold; float32 is needed for arithmetic. Convert as late as possible (per batch, ideally on the GPU).",
          "**NCHW vs NHWC**: same meaning; speed depends on hardware and kernels. Benchmark channels_last on your GPU first.",
          "**Resolution**: higher keeps small details, but compute grows with H·W (a dense layer's parameters would too). Classification often uses 224–384 px; detection and segmentation more.",
          "**Interpolation**: bilinear/area for images, nearest for labels; antialias when shrinking a lot.",
          "**Normalisation statistics**: reuse the pretrained model's when fine-tuning; compute your own when training from scratch on non-ImageNet-like data (medical, satellite, grayscale).",
        ],
      },
      { t: "h", text: "MLPs vs CNNs on images" },
      {
        t: "table",
        head: ["Aspect", "Fully connected (MLP)", "Convolutional (CNN)"],
        rows: [
          ["Parameters vs image size", "Grow in step with H·W·C", "Do not depend on H·W"],
          ["Uses pixel neighbourhoods", "No — any fixed shuffle of pixels changes nothing", "Yes — looks at nearby pixels by design"],
          ["Position handling", "Separate weights for every position", "One shared kernel; a shifted input gives a shifted output"],
          ["Input size", "Fixed", "Any (fully convolutional)"],
          ["When to pick it", "Tiny inputs, plain feature vectors, data with no spatial order", "Any data on a grid: images, spectrograms, 3-D volumes"],
        ],
      },
      {
        t: "p",
        text: "Strong inductive biases are a trade-off, not a free lunch: they help most when data is limited. With very large datasets, models with weaker spatial biases (Vision Transformers, which see the image as a set of patches plus position information) can match or beat CNNs — though they too start by cutting the image into local patches.",
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Passing HWC data to PyTorch.** A (N, H, W, 3) batch fed to `nn.Conv2d(3, …)` either errors (expected 3 channels but got 224) or, worse, runs with the wrong axis as channels when sizes happen to match. Fix: `x.permute(0, 3, 1, 2)`; assert `x.shape[1] == in_channels`.",
          "**Using reshape/view instead of permute.** `x.reshape(3, H, W)` on an HWC array keeps memory order and cuts it into three meaningless blocks, scrambling pixels across channels. Reordering axes needs `permute`/`transpose`.",
          "**BGR vs RGB.** `cv2.imread` returns BGR, so an RGB-trained model sees red and blue swapped — skies turn orange. Fix: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` or `img[..., ::-1]`.",
          "**Forgetting to scale, or scaling twice.** 0–255 values into a model trained on [0, 1] saturate its activations; `/255` after `ToTensor` (which already scaled) makes everything nearly black. Know which step scales.",
          "**Wrong normalisation statistics.** ImageNet statistics for a model trained on [−1, 1] inputs (or vice versa) shift every input. Match the preprocessing to the weights you load.",
          "**Computing statistics per batch or on the test set.** Mean/std are fixed training-set constants; recomputing them on test data leaks information and changes the input distribution.",
          "**uint8 arithmetic overflow.** In NumPy, `np.array([200], np.uint8) + np.array([100], np.uint8)` gives `[44]` (300 mod 256) with no error. Convert to float (or a wider integer) before brightness changes or averaging.",
          "**Confusing Pillow's size with array shape.** `Image.size` is (width, height); `np.asarray(img).shape` is (height, width, channels). Mixing them causes many transpose bugs, e.g. in `Resize((h, w))` calls.",
          "**Distorting aspect ratio by accident.** `Resize((224, 224))` stretches non-square images. Use `Resize(256)` + `CenterCrop(224)` or letterboxing unless you want distortion.",
          "**Interpolating label masks bilinearly.** Averaging class ids creates invalid classes at boundaries. Use nearest neighbour for masks.",
          "**Augmenting the evaluation set.** Random crops or flips at test time make metrics noisy and unrepeatable; use fixed preprocessing for validation and test (test-time augmentation is a separate, deliberate technique).",
          "**Displaying a normalised tensor directly.** `plt.imshow` on a normalised CHW tensor fails or shows garbage. Undo the normalisation, permute to HWC and clip to [0, 1] first.",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "How is a colour image represented as a tensor?", a: "As a 3-D array of pixel intensities: height × width × channels, with 3 channels (red, green, blue) for colour and 1 for grayscale. Values are usually 8-bit integers 0–255. Pillow and NumPy store H×W×C (channels-last); PyTorch layers expect C×H×W (channels-first). A batch adds a leading axis: N×C×H×W in PyTorch, N×H×W×C in TensorFlow." },
      { t: "qa", level: "beginner", q: "Why do we divide pixel values by 255 and then normalise with a mean and standard deviation?", a: "Dividing by 255 turns 8-bit integers into floats in [0, 1]. Per-channel normalisation (x − μ_c)/σ_c then centres each channel near zero with unit spread over the training set. Centred, similarly scaled inputs keep early pre-activations in a good range at initialisation and make optimisation better behaved. The statistics are dataset constants; a pretrained model needs the ones it was trained with, e.g. ImageNet's mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)." },
      { t: "qa", level: "beginner", q: "What goes wrong if you flatten a 224×224 RGB image into an MLP?", a: "The input has 150,528 values, so one 1,000-unit dense layer needs about 150.5 million weights — hard to train and prone to overfitting. The MLP also ignores which pixels are neighbours (it is invariant to any fixed reordering of pixels), cannot reuse a detector learned at one position elsewhere, and accepts only one input size. Convolutions fix all four with small shared local kernels." },
      { t: "qa", level: "intermediate", q: "What is the difference between NCHW and NHWC, and what does PyTorch's channels_last memory format do?", a: "Two orders of the same four axes. NCHW stores each channel plane contiguously (strides CHW, HW, W, 1); NHWC stores all channels of a pixel together. PyTorch's channels_last keeps the logical NCHW shape, so code is unchanged, but stores data in NHWC order (strides HWC, 1, WC, C). Convolutions on Tensor-Core GPUs and many CPU backends are faster in NHWC, so converting model and inputs to channels_last is a common speed-up with identical results." },
      { t: "qa", level: "intermediate", q: "You fine-tune an ImageNet-pretrained ResNet and accuracy is mysteriously poor. Which preprocessing issues would you check?", a: "Channel order (OpenCV loads BGR); scaling (0–255 vs [0, 1], or scaled twice); normalisation statistics (the model expects ImageNet mean/std); layout (HWC where CHW is expected, or reshape instead of permute); resolution and aspect ratio (stretched images, very different size); augmentation at evaluation; and whether training and inference use the same transforms. Looking at a de-normalised batch catches most of these in seconds." },
      { t: "qa", level: "intermediate", q: "Which interpolation method would you use to resize (a) a photo, (b) a segmentation mask, and why?", a: "(a) Bilinear, bicubic or area/antialiased resampling: they blend neighbours for smooth results; antialiasing matters when shrinking a lot. (b) Nearest neighbour: a mask holds class ids, and averaging ids (3 and 7 into 5) creates invalid labels at boundaries. Nearest neighbour keeps the exact set of values." },
      { t: "qa", level: "advanced", q: "Prove that an MLP is invariant to a fixed permutation of its input pixels, and explain why a CNN is not.", a: "Let P be a permutation matrix and W the first-layer weights, and set W' = W Pᵀ. Then W'(Px) = W PᵀP x = Wx, so the permuted-input network with W' computes exactly the same function; the hypothesis class is unchanged. If the initial weights are related this way, gradients obey ∂L/∂W' = (∂L/∂W)Pᵀ, and element-wise optimisers keep the relation, so training is identical (the experiment shows identical accuracies). A convolution applies one kernel to windows of neighbouring pixels; for a general permutation there is no kernel K' with K' ⋆ (Px) = P(K ⋆ x). Its symmetries are the translations (equivariance), so destroying adjacency removes the structure its inductive bias relies on." },
      { t: "qa", level: "advanced", q: "How can input normalisation be folded into the first convolution layer, and why might you do it?", a: "Normalisation is z_c = (x_c − μ_c)/σ_c. A convolution computes Σ_c Σ_{m,n} W[o,c,m,n] z_c + b_o = Σ W[o,c,m,n]/σ_c · x_c + (b_o − Σ_c Σ_{m,n} W[o,c,m,n] μ_c/σ_c). So dividing each input-channel slice of the weights by σ_c and subtracting the folded term from the bias gives a layer that takes [0, 1] inputs directly, saving a preprocessing pass at inference (useful on edge devices). It is exact away from the borders. With zero padding, padded zeros mean z = 0, i.e. x = μ, so border outputs differ slightly unless the padding value is adjusted." },
      { t: "qa", level: "advanced", q: "Why do strong inductive biases help CNNs on small datasets but matter less for Vision Transformers trained on huge datasets?", a: "An inductive bias limits the model to functions that fit its assumptions (locality, weight sharing, translation equivariance). If the assumptions hold, the model needs fewer examples, because it does not have to learn them from data. A Vision Transformer has weaker spatial bias — global attention between patches, learned position embeddings — so with little data it overfits or learns slowly. With enough data it learns spatial regularities itself and also captures long-range interactions a CNN builds only gradually. Hybrid designs (convolutional stems, local attention) try to get both." },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            kind: "numeric",
            q: "How many numbers are in a batch of 16 RGB images of size 128×128?",
            answer: 786432,
            tolerance: 0,
            explain: "16 × 3 × 128 × 128 = 16 × 49,152 = 786,432.",
          },
          {
            kind: "numeric",
            q: "A pixel's red value is 230. With ImageNet's red mean 0.485 and std 0.229, what is its normalised value? (2 decimals)",
            answer: 1.82,
            tolerance: 0.01,
            explain: "230/255 = 0.90196; (0.90196 − 0.485)/0.229 = 0.41696/0.229 = 1.8208.",
          },
          {
            q: "You open a 640-wide, 427-tall RGB photo with Pillow and turn it into an array with np.asarray. What shape is it?",
            options: ["(640, 427, 3)", "(427, 640, 3)", "(3, 427, 640)", "(3, 640, 427)"],
            answer: 1,
            explain: "Image arrays in NumPy are (height, width, channels). Careful: Pillow's img.size gives (640, 427) = (width, height).",
          },
          {
            kind: "numeric",
            q: "A 32×32×3 image is flattened into a dense layer with 500 units. How many parameters (weights + biases) does that layer have?",
            answer: 1536500,
            tolerance: 0,
            explain: "32·32·3 = 3,072 inputs; 3,072 × 500 + 500 = 1,536,000 + 500 = 1,536,500.",
          },
          {
            q: "In the shuffled-pixels experiment, why is the MLP's test accuracy exactly unchanged?",
            options: [
              "Because digits are easy and every model reaches the same accuracy",
              "Because shuffling the inputs is the same as shuffling the columns of the first weight matrix, so the same function and training path exist",
              "Because the DataLoader quietly undid the shuffle",
              "Because MLPs use weight sharing",
            ],
            answer: 1,
            explain: "W(Pᵀ)(Px) = Wx. If the starting weights are shuffled the same way, both runs do the very same arithmetic. An MLP has no idea which inputs are neighbours.",
          },
          {
            q: "A PyTorch tensor of shape (8, 3, 32, 32) is converted with x.contiguous(memory_format=torch.channels_last). What is x.shape now?",
            options: ["(8, 32, 32, 3)", "(8, 3, 32, 32)", "(3, 8, 32, 32)", "(8, 3072)"],
            answer: 1,
            explain: "channels_last only changes how values are laid out in memory (strides become (3072, 1, 96, 3)); the shape you see stays NCHW.",
          },
          {
            kind: "numeric",
            q: "Using BT.601 weights, what is the grayscale value of the pure-green pixel (0, 255, 0)? (1 decimal)",
            answer: 149.7,
            tolerance: 0.05,
            explain: "0.299·0 + 0.587·255 + 0.114·0 = 149.685 ≈ 149.7. Green contributes most to perceived brightness.",
          },
          {
            q: "Which resize method should be used for a segmentation label mask?",
            options: ["Bicubic", "Bilinear", "Nearest neighbour", "Area averaging"],
            answer: 2,
            explain: "Only nearest neighbour copies existing class ids. The others average ids (say 2 and 4 into 3), which makes nonsense labels.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Shapes, memory and normalisation by hand",
        prompt: "A training pipeline loads 4,000×3,000 RGB JPEGs, resizes the shorter side to 256, centre-crops 224×224, and batches 64 images. (a) One decoded photo as a uint8 NumPy array (height 3,000): what shape, and how many bytes? (b) After Resize(256), what size (H×W) is it before cropping? (c) What shape is one PyTorch batch, and how many MiB is it as float32? (d) Normalise the pixel (R, G, B) = (51, 102, 204) with ImageNet statistics. (e) How many parameters would a 1,000-unit dense layer on one cropped image need?",
        hint: "Resize(256) makes the shorter side 256 and keeps the shape's ratio (torchvision rounds the other side down). 1 MiB = 2²⁰ bytes.",
        solution: [
          {
            t: "list",
            items: [
              "(a) (3000, 4000, 3) uint8: 3,000·4,000·3 = 36,000,000 bytes (≈ 34.3 MiB).",
              "(b) Shorter side 3,000 → 256, so the scale is 256/3,000; the width becomes 4,000·256/3,000 = 341.33 → 341. Size 256×341 (H×W).",
              "(c) (64, 3, 224, 224): 64·150,528 = 9,633,792 values × 4 bytes = 38,535,168 bytes = 36.75 MiB.",
              "(d) R: 51/255 = 0.2, (0.2 − 0.485)/0.229 = −1.2445. G: 102/255 = 0.4, (0.4 − 0.456)/0.224 = −0.25. B: 204/255 = 0.8, (0.8 − 0.406)/0.225 = 1.7511.",
              "(e) 150,528·1,000 + 1,000 = 150,529,000 parameters (≈ 574 MiB in float32).",
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Dataset statistics and a verified bilinear resize",
        prompt: "Take the two scikit-learn sample photos (`load_sample_images()`), scaled to [0, 1]. (1) Compute each channel's mean and standard deviation over all images and pixels, and check that normalising with them gives every channel mean 0 and std 1. (2) Write a half-pixel bilinear resize for H×W×C arrays in NumPy and check it against `torch.nn.functional.interpolate(..., mode=\"bilinear\", align_corners=False)`, both shrinking and enlarging.",
        hint: "Average over axes (0, 1, 2) of an (N, H, W, C) array to get one number per channel. For the resize, output index i reads from source position (i + 0.5)·H/H′ − 0.5, clamped to [0, H − 1]. PyTorch's interpolate wants NCHW.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "stats_and_resize.py",
            code: String.raw`import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_sample_images


def resize_bilinear(a, out_h, out_w):
    """a: (H, W, C) float array. Half-pixel-centre bilinear interpolation."""
    h, w = a.shape[:2]
    y = np.clip((np.arange(out_h) + 0.5) * h / out_h - 0.5, 0, h - 1)
    x = np.clip((np.arange(out_w) + 0.5) * w / out_w - 0.5, 0, w - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


images = load_sample_images().images                 # two 427x640x3 uint8 photos
data = np.stack([im.astype(np.float64) / 255 for im in images])   # (N, H, W, C)

# 1) dataset statistics: average over images AND pixels, separately per channel
mean = data.mean(axis=(0, 1, 2))
std = data.std(axis=(0, 1, 2))
print("per-channel mean:", np.round(mean, 4), " std:", np.round(std, 4))
z = (data - mean) / std
print("after normalising: mean", np.round(z.mean(axis=(0, 1, 2)), 6) + 0.0, " std", np.round(z.std(axis=(0, 1, 2)), 6))

# 2) our bilinear resize vs PyTorch's (align_corners=False uses the same half-pixel rule)
for out_h, out_w in [(100, 150), (224, 224), (600, 900)]:
    ours = resize_bilinear(data[0], out_h, out_w)
    ref = F.interpolate(torch.tensor(data[:1]).permute(0, 3, 1, 2), size=(out_h, out_w),
                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    print(f"{out_h}x{out_w}: max |ours - torch| = {np.abs(ours - ref).max():.1e}")`,
            output: String.raw`per-channel mean: [0.3919 0.4295 0.3881]  std: [0.373  0.2996 0.3259]
after normalising: mean [0. 0. 0.]  std [1. 1. 1.]
100x150: max |ours - torch| = 6.0e-14
224x224: max |ours - torch| = 6.9e-14
600x900: max |ours - torch| = 7.9e-14`,
          },
          {
            t: "p",
            text: "These two photos are darker and less blue than ImageNet on average (means 0.39/0.43/0.39 vs 0.485/0.456/0.406) and more spread out — statistics belong to the dataset. The resize matches PyTorch to round-off error, shrinking and enlarging, because both use the same half-pixel mapping and clamping.",
          },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "An image is a tensor of pixel brightness values: H×W (grayscale) or H×W×3 (RGB), usually uint8 0–255 (8 bits); medical and scientific images use 12–16 bits or floats.",
          "Each channel is a 2-D plane (R, G or B; inside a CNN, a learned feature map). Other colour spaces exist (HSV, YCbCr, Lab, BGR); give a model the same space and channel order it was trained on.",
          "Layouts: HWC (files, NumPy, Keras), CHW (PyTorch); batches NCHW (PyTorch) vs NHWC (TensorFlow). `permute` reorders axes as a view; `reshape` does not. channels_last keeps the NCHW shape but stores NHWC.",
          "Preprocessing: decode → resize (keep the aspect ratio: shorter side + crop, or letterbox) → ÷255 → per-channel (x − μ)/σ with training-set statistics (ImageNet: mean 0.485/0.456/0.406, std 0.229/0.224/0.225) → batch. Augment only while training.",
          "Interpolation: bilinear/area for images (antialias when shrinking), nearest for label masks.",
          "Dense layers on images need huge numbers of parameters (150.5 M for one 1,000-unit layer on 224×224×3), ignore which pixels are neighbours (a fixed pixel shuffle changes nothing), can't reuse a detector at other positions, and fix the input size.",
          "CNNs build in locality, weight sharing and translation equivariance. Shuffling pixels hurts them, because all three rely on neighbourhoods.",
        ],
      },
      {
        t: "math",
        tex: String.raw`x = \tfrac{q}{255},\quad z_k = \tfrac{x_k - \mu_k}{\sigma_k},\quad Y = 0.299R + 0.587G + 0.114B,\quad \#\theta_{\text{dense}} = CHW\,m + m,\quad W'(P\mathbf{x}) = W\mathbf{x}\ \text{for}\ W' = WP^\top`,
        caption: "Key formulas of the lesson.",
      },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[The Convolution Operation](/learn/m08/convolution) — the operation that implements locality and weight sharing: kernels, feature maps, stride, padding, multi-channel convolution and its backward pass.",
          "[Pooling, Receptive Fields and CNN Architecture](/learn/m08/pooling-receptive-field) — how CNNs downsample, how far each neuron can “see”, what pooling really buys in translation invariance, and how a full classifier is assembled.",
          "[Regularization](/learn/m07/regularization) — data augmentation and the other techniques that let image models generalise from limited data.",
          "[Classic CNN Architectures: LeNet to ConvNeXt](/learn/m08/cnn-architectures) — the preprocessing in this lesson is exactly what these models expect at their input.",
        ],
      },
    ],
  },
};

export default lesson;
