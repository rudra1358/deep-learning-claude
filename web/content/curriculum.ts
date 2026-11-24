import type { Module, Topic, Level } from "@/lib/types";

/** Small helper to keep the curriculum table readable. */
const t = (slug: string, title: string, level: Level, summary: string, subtopics: string[]): Topic => ({
  slug,
  title,
  level,
  summary,
  subtopics,
});

export const MODULES: Module[] = [
  {
    id: "m00",
    number: 0,
    slug: "python-numpy",
    title: "Python & NumPy for Deep Learning",
    tagline: "The numerical toolkit every later lesson relies on.",
    phase: "Foundations",
    topics: [
      t("numpy-arrays", "NumPy Arrays, Shapes and Indexing", "beginner", "ndarrays, dtypes, shapes, slicing and views — the data structure behind every tensor.", ["ndarray", "dtype", "shape", "reshape", "slicing", "views vs copies", "axis"]),
      t("vectorization", "Vectorization and Broadcasting", "beginner", "Replacing Python loops with array operations, and the exact broadcasting rules.", ["vectorization", "broadcasting rules", "ufuncs", "reductions", "einsum"]),
      t("plotting-data", "Visualizing Data with Matplotlib", "beginner", "Plotting curves, scatter plots, images and training histories.", ["line plots", "scatter", "imshow", "subplots", "training curves"]),
    ],
  },
  {
    id: "m01",
    number: 1,
    slug: "introduction",
    title: "Introduction to AI and Deep Learning",
    tagline: "What deep learning is, where it came from, and why it works now.",
    phase: "Foundations",
    topics: [
      t("ai-ml-dl", "AI, Machine Learning and Deep Learning", "beginner", "How AI, ML, neural networks and DL relate, and how learning from data differs from traditional programming.", ["artificial intelligence", "machine learning", "deep learning", "traditional programming vs ML", "ML vs DL", "supervised", "unsupervised", "reinforcement learning"]),
      t("history", "History of Neural Networks", "beginner", "From McCulloch–Pitts (1943) to Transformers: the winters, the breakthroughs, and why GPUs + data changed everything.", ["McCulloch-Pitts", "perceptron", "AI winter", "backpropagation 1986", "LeNet", "ImageNet 2012", "Transformers 2017"]),
      t("dl-workflow", "The Deep Learning Workflow", "beginner", "Problem → data → model → training → evaluation → deployment, with a real-world system walkthrough.", ["problem framing", "data collection", "training", "evaluation", "deployment", "advantages", "limitations", "applications"]),
    ],
  },
  {
    id: "m02",
    number: 2,
    slug: "mathematics",
    title: "Mathematical Foundations",
    tagline: "Linear algebra, calculus, probability and optimization — with code.",
    phase: "Foundations",
    topics: [
      t("vectors-matrices", "Scalars, Vectors, Matrices and Tensors", "beginner", "The objects deep learning computes with, the operations on them, and matrix multiplication as the core of every layer.", ["scalar", "vector", "matrix", "tensor", "transpose", "dot product", "matrix multiplication", "identity matrix", "inverse matrix", "broadcasting"]),
      t("norms-spaces", "Vector Spaces, Norms and Distances", "beginner", "Length and distance in high dimensions: L1, L2, cosine similarity and why they matter for embeddings and regularization.", ["vector space", "basis", "linear independence", "L1 norm", "L2 norm", "cosine similarity", "Euclidean distance"]),
      t("eigen-svd", "Eigenvalues, Eigenvectors and SVD", "intermediate", "Matrix decompositions, what they reveal about a transformation, and their use in PCA, low-rank adaptation and stability analysis.", ["eigenvalues", "eigenvectors", "diagonalization", "singular value decomposition", "low-rank approximation", "PCA"]),
      t("derivatives-chain-rule", "Derivatives, Partial Derivatives and the Chain Rule", "beginner", "Rates of change, partial derivatives, gradients and the chain rule — the mathematics of backpropagation.", ["function", "limit", "derivative", "partial derivative", "chain rule", "gradient", "gradient vector"]),
      t("jacobians-hessians", "Jacobians, Hessians and Computational Graphs", "intermediate", "Derivatives of vector-valued functions, curvature, and how frameworks represent computations as graphs.", ["Jacobian", "Hessian", "multivariable calculus", "computational graph", "vector-Jacobian product"]),
      t("probability", "Probability, Random Variables and Distributions", "beginner", "Probability rules, Bayes' theorem, expectation, variance and the Bernoulli, binomial and Gaussian distributions.", ["probability distribution", "conditional probability", "Bayes theorem", "random variable", "expected value", "variance", "standard deviation", "Gaussian", "Bernoulli", "binomial"]),
      t("mle-map", "Maximum Likelihood and MAP Estimation", "intermediate", "Why minimizing cross-entropy is maximum likelihood, and why L2 regularization is a Gaussian prior.", ["likelihood", "log-likelihood", "maximum likelihood estimation", "maximum a posteriori", "priors"]),
      t("information-theory", "Entropy, Cross-Entropy and KL Divergence", "intermediate", "Measuring uncertainty and the difference between distributions — the basis of classification losses.", ["entropy", "cross-entropy", "KL divergence", "bits and nats"]),
      t("optimization-math", "Optimization: Objectives, Convexity and Minima", "intermediate", "Objective functions, convex vs non-convex landscapes, local/global minima, saddle points and numerical stability.", ["objective function", "convex optimization", "non-convex", "local minimum", "global minimum", "saddle point", "learning rate", "numerical stability", "log-sum-exp", "regularization"]),
    ],
  },
  {
    id: "m03",
    number: 3,
    slug: "neural-network-fundamentals",
    title: "Neural Network Fundamentals",
    tagline: "From a single neuron to multi-layer networks.",
    phase: "Neural Networks",
    topics: [
      t("linear-logistic-regression", "Linear and Logistic Regression as Neural Networks", "beginner", "The two simplest learning models — and the realization that they are one-neuron networks.", ["linear regression", "logistic regression", "decision boundary", "sigmoid", "least squares"]),
      t("artificial-neuron", "The Artificial Neuron and the Perceptron", "beginner", "Biological vs artificial neurons, McCulloch–Pitts, the perceptron learning rule, weights, biases and weighted sums.", ["biological neuron", "artificial neuron", "McCulloch-Pitts neuron", "perceptron", "weights", "bias", "weighted sum", "perceptron learning rule", "linear separability", "XOR problem"]),
      t("activation-functions", "Activation Functions", "beginner", "Why non-linearity is essential, and a full comparison from step and sigmoid to GELU, Swish, Mish and softmax.", ["binary step", "linear", "sigmoid", "tanh", "ReLU", "Leaky ReLU", "PReLU", "ELU", "SELU", "GELU", "Swish", "Mish", "softmax"]),
      t("multilayer-networks", "Multi-Layer Networks and Model Capacity", "beginner", "Input, hidden and output layers, the universal approximation theorem, parameters vs hyperparameters and capacity.", ["single-layer network", "multi-layer perceptron", "hidden layer", "universal approximation", "parameters", "hyperparameters", "model capacity"]),
      t("training-basics", "Training: Epochs, Batches and Overfitting", "beginner", "What a training loop does, epochs vs batches vs iterations, and diagnosing underfitting and overfitting.", ["epoch", "batch", "iteration", "training loop", "underfitting", "overfitting", "bias-variance trade-off"]),
    ],
  },
  {
    id: "m04",
    number: 4,
    slug: "forward-backward",
    title: "Forward Propagation and Backpropagation",
    tagline: "The complete mechanics of how a network computes and learns.",
    phase: "Neural Networks",
    topics: [
      t("forward-propagation", "Forward Propagation", "beginner", "Step-by-step computation of weighted sums and activations layer by layer, in scalar and matrix form.", ["weighted sum", "pre-activation", "activation", "matrix form", "batched forward pass", "loss calculation"]),
      t("backpropagation", "Backpropagation", "intermediate", "Deriving and computing every gradient with the chain rule, layer by layer, with full numerical examples.", ["chain rule", "computational graph", "local gradient", "upstream gradient", "weight gradients", "bias gradients", "parameter update", "gradient flow"]),
      t("gradient-problems", "Vanishing and Exploding Gradients", "intermediate", "Why gradients shrink or blow up in deep networks, how to detect it, and the remedies.", ["vanishing gradients", "exploding gradients", "saturation", "gradient norm", "remedies"]),
      t("gradient-checking", "Numerical Gradient Checking", "intermediate", "Verifying analytic gradients with finite differences — the standard debugging technique.", ["finite differences", "central difference", "relative error", "debugging backprop"]),
    ],
  },
  {
    id: "m05",
    number: 5,
    slug: "loss-functions",
    title: "Loss Functions",
    tagline: "How we measure error — and why the choice matters.",
    phase: "Neural Networks",
    topics: [
      t("regression-losses", "Regression Losses: MSE, MAE, Huber, Log-Cosh", "beginner", "Losses for continuous targets, their gradients, outlier sensitivity and when to use each.", ["mean squared error", "mean absolute error", "Huber loss", "log-cosh loss", "outliers"]),
      t("classification-losses", "Classification Losses: Cross-Entropy, Hinge, Focal", "intermediate", "Binary, categorical and sparse cross-entropy, hinge loss and focal loss for imbalance.", ["binary cross-entropy", "categorical cross-entropy", "sparse categorical cross-entropy", "hinge loss", "focal loss", "logits", "label smoothing"]),
      t("advanced-losses", "Advanced Losses: Contrastive, Triplet, Dice, IoU", "advanced", "Losses for metric learning, segmentation and generative models.", ["contrastive loss", "triplet loss", "KL divergence loss", "Dice loss", "IoU loss", "reconstruction loss"]),
    ],
  },
  {
    id: "m06",
    number: 6,
    slug: "optimization",
    title: "Optimization Algorithms",
    tagline: "How parameters actually move toward a minimum.",
    phase: "Neural Networks",
    topics: [
      t("gradient-descent", "Gradient Descent: Batch, Stochastic, Mini-Batch", "beginner", "The core update rule, the effect of learning rate, and the three ways of using data per step.", ["batch gradient descent", "stochastic gradient descent", "mini-batch", "learning rate", "convergence"]),
      t("momentum-adaptive", "Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW", "intermediate", "Optimizers that accumulate velocity or adapt per-parameter step sizes, with full update equations.", ["momentum", "Nesterov accelerated gradient", "AdaGrad", "RMSProp", "Adam", "AdamW", "Nadam", "bias correction"]),
      t("lr-schedules", "Learning Rate Schedules and Gradient Clipping", "intermediate", "Step decay, exponential decay, cosine annealing, warmup and clipping by norm or value.", ["learning rate scheduling", "warmup", "cosine annealing", "exponential decay", "step decay", "gradient clipping"]),
    ],
  },
  {
    id: "m07",
    number: 7,
    slug: "training-techniques",
    title: "Deep Networks and Training Techniques",
    tagline: "Making deep networks train reliably and generalize.",
    phase: "Neural Networks",
    topics: [
      t("initialization", "Weight Initialization: Xavier and He", "intermediate", "Why initial weight scale decides whether signals survive, and how Xavier/He are derived.", ["zero initialization", "symmetry breaking", "Xavier/Glorot initialization", "He initialization", "variance preservation"]),
      t("normalization", "Batch, Layer and Group Normalization", "intermediate", "Normalizing activations: training vs inference behaviour, where each is used, and internal covariate shift.", ["batch normalization", "layer normalization", "group normalization", "running statistics", "internal covariate shift"]),
      t("regularization", "Regularization: L1, L2, Dropout, Early Stopping", "intermediate", "Techniques that reduce overfitting and what each does to the weights and the loss.", ["L1 regularization", "L2 regularization", "weight decay", "dropout", "early stopping", "data augmentation"]),
      t("residual-connections", "Residual and Skip Connections", "intermediate", "How identity shortcuts keep gradients flowing in very deep networks.", ["residual connection", "skip connection", "identity mapping", "degradation problem", "dead ReLU", "training instability"]),
    ],
  },
  {
    id: "m08",
    number: 8,
    slug: "cnn",
    title: "Convolutional Neural Networks",
    tagline: "Learning spatial features from images.",
    phase: "Core Architectures",
    topics: [
      t("images-as-tensors", "Images as Tensors and Why CNNs", "beginner", "Pixels, channels, image tensors and why fully-connected layers fail on images.", ["pixels", "channels", "image tensor", "NCHW vs NHWC", "parameter explosion"]),
      t("convolution", "The Convolution Operation", "beginner", "Kernels sliding over pixels, feature maps, stride, padding, output-size formula and multi-channel convolution.", ["filter", "kernel", "convolution", "cross-correlation", "feature map", "stride", "padding", "valid padding", "same padding", "multi-channel convolution", "parameter sharing", "translation equivariance"]),
      t("pooling-receptive-field", "Pooling, Receptive Fields and CNN Architecture", "beginner", "Max/average/global pooling, flattening, receptive fields and putting a full CNN together.", ["max pooling", "average pooling", "global average pooling", "flattening", "fully connected layer", "receptive field"]),
      t("conv-variants", "Dilated, Transposed, Depthwise and Pointwise Convolutions", "intermediate", "The convolution variants behind segmentation networks and efficient mobile models.", ["dilated convolution", "transposed convolution", "depthwise convolution", "pointwise convolution", "depthwise separable convolution"]),
      t("cnn-architectures", "Classic CNN Architectures: LeNet to ConvNeXt", "intermediate", "The innovations of LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, MobileNet, EfficientNet, Xception and ConvNeXt.", ["LeNet", "AlexNet", "VGG", "GoogLeNet", "Inception", "ResNet", "DenseNet", "MobileNet", "EfficientNet", "Xception", "ConvNeXt"]),
    ],
  },
  {
    id: "m09",
    number: 9,
    slug: "rnn",
    title: "Recurrent Neural Networks",
    tagline: "Models with memory for sequences.",
    phase: "Core Architectures",
    topics: [
      t("rnn-basics", "Sequential Data and Vanilla RNNs", "intermediate", "Hidden states, recurrent connections and unrolling a network through time.", ["sequential data", "time series", "hidden state", "recurrent connection", "unrolling", "stateful vs stateless"]),
      t("bptt", "Backpropagation Through Time", "intermediate", "Gradients across time steps, truncated BPTT and why vanilla RNNs forget.", ["backpropagation through time", "truncated BPTT", "vanishing gradients in RNNs", "exploding gradients"]),
      t("lstm", "Long Short-Term Memory (LSTM)", "intermediate", "Forget, input and output gates, candidate and cell states — every equation, with a gate-level simulator.", ["forget gate", "input gate", "candidate cell state", "cell state update", "output gate", "hidden state", "peephole"]),
      t("gru-variants", "GRU, Bidirectional and Stacked RNNs", "intermediate", "Gated recurrent units, bidirectional and deep recurrent networks.", ["GRU", "update gate", "reset gate", "bidirectional RNN", "bidirectional LSTM", "stacked RNN"]),
    ],
  },
  {
    id: "m10",
    number: 10,
    slug: "attention",
    title: "Attention Mechanisms",
    tagline: "Letting every position look at every other position.",
    phase: "Core Architectures",
    topics: [
      t("attention-intro", "From Seq2Seq Bottlenecks to Attention", "intermediate", "Why fixed-size context vectors fail and how additive (Bahdanau) attention fixed them.", ["encoder-decoder", "context vector bottleneck", "Bahdanau attention", "alignment"]),
      t("scaled-dot-product-attention", "Queries, Keys, Values and Scaled Dot-Product Attention", "intermediate", "The attention equation computed by hand, the √d scaling argument, masks and self- vs cross-attention.", ["query", "key", "value", "attention scores", "scaled dot-product attention", "softmax normalization", "self-attention", "cross-attention", "attention mask", "causal mask"]),
      t("multi-head-attention", "Multi-Head Attention", "intermediate", "Splitting attention into heads, projecting, concatenating — and what different heads learn.", ["multi-head attention", "head dimension", "output projection"]),
      t("positional-encoding", "Positional Encoding", "intermediate", "Injecting order into permutation-invariant attention: sinusoidal, learned and rotary encodings.", ["sinusoidal positional encoding", "learned positional embeddings", "relative position", "RoPE"]),
    ],
  },
  {
    id: "m11",
    number: 11,
    slug: "transformers",
    title: "Transformers",
    tagline: "The architecture behind modern AI, built from scratch.",
    phase: "Core Architectures",
    topics: [
      t("transformer-architecture", "The Transformer Architecture", "advanced", "Encoder and decoder blocks, feed-forward layers, residuals, LayerNorm and full data flow.", ["Attention Is All You Need", "encoder", "decoder", "feed-forward network", "residual connections", "layer normalization", "encoder-decoder attention", "masked self-attention"]),
      t("tokens-embeddings", "Tokenization, Embeddings and Context Windows", "intermediate", "How text becomes token IDs and vectors, and what limits the context window.", ["tokenization", "embeddings", "context window", "vocabulary"]),
      t("training-inference", "Training, Autoregressive Generation and Complexity", "advanced", "Teacher forcing, causal language modelling, decoding and the O(n²) cost of attention.", ["autoregressive generation", "teacher forcing", "computational complexity", "training vs inference"]),
      t("transformer-families", "BERT, GPT, T5, ViT and Swin", "advanced", "Encoder-only, decoder-only and encoder-decoder families, and Transformers for vision.", ["encoder-only", "decoder-only", "encoder-decoder", "BERT", "GPT", "T5", "Vision Transformer", "Swin Transformer"]),
    ],
  },
  {
    id: "m12",
    number: 12,
    slug: "computer-vision",
    title: "Computer Vision with Deep Learning",
    tagline: "Classification, detection, segmentation and beyond.",
    phase: "Applied Deep Learning",
    topics: [
      t("transfer-learning", "Image Classification, Transfer Learning and Fine-Tuning", "intermediate", "Reusing pretrained backbones: feature extraction vs fine-tuning.", ["image classification", "transfer learning", "fine-tuning", "feature extraction", "freezing layers"]),
      t("object-detection", "Object Detection: R-CNN Family, SSD, YOLO, DETR", "advanced", "Bounding boxes, anchors, IoU, NMS and the two-stage vs one-stage vs set-prediction approaches.", ["object localization", "bounding box", "IoU", "non-max suppression", "anchor boxes", "Faster R-CNN", "SSD", "YOLO", "DETR", "mAP"]),
      t("segmentation", "Semantic, Instance and Panoptic Segmentation", "advanced", "Pixel-level prediction with FCN, U-Net, DeepLab and Mask R-CNN.", ["semantic segmentation", "instance segmentation", "panoptic segmentation", "FCN", "U-Net", "DeepLab", "Mask R-CNN"]),
      t("vision-tasks", "Faces, Pose, Optical Flow, Super-Resolution and Captioning", "advanced", "A tour of other core vision tasks and the architectures used.", ["face detection", "face recognition", "pose estimation", "optical flow", "super-resolution", "image captioning"]),
    ],
  },
  {
    id: "m13",
    number: 13,
    slug: "nlp",
    title: "Natural Language Processing",
    tagline: "From words to meaning.",
    phase: "Applied Deep Learning",
    topics: [
      t("text-preprocessing", "Text Preprocessing and Subword Tokenization", "intermediate", "Normalization, vocabularies, BPE, WordPiece and SentencePiece.", ["text preprocessing", "tokenization", "vocabulary", "subword tokenization", "BPE", "WordPiece", "SentencePiece"]),
      t("word-embeddings", "Word Embeddings: Word2Vec, GloVe, FastText", "intermediate", "Learning vector representations of words: CBOW, Skip-Gram, negative sampling.", ["word embeddings", "Word2Vec", "CBOW", "Skip-Gram", "negative sampling", "GloVe", "FastText"]),
      t("nlp-tasks", "Classification, NER, QA, Summarization, Translation", "advanced", "The major NLP tasks, how they are framed and which architectures solve them.", ["text classification", "sentiment analysis", "named entity recognition", "machine translation", "summarization", "question answering"]),
      t("language-modeling", "Language Modelling and Perplexity", "intermediate", "Predicting the next token, and evaluating language models with perplexity.", ["language modelling", "n-gram", "perplexity", "BLEU", "ROUGE"]),
    ],
  },
  {
    id: "m14",
    number: 14,
    slug: "generative",
    title: "Generative Deep Learning",
    tagline: "Autoencoders, GANs and diffusion models.",
    phase: "Generative AI",
    topics: [
      t("autoencoders", "Autoencoders", "intermediate", "Encoders, decoders and latent spaces: undercomplete, sparse, denoising and contractive variants.", ["autoencoder", "encoder", "decoder", "latent space", "undercomplete", "sparse autoencoder", "denoising autoencoder", "contractive autoencoder", "reconstruction loss"]),
      t("vae", "Variational Autoencoders", "advanced", "The ELBO, the reparameterization trick and the KL term, derived and implemented.", ["variational autoencoder", "ELBO", "reparameterization trick", "KL divergence", "latent sampling"]),
      t("gans", "Generative Adversarial Networks", "advanced", "Generator vs discriminator, the minimax objective, mode collapse, DCGAN, cGAN, WGAN(-GP), CycleGAN, Pix2Pix, StyleGAN.", ["generator", "discriminator", "minimax objective", "mode collapse", "DCGAN", "conditional GAN", "WGAN", "WGAN-GP", "CycleGAN", "Pix2Pix", "StyleGAN"]),
      t("diffusion", "Diffusion Models", "advanced", "Forward noising, reverse denoising, DDPM/DDIM, guidance and latent diffusion (Stable Diffusion).", ["forward diffusion", "reverse diffusion", "noise schedule", "noise prediction", "DDPM", "DDIM", "latent diffusion", "classifier guidance", "classifier-free guidance", "Stable Diffusion"]),
    ],
  },
  {
    id: "m15",
    number: 15,
    slug: "advanced",
    title: "Advanced Deep Learning",
    tagline: "Self-supervision, graphs, RL and multimodal models.",
    phase: "Advanced & Research",
    topics: [
      t("self-supervised", "Self-Supervised and Contrastive Learning", "advanced", "Learning without labels: SimCLR, MoCo, BYOL, semi-supervised learning.", ["self-supervised learning", "semi-supervised learning", "contrastive learning", "SimCLR", "MoCo", "BYOL"]),
      t("learning-paradigms", "Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning", "advanced", "Training paradigms beyond plain supervised learning.", ["knowledge distillation", "curriculum learning", "meta-learning", "few-shot learning", "zero-shot learning", "metric learning", "Siamese networks", "triplet networks"]),
      t("graph-neural-networks", "Graph Neural Networks", "advanced", "Message passing, GCN and GAT.", ["graph neural network", "message passing", "GCN", "GAT"]),
      t("deep-rl", "Deep Reinforcement Learning", "advanced", "DQN, policy gradients, actor-critic and PPO.", ["Markov decision process", "Deep Q-Network", "policy gradient", "actor-critic", "PPO"]),
      t("multimodal", "Multimodal Models, CLIP and RAG Foundations", "advanced", "Joint embedding spaces, vision-language models and retrieval-augmented generation.", ["multimodal learning", "vision-language models", "CLIP", "retrieval-augmented generation", "Neural ODEs"]),
    ],
  },
  {
    id: "m16",
    number: 16,
    slug: "frameworks",
    title: "Frameworks and Implementation",
    tagline: "NumPy from scratch, PyTorch and TensorFlow/Keras.",
    phase: "Engineering & Deployment",
    topics: [
      t("numpy-from-scratch", "A Complete Neural Network in NumPy", "intermediate", "A modular from-scratch library: layers, losses, optimizers, training loop.", ["layers", "forward", "backward", "optimizers", "training loop"]),
      t("pytorch", "PyTorch: Tensors to Distributed Training", "intermediate", "Tensors, autograd, nn.Module, Dataset/DataLoader, GPU, AMP, checkpointing.", ["tensors", "autograd", "nn.Module", "Dataset", "DataLoader", "GPU", "mixed precision", "checkpointing", "distributed training"]),
      t("tensorflow-keras", "TensorFlow and Keras", "intermediate", "Sequential and functional APIs, custom layers, GradientTape, callbacks — and when to choose which framework.", ["Sequential API", "Functional API", "custom layers", "GradientTape", "callbacks", "SavedModel", "PyTorch vs TensorFlow"]),
    ],
  },
  {
    id: "m17",
    number: 17,
    slug: "evaluation",
    title: "Model Evaluation and Experimentation",
    tagline: "Measuring models honestly.",
    phase: "Engineering & Deployment",
    topics: [
      t("data-splits", "Train/Validation/Test Splits and Cross-Validation", "beginner", "Holding out data correctly and avoiding leakage.", ["training set", "validation set", "test set", "cross-validation", "data leakage"]),
      t("classification-metrics", "Classification Metrics, ROC and PR Curves", "intermediate", "Confusion matrix, precision, recall, F1, ROC/AUC, PR curves, calibration and imbalance.", ["accuracy", "precision", "recall", "F1 score", "confusion matrix", "ROC curve", "AUC", "PR curve", "calibration", "class imbalance"]),
      t("regression-metrics", "Regression Metrics", "beginner", "MAE, MSE, RMSE, R².", ["MAE", "MSE", "RMSE", "R-squared"]),
      t("experimentation", "Hyperparameter Tuning, Tracking and Reproducibility", "intermediate", "Grid/random/Bayesian search, experiment tracking, seeds, ablations and error analysis.", ["hyperparameter tuning", "grid search", "random search", "Bayesian optimization", "experiment tracking", "reproducibility", "random seeds", "ablation studies", "error analysis"]),
    ],
  },
  {
    id: "m18",
    number: 18,
    slug: "deployment",
    title: "Deployment and MLOps",
    tagline: "Taking models to production.",
    phase: "Engineering & Deployment",
    topics: [
      t("serving", "Serialization, Inference Pipelines and APIs", "intermediate", "Saving models, TorchScript/ONNX export, FastAPI/Flask serving, Docker and GPU deployment.", ["model serialization", "inference pipeline", "REST API", "FastAPI", "Flask", "Docker", "ONNX", "TorchScript", "GPU deployment"]),
      t("model-optimization", "Quantization, Pruning and Distillation", "advanced", "Making models smaller and faster with measured accuracy trade-offs.", ["quantization", "pruning", "knowledge distillation", "latency", "throughput"]),
      t("mlops", "Monitoring, Drift, CI/CD and Versioning", "intermediate", "Keeping models healthy in production: drift, monitoring, MLflow, W&B, CI/CD and cost.", ["model monitoring", "data drift", "concept drift", "CI/CD for ML", "MLflow", "Weights & Biases", "model versioning", "cloud deployment", "scalability", "cost optimization"]),
    ],
  },
  {
    id: "m19",
    number: 19,
    slug: "research",
    title: "Research-Level Deep Learning",
    tagline: "Reading, reproducing and extending research.",
    phase: "Advanced & Research",
    topics: [
      t("reading-papers", "Reading and Reproducing Research Papers", "advanced", "How to read papers, decode notation, reproduce results and benchmark fairly.", ["reading papers", "mathematical notation", "reproduction", "benchmarking", "ablation studies", "novel architecture design"]),
      t("scaling-efficiency", "Scaling Laws, LoRA, QLoRA and PEFT", "research", "How performance scales with compute, and parameter-efficient fine-tuning.", ["scaling laws", "compute efficiency", "parameter efficiency", "LoRA", "QLoRA", "PEFT"]),
      t("interpretability", "Interpretability: Saliency, Grad-CAM, SHAP, LIME", "advanced", "Explaining model predictions.", ["interpretability", "explainable AI", "saliency maps", "Grad-CAM", "SHAP", "LIME"]),
      t("robustness-safety", "Adversarial Examples, Robustness and AI Safety", "advanced", "FGSM/PGD attacks, robustness, safety and responsible AI.", ["adversarial examples", "FGSM", "PGD", "robustness", "AI safety", "responsible AI"]),
    ],
  },
  {
    id: "m20",
    number: 20,
    slug: "llms",
    title: "Large Language Models",
    tagline: "How modern LLMs are trained, aligned and served.",
    phase: "Generative AI",
    topics: [
      t("llm-pretraining", "Pretraining Large Language Models", "advanced", "Data, objectives, scaling and the training recipe of modern decoder-only LLMs.", ["pretraining", "next-token prediction", "data mixtures", "compute-optimal training"]),
      t("decoding", "Decoding: Greedy, Beam, Temperature, Top-k, Top-p", "intermediate", "How a probability distribution over tokens becomes text.", ["greedy decoding", "beam search", "temperature", "top-k sampling", "top-p sampling"]),
      t("alignment", "Instruction Tuning, RLHF and DPO", "research", "Turning a base model into an assistant.", ["instruction tuning", "supervised fine-tuning", "reward model", "RLHF", "DPO"]),
      t("llm-inference", "Efficient Inference: KV Cache, FlashAttention, MoE", "research", "The engineering that makes LLMs fast: KV caching, FlashAttention, mixture-of-experts and quantization.", ["KV cache", "FlashAttention", "mixture of experts", "speculative decoding", "LLM quantization"]),
    ],
  },
];

export const PHASES = [
  "Foundations",
  "Neural Networks",
  "Core Architectures",
  "Applied Deep Learning",
  "Generative AI",
  "Advanced & Research",
  "Engineering & Deployment",
] as const;

/** Linear learning order (used for previous/next navigation). */
export const LEARNING_ORDER: string[] = MODULES.flatMap((m) => m.topics.map((tp) => `${m.id}/${tp.slug}`));

export function getModule(id: string) {
  return MODULES.find((m) => m.id === id);
}

export function getTopic(topicId: string) {
  const [mid, slug] = topicId.split("/");
  const mod = getModule(mid);
  const topic = mod?.topics.find((tp) => tp.slug === slug);
  return mod && topic ? { module: mod, topic } : undefined;
}

export const TOTAL_TOPICS = LEARNING_ORDER.length;
