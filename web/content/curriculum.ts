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
    tagline: "The number-crunching tools every later lesson uses.",
    phase: "Foundations",
    topics: [
      t("numpy-arrays", "NumPy Arrays, Shapes and Indexing", "beginner", "Arrays, data types, shapes, slicing and views — the building block behind every tensor.", ["ndarray", "dtype", "shape", "reshape", "slicing", "views vs copies", "axis"]),
      t("vectorization", "Vectorization and Broadcasting", "beginner", "Swapping slow Python loops for whole-array math, and exactly how broadcasting works.", ["vectorization", "broadcasting rules", "ufuncs", "reductions", "einsum"]),
      t("plotting-data", "Visualizing Data with Matplotlib", "beginner", "Drawing curves, scatter plots, images and training progress.", ["line plots", "scatter", "imshow", "subplots", "training curves"]),
    ],
  },
  {
    id: "m01",
    number: 1,
    slug: "introduction",
    title: "Introduction to AI and Deep Learning",
    tagline: "What deep learning is, where it came from, and why it works today.",
    phase: "Foundations",
    topics: [
      t("ai-ml-dl", "AI, Machine Learning and Deep Learning", "beginner", "How AI, ML, neural networks and DL fit together, and how learning from data differs from writing rules by hand.", ["artificial intelligence", "machine learning", "deep learning", "traditional programming vs ML", "ML vs DL", "supervised", "unsupervised", "reinforcement learning"]),
      t("history", "History of Neural Networks", "beginner", "From McCulloch–Pitts (1943) to Transformers: the slow years, the big wins, and why GPUs + data changed everything.", ["McCulloch-Pitts", "perceptron", "AI winter", "backpropagation 1986", "LeNet", "ImageNet 2012", "Transformers 2017"]),
      t("dl-workflow", "The Deep Learning Workflow", "beginner", "Problem → data → model → training → testing → deployment, walked through on a real system.", ["problem framing", "data collection", "training", "evaluation", "deployment", "advantages", "limitations", "applications"]),
    ],
  },
  {
    id: "m02",
    number: 2,
    slug: "mathematics",
    title: "Math Foundations",
    tagline: "Linear algebra, calculus, probability and optimization — with code.",
    phase: "Foundations",
    topics: [
      t("vectors-matrices", "Scalars, Vectors, Matrices and Tensors", "beginner", "The objects deep learning works with, what you can do with them, and why matrix multiplication sits inside every layer.", ["scalar", "vector", "matrix", "tensor", "transpose", "dot product", "matrix multiplication", "identity matrix", "inverse matrix", "broadcasting"]),
      t("norms-spaces", "Vector Spaces, Norms and Distances", "beginner", "How long a vector is and how far apart two are: L1, L2, cosine similarity, and why they matter for embeddings and regularization.", ["vector space", "basis", "linear independence", "L1 norm", "L2 norm", "cosine similarity", "Euclidean distance"]),
      t("eigen-svd", "Eigenvalues, Eigenvectors and SVD", "intermediate", "Breaking a matrix into simpler pieces, what that tells you, and how it powers PCA, low-rank adaptation and stability checks.", ["eigenvalues", "eigenvectors", "diagonalization", "singular value decomposition", "low-rank approximation", "PCA"]),
      t("derivatives-chain-rule", "Derivatives, Partial Derivatives and the Chain Rule", "beginner", "How fast things change: derivatives, partial derivatives, gradients and the chain rule — the math behind backpropagation.", ["function", "limit", "derivative", "partial derivative", "chain rule", "gradient", "gradient vector"]),
      t("jacobians-hessians", "Jacobians, Hessians and Computational Graphs", "intermediate", "Derivatives when the output is a vector, curvature, and how frameworks turn your code into a graph of steps.", ["Jacobian", "Hessian", "multivariable calculus", "computational graph", "vector-Jacobian product"]),
      t("probability", "Probability, Random Variables and Distributions", "beginner", "The rules of chance, Bayes' theorem, average and spread (expectation, variance), and the Bernoulli, binomial and Gaussian distributions.", ["probability distribution", "conditional probability", "Bayes theorem", "random variable", "expected value", "variance", "standard deviation", "Gaussian", "Bernoulli", "binomial"]),
      t("mle-map", "Maximum Likelihood and MAP Estimation", "intermediate", "Why minimizing cross-entropy is maximum likelihood, and why L2 regularization is a Gaussian prior.", ["likelihood", "log-likelihood", "maximum likelihood estimation", "maximum a posteriori", "priors"]),
      t("information-theory", "Entropy, Cross-Entropy and KL Divergence", "intermediate", "Measuring uncertainty and how far apart two distributions are — the idea behind classification losses.", ["entropy", "cross-entropy", "KL divergence", "bits and nats"]),
      t("optimization-math", "Optimization: Objectives, Convexity and Minima", "intermediate", "What we minimize, bowl-shaped vs bumpy loss surfaces, local and global minima, saddle points, and avoiding number overflow.", ["objective function", "convex optimization", "non-convex", "local minimum", "global minimum", "saddle point", "learning rate", "numerical stability", "log-sum-exp", "regularization"]),
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
      t("linear-logistic-regression", "Linear and Logistic Regression as Neural Networks", "beginner", "The two simplest learning models — and why each is really a one-neuron network.", ["linear regression", "logistic regression", "decision boundary", "sigmoid", "least squares"]),
      t("artificial-neuron", "The Artificial Neuron and the Perceptron", "beginner", "Real vs artificial neurons, McCulloch–Pitts, weights, biases, weighted sums, and how the perceptron learns.", ["biological neuron", "artificial neuron", "McCulloch-Pitts neuron", "perceptron", "weights", "bias", "weighted sum", "perceptron learning rule", "linear separability", "XOR problem"]),
      t("activation-functions", "Activation Functions", "beginner", "Why networks need bends, not just straight lines, and a side-by-side look from step and sigmoid to GELU, Swish, Mish and softmax.", ["binary step", "linear", "sigmoid", "tanh", "ReLU", "Leaky ReLU", "PReLU", "ELU", "SELU", "GELU", "Swish", "Mish", "softmax"]),
      t("multilayer-networks", "Multi-Layer Networks and Model Capacity", "beginner", "Input, hidden and output layers, why a network can fit almost any function, parameters vs hyperparameters, and capacity.", ["single-layer network", "multi-layer perceptron", "hidden layer", "universal approximation", "parameters", "hyperparameters", "model capacity"]),
      t("training-basics", "Training: Epochs, Batches and Overfitting", "beginner", "What a training loop does, epochs vs batches vs iterations, and how to spot underfitting and overfitting.", ["epoch", "batch", "iteration", "training loop", "underfitting", "overfitting", "bias-variance trade-off"]),
    ],
  },
  {
    id: "m04",
    number: 4,
    slug: "forward-backward",
    title: "Forward Propagation and Backpropagation",
    tagline: "Exactly how a network makes a prediction and learns from it.",
    phase: "Neural Networks",
    topics: [
      t("forward-propagation", "Forward Propagation", "beginner", "Follow the numbers layer by layer: weighted sums and activations, one value at a time and as matrices.", ["weighted sum", "pre-activation", "activation", "matrix form", "batched forward pass", "loss calculation"]),
      t("backpropagation", "Backpropagation", "intermediate", "Working out every gradient with the chain rule, layer by layer, with real numbers at every step.", ["chain rule", "computational graph", "local gradient", "upstream gradient", "weight gradients", "bias gradients", "parameter update", "gradient flow"]),
      t("gradient-problems", "Vanishing and Exploding Gradients", "intermediate", "Why gradients shrink or blow up in deep networks, how to spot it, and how to fix it.", ["vanishing gradients", "exploding gradients", "saturation", "gradient norm", "remedies"]),
      t("gradient-checking", "Numerical Gradient Checking", "intermediate", "Checking your gradients by nudging each weight a little — the standard way to debug backprop.", ["finite differences", "central difference", "relative error", "debugging backprop"]),
    ],
  },
  {
    id: "m05",
    number: 5,
    slug: "loss-functions",
    title: "Loss Functions",
    tagline: "How we measure mistakes — and why the choice matters.",
    phase: "Neural Networks",
    topics: [
      t("regression-losses", "Regression Losses: MSE, MAE, Huber, Log-Cosh", "beginner", "Losses for predicting numbers: their gradients, how outliers affect them, and when to use each.", ["mean squared error", "mean absolute error", "Huber loss", "log-cosh loss", "outliers"]),
      t("classification-losses", "Classification Losses: Cross-Entropy, Hinge, Focal", "intermediate", "Binary, categorical and sparse cross-entropy, hinge loss, and focal loss for rare classes.", ["binary cross-entropy", "categorical cross-entropy", "sparse categorical cross-entropy", "hinge loss", "focal loss", "logits", "label smoothing"]),
      t("advanced-losses", "Advanced Losses: Contrastive, Triplet, Dice, IoU", "advanced", "Losses for learning similarity, outlining objects (segmentation) and generative models.", ["contrastive loss", "triplet loss", "KL divergence loss", "Dice loss", "IoU loss", "reconstruction loss"]),
    ],
  },
  {
    id: "m06",
    number: 6,
    slug: "optimization",
    title: "Optimization Algorithms",
    tagline: "How the weights actually move downhill to a minimum.",
    phase: "Neural Networks",
    topics: [
      t("gradient-descent", "Gradient Descent: Batch, Stochastic, Mini-Batch", "beginner", "The basic update rule, what the learning rate does, and three ways to use data in each step.", ["batch gradient descent", "stochastic gradient descent", "mini-batch", "learning rate", "convergence"]),
      t("momentum-adaptive", "Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW", "intermediate", "Optimizers that build up speed or give each weight its own step size, with every update equation.", ["momentum", "Nesterov accelerated gradient", "AdaGrad", "RMSProp", "Adam", "AdamW", "Nadam", "bias correction"]),
      t("lr-schedules", "Learning Rate Schedules and Gradient Clipping", "intermediate", "Changing the learning rate over time (step, exponential, cosine, warmup) and capping large gradients.", ["learning rate scheduling", "warmup", "cosine annealing", "exponential decay", "step decay", "gradient clipping"]),
    ],
  },
  {
    id: "m07",
    number: 7,
    slug: "training-techniques",
    title: "Deep Networks and Training Techniques",
    tagline: "Getting deep networks to train reliably and work on new data.",
    phase: "Neural Networks",
    topics: [
      t("initialization", "Weight Initialization: Xavier and He", "intermediate", "Why the starting size of the weights decides whether signals survive, and where Xavier/He come from.", ["zero initialization", "symmetry breaking", "Xavier/Glorot initialization", "He initialization", "variance preservation"]),
      t("normalization", "Batch, Layer and Group Normalization", "intermediate", "Rescaling activations: how training and inference differ, where each type is used, and internal covariate shift.", ["batch normalization", "layer normalization", "group normalization", "running statistics", "internal covariate shift"]),
      t("regularization", "Regularization: L1, L2, Dropout, Early Stopping", "intermediate", "Ways to reduce overfitting, and what each one does to the weights and the loss.", ["L1 regularization", "L2 regularization", "weight decay", "dropout", "early stopping", "data augmentation"]),
      t("residual-connections", "Residual and Skip Connections", "intermediate", "How shortcuts that skip layers keep gradients flowing in very deep networks.", ["residual connection", "skip connection", "identity mapping", "degradation problem", "dead ReLU", "training instability"]),
    ],
  },
  {
    id: "m08",
    number: 8,
    slug: "cnn",
    title: "Convolutional Neural Networks",
    tagline: "Learning to spot patterns in images.",
    phase: "Core Architectures",
    topics: [
      t("images-as-tensors", "Images as Tensors and Why CNNs", "beginner", "Pixels, channels, image tensors, and why plain fully-connected layers struggle with images.", ["pixels", "channels", "image tensor", "NCHW vs NHWC", "parameter explosion"]),
      t("convolution", "The Convolution Operation", "beginner", "Small filters sliding over pixels, feature maps, stride, padding, the output-size formula and multi-channel convolution.", ["filter", "kernel", "convolution", "cross-correlation", "feature map", "stride", "padding", "valid padding", "same padding", "multi-channel convolution", "parameter sharing", "translation equivariance"]),
      t("pooling-receptive-field", "Pooling, Receptive Fields and CNN Architecture", "beginner", "Max/average/global pooling, flattening, how much of the image each neuron sees, and building a full CNN.", ["max pooling", "average pooling", "global average pooling", "flattening", "fully connected layer", "receptive field"]),
      t("conv-variants", "Dilated, Transposed, Depthwise and Pointwise Convolutions", "intermediate", "The convolution variants behind segmentation networks and fast models for phones.", ["dilated convolution", "transposed convolution", "depthwise convolution", "pointwise convolution", "depthwise separable convolution"]),
      t("cnn-architectures", "Classic CNN Architectures: LeNet to ConvNeXt", "intermediate", "The big new idea in LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, MobileNet, EfficientNet, Xception and ConvNeXt.", ["LeNet", "AlexNet", "VGG", "GoogLeNet", "Inception", "ResNet", "DenseNet", "MobileNet", "EfficientNet", "Xception", "ConvNeXt"]),
    ],
  },
  {
    id: "m09",
    number: 9,
    slug: "rnn",
    title: "Recurrent Neural Networks",
    tagline: "Models with memory, for data that comes in order.",
    phase: "Core Architectures",
    topics: [
      t("rnn-basics", "Sequential Data and Vanilla RNNs", "intermediate", "Hidden states, connections that loop back, and unrolling a network through time.", ["sequential data", "time series", "hidden state", "recurrent connection", "unrolling", "stateful vs stateless"]),
      t("bptt", "Backpropagation Through Time", "intermediate", "Gradients across time steps, truncated BPTT, and why plain RNNs forget.", ["backpropagation through time", "truncated BPTT", "vanishing gradients in RNNs", "exploding gradients"]),
      t("lstm", "Long Short-Term Memory (LSTM)", "intermediate", "Forget, input and output gates, candidate and cell states — every equation, with a simulator for each gate.", ["forget gate", "input gate", "candidate cell state", "cell state update", "output gate", "hidden state", "peephole"]),
      t("gru-variants", "GRU, Bidirectional and Stacked RNNs", "intermediate", "Gated recurrent units, RNNs that read both ways, and stacked (deep) RNNs.", ["GRU", "update gate", "reset gate", "bidirectional RNN", "bidirectional LSTM", "stacked RNN"]),
    ],
  },
  {
    id: "m10",
    number: 10,
    slug: "attention",
    title: "Attention Mechanisms",
    tagline: "Letting every word look at every other word.",
    phase: "Core Architectures",
    topics: [
      t("attention-intro", "From Seq2Seq Bottlenecks to Attention", "intermediate", "Why squeezing a sentence into one fixed vector fails, and how Bahdanau attention fixed it.", ["encoder-decoder", "context vector bottleneck", "Bahdanau attention", "alignment"]),
      t("scaled-dot-product-attention", "Queries, Keys, Values and Scaled Dot-Product Attention", "intermediate", "The attention equation worked by hand, why we divide by √d, masks, and self- vs cross-attention.", ["query", "key", "value", "attention scores", "scaled dot-product attention", "softmax normalization", "self-attention", "cross-attention", "attention mask", "causal mask"]),
      t("multi-head-attention", "Multi-Head Attention", "intermediate", "Splitting attention into several heads, joining them back — and what each head learns.", ["multi-head attention", "head dimension", "output projection"]),
      t("positional-encoding", "Positional Encoding", "intermediate", "Attention ignores word order, so we add it back: sinusoidal, learned and rotary encodings.", ["sinusoidal positional encoding", "learned positional embeddings", "relative position", "RoPE"]),
    ],
  },
  {
    id: "m11",
    number: 11,
    slug: "transformers",
    title: "Transformers",
    tagline: "The design behind modern AI, built from scratch.",
    phase: "Core Architectures",
    topics: [
      t("transformer-architecture", "The Transformer Architecture", "advanced", "Encoder and decoder blocks, feed-forward layers, residuals, LayerNorm, and how data flows through.", ["Attention Is All You Need", "encoder", "decoder", "feed-forward network", "residual connections", "layer normalization", "encoder-decoder attention", "masked self-attention"]),
      t("tokens-embeddings", "Tokenization, Embeddings and Context Windows", "intermediate", "How text becomes token IDs and vectors, and what limits how much the model can read at once.", ["tokenization", "embeddings", "context window", "vocabulary"]),
      t("training-inference", "Training, Autoregressive Generation and Complexity", "advanced", "Teacher forcing, next-token prediction, generating text, and the O(n²) cost of attention.", ["autoregressive generation", "teacher forcing", "computational complexity", "training vs inference"]),
      t("transformer-families", "BERT, GPT, T5, ViT and Swin", "advanced", "Encoder-only, decoder-only and encoder-decoder models, and Transformers for images.", ["encoder-only", "decoder-only", "encoder-decoder", "BERT", "GPT", "T5", "Vision Transformer", "Swin Transformer"]),
    ],
  },
  {
    id: "m12",
    number: 12,
    slug: "computer-vision",
    title: "Computer Vision with Deep Learning",
    tagline: "Naming, finding and outlining things in images.",
    phase: "Applied Deep Learning",
    topics: [
      t("transfer-learning", "Image Classification, Transfer Learning and Fine-Tuning", "intermediate", "Reusing an already-trained model: feature extraction vs fine-tuning.", ["image classification", "transfer learning", "fine-tuning", "feature extraction", "freezing layers"]),
      t("object-detection", "Object Detection: R-CNN Family, SSD, YOLO, DETR", "advanced", "Bounding boxes, anchors, IoU, NMS, and the two-stage, one-stage and set-prediction approaches.", ["object localization", "bounding box", "IoU", "non-max suppression", "anchor boxes", "Faster R-CNN", "SSD", "YOLO", "DETR", "mAP"]),
      t("segmentation", "Semantic, Instance and Panoptic Segmentation", "advanced", "Labeling every pixel with FCN, U-Net, DeepLab and Mask R-CNN.", ["semantic segmentation", "instance segmentation", "panoptic segmentation", "FCN", "U-Net", "DeepLab", "Mask R-CNN"]),
      t("vision-tasks", "Faces, Pose, Optical Flow, Super-Resolution and Captioning", "advanced", "A tour of other key vision tasks and the models used for them.", ["face detection", "face recognition", "pose estimation", "optical flow", "super-resolution", "image captioning"]),
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
      t("text-preprocessing", "Text Preprocessing and Subword Tokenization", "intermediate", "Cleaning text, building vocabularies, and BPE, WordPiece and SentencePiece.", ["text preprocessing", "tokenization", "vocabulary", "subword tokenization", "BPE", "WordPiece", "SentencePiece"]),
      t("word-embeddings", "Word Embeddings: Word2Vec, GloVe, FastText", "intermediate", "Turning words into meaningful vectors: CBOW, Skip-Gram, negative sampling.", ["word embeddings", "Word2Vec", "CBOW", "Skip-Gram", "negative sampling", "GloVe", "FastText"]),
      t("nlp-tasks", "Classification, NER, QA, Summarization, Translation", "advanced", "The main NLP tasks, how each is set up, and which models solve them.", ["text classification", "sentiment analysis", "named entity recognition", "machine translation", "summarization", "question answering"]),
      t("language-modeling", "Language Modelling and Perplexity", "intermediate", "Predicting the next token, and scoring language models with perplexity.", ["language modelling", "n-gram", "perplexity", "BLEU", "ROUGE"]),
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
      t("autoencoders", "Autoencoders", "intermediate", "Squeeze data down, then rebuild it: encoders, decoders, latent spaces, and the main variants.", ["autoencoder", "encoder", "decoder", "latent space", "undercomplete", "sparse autoencoder", "denoising autoencoder", "contractive autoencoder", "reconstruction loss"]),
      t("vae", "Variational Autoencoders", "advanced", "The ELBO, the reparameterization trick and the KL term, worked out and coded.", ["variational autoencoder", "ELBO", "reparameterization trick", "KL divergence", "latent sampling"]),
      t("gans", "Generative Adversarial Networks", "advanced", "A generator vs a discriminator, the minimax game, mode collapse, DCGAN, cGAN, WGAN(-GP), CycleGAN, Pix2Pix, StyleGAN.", ["generator", "discriminator", "minimax objective", "mode collapse", "DCGAN", "conditional GAN", "WGAN", "WGAN-GP", "CycleGAN", "Pix2Pix", "StyleGAN"]),
      t("diffusion", "Diffusion Models", "advanced", "Add noise, then learn to remove it: DDPM/DDIM, guidance and latent diffusion (Stable Diffusion).", ["forward diffusion", "reverse diffusion", "noise schedule", "noise prediction", "DDPM", "DDIM", "latent diffusion", "classifier guidance", "classifier-free guidance", "Stable Diffusion"]),
    ],
  },
  {
    id: "m15",
    number: 15,
    slug: "advanced",
    title: "Advanced Deep Learning",
    tagline: "Learning without labels, graphs, RL, and text + image models.",
    phase: "Advanced & Research",
    topics: [
      t("self-supervised", "Self-Supervised and Contrastive Learning", "advanced", "Learning without labels: SimCLR, MoCo, BYOL, semi-supervised learning.", ["self-supervised learning", "semi-supervised learning", "contrastive learning", "SimCLR", "MoCo", "BYOL"]),
      t("learning-paradigms", "Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning", "advanced", "Ways to train beyond plain supervised learning.", ["knowledge distillation", "curriculum learning", "meta-learning", "few-shot learning", "zero-shot learning", "metric learning", "Siamese networks", "triplet networks"]),
      t("graph-neural-networks", "Graph Neural Networks", "advanced", "Networks for graph data: message passing, GCN and GAT.", ["graph neural network", "message passing", "GCN", "GAT"]),
      t("deep-rl", "Deep Reinforcement Learning", "advanced", "Learning from rewards: DQN, policy gradients, actor-critic and PPO.", ["Markov decision process", "Deep Q-Network", "policy gradient", "actor-critic", "PPO"]),
      t("multimodal", "Multimodal Models, CLIP and RAG Foundations", "advanced", "One shared space for images and text, vision-language models, and retrieval-augmented generation.", ["multimodal learning", "vision-language models", "CLIP", "retrieval-augmented generation", "Neural ODEs"]),
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
      t("numpy-from-scratch", "A Complete Neural Network in NumPy", "intermediate", "Build a small library from scratch: layers, losses, optimizers, training loop.", ["layers", "forward", "backward", "optimizers", "training loop"]),
      t("pytorch", "PyTorch: Tensors to Distributed Training", "intermediate", "Tensors, autograd, nn.Module, Dataset/DataLoader, GPU, AMP, checkpointing.", ["tensors", "autograd", "nn.Module", "Dataset", "DataLoader", "GPU", "mixed precision", "checkpointing", "distributed training"]),
      t("tensorflow-keras", "TensorFlow and Keras", "intermediate", "Sequential and functional APIs, custom layers, GradientTape, callbacks — and how to pick a framework.", ["Sequential API", "Functional API", "custom layers", "GradientTape", "callbacks", "SavedModel", "PyTorch vs TensorFlow"]),
    ],
  },
  {
    id: "m17",
    number: 17,
    slug: "evaluation",
    title: "Model Evaluation and Experimentation",
    tagline: "Measuring how good a model really is.",
    phase: "Engineering & Deployment",
    topics: [
      t("data-splits", "Train/Validation/Test Splits and Cross-Validation", "beginner", "Setting data aside correctly, and stopping test data leaking into training.", ["training set", "validation set", "test set", "cross-validation", "data leakage"]),
      t("classification-metrics", "Classification Metrics, ROC and PR Curves", "intermediate", "Confusion matrix, precision, recall, F1, ROC/AUC, PR curves, calibration and rare classes.", ["accuracy", "precision", "recall", "F1 score", "confusion matrix", "ROC curve", "AUC", "PR curve", "calibration", "class imbalance"]),
      t("regression-metrics", "Regression Metrics", "beginner", "MAE, MSE, RMSE, R².", ["MAE", "MSE", "RMSE", "R-squared"]),
      t("experimentation", "Hyperparameter Tuning, Tracking and Reproducibility", "intermediate", "Grid/random/Bayesian search, tracking experiments, seeds, ablations and error analysis.", ["hyperparameter tuning", "grid search", "random search", "Bayesian optimization", "experiment tracking", "reproducibility", "random seeds", "ablation studies", "error analysis"]),
    ],
  },
  {
    id: "m18",
    number: 18,
    slug: "deployment",
    title: "Deployment and MLOps",
    tagline: "Putting models into real use.",
    phase: "Engineering & Deployment",
    topics: [
      t("serving", "Serialization, Inference Pipelines and APIs", "intermediate", "Saving models, TorchScript/ONNX export, serving with FastAPI/Flask, Docker and GPUs.", ["model serialization", "inference pipeline", "REST API", "FastAPI", "Flask", "Docker", "ONNX", "TorchScript", "GPU deployment"]),
      t("model-optimization", "Quantization, Pruning and Distillation", "advanced", "Making models smaller and faster, and measuring the accuracy you give up.", ["quantization", "pruning", "knowledge distillation", "latency", "throughput"]),
      t("mlops", "Monitoring, Drift, CI/CD and Versioning", "intermediate", "Keeping live models healthy: drift, monitoring, MLflow, W&B, CI/CD and cost.", ["model monitoring", "data drift", "concept drift", "CI/CD for ML", "MLflow", "Weights & Biases", "model versioning", "cloud deployment", "scalability", "cost optimization"]),
    ],
  },
  {
    id: "m19",
    number: 19,
    slug: "research",
    title: "Deep Learning Research",
    tagline: "Reading, reproducing and building on research.",
    phase: "Advanced & Research",
    topics: [
      t("reading-papers", "Reading and Reproducing Research Papers", "advanced", "How to read papers, decode the notation, reproduce results and compare fairly.", ["reading papers", "mathematical notation", "reproduction", "benchmarking", "ablation studies", "novel architecture design"]),
      t("scaling-efficiency", "Scaling Laws, LoRA, QLoRA and PEFT", "research", "How results grow with compute, and fine-tuning only a few weights.", ["scaling laws", "compute efficiency", "parameter efficiency", "LoRA", "QLoRA", "PEFT"]),
      t("interpretability", "Interpretability: Saliency, Grad-CAM, SHAP, LIME", "advanced", "Seeing why a model predicted what it did.", ["interpretability", "explainable AI", "saliency maps", "Grad-CAM", "SHAP", "LIME"]),
      t("robustness-safety", "Adversarial Examples, Robustness and AI Safety", "advanced", "Fooling models (FGSM/PGD), making them sturdier, and responsible AI.", ["adversarial examples", "FGSM", "PGD", "robustness", "AI safety", "responsible AI"]),
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
      t("llm-pretraining", "Pretraining Large Language Models", "advanced", "The data, goal, scale and training recipe of modern decoder-only LLMs.", ["pretraining", "next-token prediction", "data mixtures", "compute-optimal training"]),
      t("decoding", "Decoding: Greedy, Beam, Temperature, Top-k, Top-p", "intermediate", "How next-token probabilities become actual text.", ["greedy decoding", "beam search", "temperature", "top-k sampling", "top-p sampling"]),
      t("alignment", "Instruction Tuning, RLHF and DPO", "research", "Turning a raw base model into a helpful assistant.", ["instruction tuning", "supervised fine-tuning", "reward model", "RLHF", "DPO"]),
      t("llm-inference", "Efficient Inference: KV Cache, FlashAttention, MoE", "research", "The tricks that make LLMs fast: KV caching, FlashAttention, mixture-of-experts and quantization.", ["KV cache", "FlashAttention", "mixture of experts", "speculative decoding", "LLM quantization"]),
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
