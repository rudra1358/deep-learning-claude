export interface Project {
  id: string;
  short: string;
  title: string;
  domain: string;
  summary: string;
  problem: string;
  useCases: string[];
  requirements: { functional: string[]; nonFunctional: string[] };
  architecture: string[]; // data-flow steps
  stack: { layer: string; choice: string; why: string }[];
  data: string[];
  models: { name: string; why: string }[];
  milestones: { id: string; title: string; detail: string }[];
  tree: string;
  prerequisites: string[]; // topic ids
  hardware: string;
}

export const PROJECTS: Project[] = [
  {
    id: "vision",
    short: "Real-time visual analysis",
    title: "Real-Time Object Detection & Visual Analytics System",
    domain: "Computer Vision",
    summary: "Find, follow and count objects in images, videos and live webcam feeds, with a stats dashboard and a side-by-side model comparison.",
    problem:
      "Shops, warehouses and traffic teams want to know what a camera sees and how it changes over time — how many of each thing, how long they stay, who crosses a line — without a person watching every screen. It must run almost in real time on one home GPU, or on a CPU at lower resolution.",
    useCases: ["Footfall and occupancy counting", "Vehicle counting at intersections", "Shelf / inventory monitoring", "Safety: detecting people in restricted zones"],
    requirements: {
      functional: ["Upload image / video, or stream from webcam", "Bounding boxes with class + confidence", "Multi-object tracking with persistent IDs", "Per-class counts over time, line-crossing counts", "Detection history with search & export", "Side-by-side comparison of two model sizes", "Metrics page: mAP on a validation set, latency, FPS"],
      nonFunctional: ["≥ 15 FPS at 640 px on a mid-range GPU; documented CPU fallback", "No hard-coded predictions — every box comes from model inference", "Reproducible training (seeded, config-file driven)", "Containerized; one-command local start"],
    },
    architecture: [
      "Frontend (Next.js) captures frames (webcam via getUserMedia) or uploads files",
      "FastAPI backend receives frames over WebSocket (stream) or HTTP (files)",
      "Preprocess: letterbox resize to 640×640, normalize, NCHW tensor",
      "Detector (YOLO-family, PyTorch → ONNX Runtime) produces raw boxes",
      "Postprocess: confidence threshold + class-wise non-max suppression",
      "Tracker (ByteTrack-style association by IoU + Kalman filter) assigns IDs",
      "Analytics service aggregates counts / crossings; results stored in PostgreSQL",
      "Dashboard reads aggregates; history page queries stored detections",
    ],
    stack: [
      { layer: "Model", choice: "Ultralytics YOLO (pretrained on COCO), fine-tuned on a domain dataset", why: "Accurate yet fast, well-documented training, and easy ONNX export." },
      { layer: "Inference", choice: "ONNX Runtime (CUDA or CPU provider)", why: "Runs the model without needing PyTorch, and is faster on CPU than plain PyTorch." },
      { layer: "Backend", choice: "FastAPI + WebSockets, Pydantic, SQLAlchemy", why: "Handles live streams well, checks data types, and writes API docs for you." },
      { layer: "Database", choice: "PostgreSQL", why: "Easy queries over detections as they happen over time." },
      { layer: "Frontend", choice: "Next.js + Canvas overlay", why: "Draws boxes on top of video frames quickly." },
      { layer: "Ops", choice: "Docker Compose, pytest, GitHub Actions", why: "Same setup on every machine, plus automatic tests." },
    ],
    data: ["COCO 2017 (pretrained weights; val split for baseline mAP)", "A domain dataset for fine-tuning (e.g. a public vehicle or retail dataset), converted to YOLO format", "EDA: class balance, box-size distribution, image resolution"],
    models: [
      { name: "YOLO nano vs small", why: "Weigh accuracy (mAP@0.5:0.95) against speed to pick the model to ship." },
      { name: "Faster R-CNN (torchvision) baseline", why: "A two-stage detector to compare against on accuracy vs speed." },
    ],
    milestones: [
      { id: "eda", title: "Dataset acquisition & EDA", detail: "Download, convert to YOLO format, and look at class and box statistics." },
      { id: "baseline", title: "Baseline evaluation", detail: "Test the ready-made COCO model on the validation set; note mAP and speed." },
      { id: "finetune", title: "Fine-tuning pipeline", detail: "Training set up from a config file, with augmentation, early stopping and experiment tracking." },
      { id: "error-analysis", title: "Error analysis", detail: "AP for each class, which similar classes get mixed up, and where small objects are missed." },
      { id: "export", title: "ONNX export & optimization", detail: "Export, check the outputs still match, and measure FP16/INT8 speed-ups." },
      { id: "tracking", title: "Tracking & analytics", detail: "IoU + Kalman tracker, a line-crossing counter, and summary queries." },
      { id: "api", title: "Backend API", detail: "Upload, stream and history endpoints with tests." },
      { id: "frontend", title: "Frontend dashboard", detail: "Live boxes on video, charts, history and model comparison views." },
      { id: "deploy", title: "Docker deployment", detail: "Compose file for API, DB and frontend; GPU and CPU profiles." },
    ],
    tree: `vision-analytics/
├── backend/
│   ├── app/            # FastAPI: main.py, routers/, schemas.py, db.py
│   ├── inference/      # detector.py (ONNX), postprocess.py (NMS), tracker.py
│   └── tests/
├── training/
│   ├── configs/        # dataset.yaml, train.yaml
│   ├── prepare_data.py # download + convert + EDA report
│   ├── train.py  evaluate.py  export_onnx.py
├── frontend/           # Next.js dashboard
├── docker-compose.yml
└── README.md`,
    prerequisites: ["m08/convolution", "m08/cnn-architectures", "m12/transfer-learning", "m12/object-detection", "m17/classification-metrics", "m18/serving"],
    hardware: "Fine-tuning: one GPU with ≥ 8 GB (or a free cloud GPU notebook). Running the model: works on CPU, just at fewer frames per second.",
  },
  {
    id: "nlp",
    short: "Document intelligence",
    title: "Context-Aware Document Intelligence Platform",
    domain: "NLP & Transformers",
    summary: "Upload PDFs, then search them by meaning, ask questions and get answers with sources, summarize, pull out names and sort documents.",
    problem:
      "Companies have thousands of contracts, papers and reports, and keyword search handles them badly. People need answers that come from the documents, with exact sources — not generic chatbot text.",
    useCases: ["Legal / contract review", "Research literature Q&A", "Internal knowledge-base search", "Compliance document triage"],
    requirements: {
      functional: ["PDF / DOCX / TXT upload and text extraction", "Chunking with overlap; page-level source mapping", "Semantic search over embeddings", "Question answering with cited passages", "Abstractive summarization per document", "Named entity recognition and document classification", "Chat interface with conversation history"],
      nonFunctional: ["Every answer shows its source chunks", "Retrieval quality measured (recall@k, MRR) on a labelled set", "Works fully offline with open-weight models; optional hosted LLM"],
    },
    architecture: [
      "Upload → text extraction (PyMuPDF) with page numbers",
      "Cleaning + sentence-aware chunking (≈300–500 tokens, overlap)",
      "Sentence-Transformer encoder embeds chunks → vector index (FAISS / pgvector)",
      "Query embedded with the same encoder → top-k cosine similarity retrieval",
      "Cross-encoder re-ranks candidates",
      "Extractive QA model or generator LLM answers from retrieved context with citations",
      "Separate pipelines: summarization (seq2seq), NER (token classification), classification (fine-tuned encoder)",
    ],
    stack: [
      { layer: "Embeddings", choice: "Sentence-Transformers (MiniLM / MPNet class)", why: "Finds text by meaning well, yet is small enough to run on CPU." },
      { layer: "Vector store", choice: "FAISS (local) or PostgreSQL + pgvector", why: "Finds the closest vectors, exactly or approximately." },
      { layer: "QA / generation", choice: "Extractive QA (SQuAD-fine-tuned encoder) and an open-weight instruction LLM", why: "Answers copied from the text are easy to check; generation gives smooth summaries." },
      { layer: "NER / classification", choice: "Fine-tuned BERT-family encoder (Hugging Face Transformers)", why: "One pretrained encoder with small heads for labeling words and whole documents." },
      { layer: "Backend", choice: "FastAPI, background workers for ingestion", why: "Slow embedding jobs run in the background, so the app stays fast." },
      { layer: "Frontend", choice: "Next.js chat + document viewer with highlighted sources", why: "Answers you can check against the source." },
    ],
    data: ["User-uploaded documents", "SQuAD 2.0 for QA evaluation", "CoNLL-2003 for NER fine-tuning/evaluation", "A labelled query → passage set built from sample documents for retrieval metrics"],
    models: [
      { name: "Bi-encoder retrieval + cross-encoder re-ranking", why: "Bi-encoders are fast at scale (vectors computed ahead of time); cross-encoders are more accurate on a short list." },
      { name: "Fine-tuned encoder for NER/classification", why: "Teaches transfer learning with Transformers; easy to measure with F1." },
    ],
    milestones: [
      { id: "ingest", title: "Ingestion & chunking", detail: "Pull out text with page numbers; try different ways to split it into chunks." },
      { id: "embed", title: "Embeddings & index", detail: "Turn chunks into vectors, build a FAISS/pgvector index, measure recall@k." },
      { id: "rerank", title: "Re-ranking", detail: "Re-rank with a cross-encoder; compare MRR before and after." },
      { id: "qa", title: "Question answering with citations", detail: "Answers copied or generated from the retrieved chunks only." },
      { id: "ner", title: "NER & classification fine-tuning", detail: "Fine-tune, measure F1, study the mistakes." },
      { id: "summarize", title: "Summarization", detail: "Summarize long documents piece by piece, then combine; score with ROUGE." },
      { id: "api", title: "Backend API & workers", detail: "Endpoints, background ingestion, tests." },
      { id: "frontend", title: "Chat & document viewer", detail: "Answers with sources you can click to see highlighted." },
      { id: "deploy", title: "Deployment", detail: "Docker Compose; model caching; resource limits." },
    ],
    tree: `doc-intelligence/
├── backend/
│   ├── app/            # main.py, routers/ (docs, search, qa, chat), db.py
│   ├── ingest/         # extract.py, chunk.py, embed.py, index.py
│   ├── nlp/            # qa.py, summarize.py, ner.py, classify.py, rerank.py
│   └── tests/
├── training/           # finetune_ner.py, finetune_classifier.py, eval_retrieval.py
├── frontend/
├── docker-compose.yml
└── README.md`,
    prerequisites: ["m10/scaled-dot-product-attention", "m11/transformer-architecture", "m11/transformer-families", "m13/text-preprocessing", "m13/nlp-tasks", "m15/multimodal"],
    hardware: "Search and extractive QA run on CPU. Fine-tuning NER/classifiers: a small GPU or a free cloud notebook. Generative LLM: a GPU with ≥ 8 GB for a small quantized model, or a hosted API.",
  },
  {
    id: "generative",
    short: "Image generation studio",
    title: "Diffusion-Based Image Generation & Editing Studio",
    domain: "Generative AI",
    summary: "Make images from text, change existing images, repaint parts of a picture and explore the latent space, with saved history and settings.",
    problem:
      "Designers want to try visual ideas fast — make an image from a prompt, vary an existing one, edit one area — while seeing and controlling what the model does (steps, guidance, seed, scheduler).",
    useCases: ["Concept art and mood boards", "Product mock-ups and variations", "Image editing via inpainting", "Teaching how diffusion works by exposing its internals"],
    requirements: {
      functional: ["Text-to-image with prompt / negative prompt", "Image-to-image with strength control", "Inpainting with a drawn mask", "Settings: steps, guidance scale, seed, scheduler, size", "Show intermediate denoising steps (decoded latents)", "Latent interpolation between two seeds (slerp)", "Generation history with parameters; download"],
      nonFunctional: ["Clearly labelled real inference (no canned images)", "Queueing so one GPU serves requests sequentially", "Content safety filtering enabled"],
    },
    architecture: [
      "Prompt → CLIP text encoder → token embeddings",
      "Random latent z_T ~ N(0, I) from seed (64×64×4 for 512² images)",
      "U-Net predicts noise ε at each step, conditioned via cross-attention on text embeddings",
      "Classifier-free guidance: ε = ε_uncond + s·(ε_cond − ε_uncond)",
      "Scheduler (DDIM / DPM-Solver) updates z_t → z_{t−1}",
      "VAE decoder maps final latent to pixels",
      "img2img / inpainting: start from a noised encoding of the input image (masked)",
    ],
    stack: [
      { layer: "Model", choice: "Stable Diffusion (latent diffusion) via Hugging Face Diffusers", why: "Free-to-download weights, and every part (scheduler, U-Net, VAE, text encoder) is open to inspect." },
      { layer: "From-scratch component", choice: "A small DDPM trained on MNIST / Fashion-MNIST in PyTorch", why: "You build the noising process, the noise-prediction loss and the sampler yourself." },
      { layer: "Backend", choice: "FastAPI + job queue, PostgreSQL for history, object storage for images", why: "Handles slow GPU jobs and keeps the results." },
      { layer: "Frontend", choice: "Next.js studio with canvas mask editor", why: "Draw inpainting masks and watch step-by-step previews." },
    ],
    data: ["Pretrained Stable Diffusion weights (license must be accepted by the user)", "MNIST / Fashion-MNIST for the from-scratch DDPM"],
    models: [
      { name: "From-scratch DDPM (small U-Net)", why: "Shows the full training goal and sampling loop on a small, easy dataset." },
      { name: "Pretrained latent diffusion", why: "Real, high-quality results — and we explain each part instead of treating it as a black box." },
    ],
    milestones: [
      { id: "ddpm", title: "DDPM from scratch", detail: "Noise schedule, noising process, U-Net, ε-prediction loss and a sampler, on MNIST." },
      { id: "ddim", title: "DDIM sampling", detail: "Faster sampling with fewer, non-random steps; compare quality vs steps." },
      { id: "sd-pipeline", title: "Latent diffusion pipeline", detail: "Load each part on its own and write the denoising loop by hand." },
      { id: "cfg", title: "Classifier-free guidance study", detail: "What guidance scale, seeds and schedulers change." },
      { id: "img2img", title: "Image-to-image & inpainting", detail: "The strength setting and masked latents." },
      { id: "latent", title: "Latent exploration", detail: "Blend between two seeds (slerp), and preview the in-between steps." },
      { id: "api", title: "Backend & job queue", detail: "Generation endpoints, history, tests." },
      { id: "frontend", title: "Studio UI", detail: "Prompt panel, mask editor, history gallery." },
      { id: "deploy", title: "Deployment", detail: "GPU container, model caching, safety checker." },
    ],
    tree: `diffusion-studio/
├── backend/
│   ├── app/            # main.py, routers/ (generate, history), queue.py, db.py
│   ├── pipelines/      # txt2img.py, img2img.py, inpaint.py, latents.py
│   └── tests/
├── ddpm_from_scratch/  # schedule.py, unet.py, train.py, sample.py
├── frontend/
├── docker-compose.yml
└── README.md`,
    prerequisites: ["m14/autoencoders", "m14/vae", "m14/diffusion", "m10/scaled-dot-product-attention", "m15/multimodal", "m18/serving"],
    hardware: "From-scratch DDPM: slow on CPU, minutes on a GPU. Running Stable Diffusion: a GPU with ≥ 6–8 GB VRAM (or a cloud GPU); CPU works but takes minutes per image.",
  },
];
