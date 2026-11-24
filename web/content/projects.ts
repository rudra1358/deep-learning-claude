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
    summary: "Detect, track and count objects in images, video and live webcam streams, with an analytics dashboard and model comparison.",
    problem:
      "Retail stores, warehouses and traffic operators need to know what is in a camera's view and how it changes over time — counts per class, dwell time, line crossings — without a person watching every feed. The system must run in near real time on a single consumer GPU or a CPU at reduced resolution.",
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
      { layer: "Model", choice: "Ultralytics YOLO (pretrained on COCO), fine-tuned on a domain dataset", why: "Strong accuracy/speed trade-off, well-documented training pipeline, ONNX export." },
      { layer: "Inference", choice: "ONNX Runtime (CUDA or CPU provider)", why: "Framework-independent serving, faster CPU inference than eager PyTorch." },
      { layer: "Backend", choice: "FastAPI + WebSockets, Pydantic, SQLAlchemy", why: "Async I/O for streaming, typed schemas, auto OpenAPI docs." },
      { layer: "Database", choice: "PostgreSQL", why: "Time-series style queries over detection events." },
      { layer: "Frontend", choice: "Next.js + Canvas overlay", why: "Draw boxes over video frames efficiently." },
      { layer: "Ops", choice: "Docker Compose, pytest, GitHub Actions", why: "Reproducible environment and CI." },
    ],
    data: ["COCO 2017 (pretrained weights; val split for baseline mAP)", "A domain dataset for fine-tuning (e.g. a public vehicle or retail dataset), converted to YOLO format", "EDA: class balance, box-size distribution, image resolution"],
    models: [
      { name: "YOLO nano vs small", why: "Compare accuracy (mAP@0.5:0.95) against latency to choose the deployment model." },
      { name: "Faster R-CNN (torchvision) baseline", why: "Two-stage reference point for accuracy vs speed discussion." },
    ],
    milestones: [
      { id: "eda", title: "Dataset acquisition & EDA", detail: "Download, convert to YOLO format, analyse class and box statistics." },
      { id: "baseline", title: "Baseline evaluation", detail: "Evaluate pretrained COCO weights on the validation split; record mAP and latency." },
      { id: "finetune", title: "Fine-tuning pipeline", detail: "Config-driven training with augmentation, early stopping, experiment tracking." },
      { id: "error-analysis", title: "Error analysis", detail: "Per-class AP, confusion of similar classes, small-object failure cases." },
      { id: "export", title: "ONNX export & optimization", detail: "Export, verify numerical parity, measure FP16/INT8 speed-ups." },
      { id: "tracking", title: "Tracking & analytics", detail: "IoU + Kalman tracker, line-crossing counter, aggregation queries." },
      { id: "api", title: "Backend API", detail: "Upload, stream and history endpoints with tests." },
      { id: "frontend", title: "Frontend dashboard", detail: "Live overlay, charts, history and model comparison views." },
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
    hardware: "Fine-tuning: a single GPU with ≥ 8 GB (or a free cloud GPU notebook). Inference: runs on CPU at reduced FPS.",
  },
  {
    id: "nlp",
    short: "Document intelligence",
    title: "Context-Aware Document Intelligence Platform",
    domain: "NLP & Transformers",
    summary: "Upload PDFs, then search them semantically, ask questions with cited answers, summarize, extract entities and classify documents.",
    problem:
      "Organizations hold thousands of contracts, papers and reports that keyword search handles poorly: users need answers grounded in the documents, with exact source references, not generic chatbot text.",
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
      { layer: "Embeddings", choice: "Sentence-Transformers (MiniLM / MPNet class)", why: "Strong semantic retrieval at small size; runs on CPU." },
      { layer: "Vector store", choice: "FAISS (local) or PostgreSQL + pgvector", why: "Exact or approximate nearest neighbour search." },
      { layer: "QA / generation", choice: "Extractive QA (SQuAD-fine-tuned encoder) and an open-weight instruction LLM", why: "Extractive answers are verifiable; generation for fluent summaries." },
      { layer: "NER / classification", choice: "Fine-tuned BERT-family encoder (Hugging Face Transformers)", why: "Token- and sequence-classification heads on a shared pretrained encoder." },
      { layer: "Backend", choice: "FastAPI, background workers for ingestion", why: "Long-running embedding jobs off the request path." },
      { layer: "Frontend", choice: "Next.js chat + document viewer with highlighted sources", why: "Grounded, inspectable answers." },
    ],
    data: ["User-uploaded documents", "SQuAD 2.0 for QA evaluation", "CoNLL-2003 for NER fine-tuning/evaluation", "A labelled query → passage set built from sample documents for retrieval metrics"],
    models: [
      { name: "Bi-encoder retrieval + cross-encoder re-ranking", why: "Bi-encoders scale (precomputed vectors); cross-encoders are more accurate on a short list." },
      { name: "Fine-tuned encoder for NER/classification", why: "Teaches transfer learning on Transformers; measurable with F1." },
    ],
    milestones: [
      { id: "ingest", title: "Ingestion & chunking", detail: "Extraction with page mapping; chunking strategy experiments." },
      { id: "embed", title: "Embeddings & index", detail: "Encode chunks, build FAISS/pgvector index, measure recall@k." },
      { id: "rerank", title: "Re-ranking", detail: "Cross-encoder re-ranking; compare MRR before/after." },
      { id: "qa", title: "Question answering with citations", detail: "Extractive and generative answering grounded in retrieved chunks." },
      { id: "ner", title: "NER & classification fine-tuning", detail: "Fine-tune, evaluate F1, error analysis." },
      { id: "summarize", title: "Summarization", detail: "Map-reduce summarization of long documents; ROUGE evaluation." },
      { id: "api", title: "Backend API & workers", detail: "Endpoints, background ingestion, tests." },
      { id: "frontend", title: "Chat & document viewer", detail: "Answer view with clickable source highlights." },
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
    hardware: "Retrieval and extractive QA run on CPU. Fine-tuning NER/classifiers: a small GPU or free cloud notebook. Generative LLM: GPU with ≥ 8 GB for a quantized small model, or a hosted API.",
  },
  {
    id: "generative",
    short: "Image generation studio",
    title: "Diffusion-Based Image Generation & Editing Studio",
    domain: "Generative AI",
    summary: "Text-to-image, image-to-image, inpainting and latent-space exploration on a latent diffusion model, with generation history and settings.",
    problem:
      "Designers want to iterate quickly on visual concepts — generate from a prompt, vary an existing image, edit a region — while understanding and controlling what the model does (steps, guidance, seed, scheduler).",
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
      { layer: "Model", choice: "Stable Diffusion (latent diffusion) via Hugging Face Diffusers", why: "Open weights, modular pipeline exposing scheduler, U-Net, VAE and text encoder." },
      { layer: "From-scratch component", choice: "A small DDPM trained on MNIST / Fashion-MNIST in PyTorch", why: "Learners implement the forward process, noise-prediction loss and sampler themselves." },
      { layer: "Backend", choice: "FastAPI + job queue, PostgreSQL for history, object storage for images", why: "Long-running GPU jobs, persisted results." },
      { layer: "Frontend", choice: "Next.js studio with canvas mask editor", why: "Inpainting masks and step-by-step previews." },
    ],
    data: ["Pretrained Stable Diffusion weights (license must be accepted by the user)", "MNIST / Fashion-MNIST for the from-scratch DDPM"],
    models: [
      { name: "From-scratch DDPM (small U-Net)", why: "Shows the full training objective and sampling loop on a tractable dataset." },
      { name: "Pretrained latent diffusion", why: "Production-quality generation; explain each component rather than treat it as a black box." },
    ],
    milestones: [
      { id: "ddpm", title: "DDPM from scratch", detail: "Noise schedule, forward process, U-Net, ε-prediction loss, sampler on MNIST." },
      { id: "ddim", title: "DDIM sampling", detail: "Deterministic fewer-step sampling; compare quality vs steps." },
      { id: "sd-pipeline", title: "Latent diffusion pipeline", detail: "Load components separately and write the denoising loop by hand." },
      { id: "cfg", title: "Classifier-free guidance study", detail: "Effect of guidance scale; seeds; schedulers." },
      { id: "img2img", title: "Image-to-image & inpainting", detail: "Strength parameter, masked latents." },
      { id: "latent", title: "Latent exploration", detail: "Slerp interpolation, intermediate step previews." },
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
    hardware: "From-scratch DDPM: trains on CPU slowly, minutes on a GPU. Stable Diffusion inference: GPU with ≥ 6–8 GB VRAM (or a cloud GPU); CPU inference works but takes minutes per image.",
  },
];
