# Interview questions on this project — with answers

Answer in your own words; these are model answers to learn from. Numbers you quote must come from
[RESULTS.md](RESULTS.md).

## Project & design

**1. Walk me through your project.**
Business problem (counting/tracking objects in camera feeds) → data (COCO baseline, VisDrone fine-tuning,
EDA showed mostly tiny objects) → model (YOLO11 n vs s, Faster R-CNN reference) → evaluation (mAP,
per-class, by size, domain gap before/after) → export to ONNX with a parity check → serving pipeline
I wrote myself (letterbox, NMS, ByteTrack tracker, line counting) → FastAPI (HTTP + WebSocket,
background video jobs) → PostgreSQL → Next.js dashboard → Docker + CI. Close with one limitation and
what you'd do next.

**2. Why YOLO and not Faster R-CNN?**
Real-time requirement. YOLO is one-stage: one forward pass for all boxes. Faster R-CNN first proposes
regions then classifies each, which is slower. I measured both on the same 500 COCO images; quote the
latency and mAP rows from RESULTS.md. Faster R-CNN would make sense for offline, accuracy-critical work.

**3. Why did you fine-tune? How did you prove it helped?**
Domain shift: top-down drone views, tiny objects, classes missing from COCO (van, tricycle). I evaluated
the COCO model on VisDrone (only on shared classes so it is fair) and then the fine-tuned model on the
same classes and images — the before/after numbers are in RESULTS.md.

**4. How did you choose between nano and small?**
Accuracy vs latency on my target hardware. Small has more capacity (better mAP, especially on small
objects) but is slower. For a live webcam at ≥ 15 FPS on a laptop GPU, both qualify; I'd pick the one
whose latency leaves headroom for tracking and the network — explain with the measured table.

**5. What does your system do when the server is slower than the camera?**
WebSocket back-pressure: the client sends a new frame only after the previous result arrives, so the
effective FPS drops but latency stays bounded; no queue of stale frames.

## Detection fundamentals

**6. What is IoU? What is NMS and why is it needed?**
IoU = intersection / union of two boxes. A detector fires several overlapping boxes for the same
object; NMS keeps the highest-scoring box and removes others with IoU above a threshold (0.45 here).
I do it per class (offset trick) so a rider and their bicycle both survive.

**7. Explain mAP@0.5:0.95.**
For each class: sort predictions by confidence, compute precision/recall at each step, AP = area under
the PR curve. mAP averages over classes; @0.5:0.95 also averages over IoU thresholds 0.5…0.95, so it
rewards precise localisation. Evaluate with a very low confidence threshold because AP integrates over
all thresholds.

**8. Precision vs recall — which matters more here?**
Depends on the use case. Counting cars for traffic statistics: balanced (F1). Safety alert "person in
restricted zone": recall (missing a person is worse than a false alarm). The confidence threshold is the
knob; choose it from the PR curve for the operating point the business needs.

**9. What is letterboxing and why not just resize to 640×640?**
Resize keeping aspect ratio and pad. Plain resize distorts shapes compared to training. Remember the
scale and padding to map boxes back to the original image.

**10. Why does YOLO output 8400 boxes?**
Three feature maps at strides 8/16/32 on a 640 input: 80² + 40² + 20² = 8400 locations, one candidate
each (anchor-free). Stride 8 handles small objects, stride 32 large ones.

**11. Why is small-object detection hard and how would you improve it?**
After downsampling, a 10-px object covers about one cell of the finest grid; little information
survives. Improvements: higher input resolution, tiled inference (SAHI), an extra P2 (stride-4) head,
more small-object augmentation (copy-paste) — each costs latency.

## Tracking

**12. Explain your tracker.**
Kalman filter predicts each track's box; cost = 1 − IoU between predictions and detections; Hungarian
algorithm finds the best one-to-one assignment; ByteTrack's two rounds (high-score detections first,
then low-score ones for unmatched tracks); tentative → confirmed → lost (30 frames) → removed.

**13. What does the Kalman filter actually do?**
Keeps a state (position, size, velocities) and its uncertainty. Predict step moves it with constant
velocity and increases uncertainty. Update step blends the prediction with the measured box using the
Kalman gain = how much more we trust the measurement than the prediction. It smooths jittery boxes and
lets a track coast through short occlusions.

**14. What is the Hungarian algorithm and why not greedy matching?**
Solves the assignment problem: minimum total cost one-to-one matching, O(n³). Greedy (match the best
pair first) can make a locally good choice that forces a bad match elsewhere, causing ID switches in
crowds.

**15. What is ByteTrack's main contribution?**
Using low-confidence detections in a second association round instead of discarding them; occluded
objects often get low scores, so this reduces broken tracks and ID switches.

**16. When does your tracker fail?**
Very fast motion (IoU between prediction and detection is 0), long occlusions (> 30 frames), people
crossing paths (no appearance features). Fix: re-ID appearance embeddings (DeepSORT/BoT-SORT), camera
motion compensation.

**17. How would you evaluate the tracker?**
MOT metrics on a dataset with track IDs (e.g. VisDrone-MOT): MOTA (misses + false positives + ID
switches), IDF1 (how long IDs stay correct), HOTA (balances detection and association). Not done yet
because VisDrone-DET has no IDs — say this honestly.

**18. How do you count line crossings without double counting?**
Side of the line from the sign of a cross product; count when the centre changes side *and* the movement
segment intersects the line segment; each track ID counted at most once per direction.

## Deployment & engineering

**19. Why ONNX Runtime? How do you know the export is correct?**
Smaller serving image (no PyTorch), graph optimisations, hardware portability via execution providers.
Parity check: same images through PyTorch and my ONNX pipeline, match boxes by IoU; report match rate and
max score difference.

**20. FP16 / INT8 — what are they and did they help?**
FP16 halves memory traffic and uses tensor cores on GPU; INT8 stores weights in 8 bits. Dynamic INT8
quantises weights ahead of time and activations at runtime (no calibration); static INT8 needs a
calibration set but is faster. Quote the measured latencies — if one didn't help on your hardware, say
so and explain why (e.g. dynamic quantisation of convolutions adds overhead on CPU).

**21. How does the video endpoint work and how would you scale it?**
Stream upload to disk, insert a `queued` run, return 202, worker thread processes and updates progress,
client polls. Scale: Celery/RQ + Redis queue, N GPU worker containers, object storage (S3) for videos,
batch frames for throughput, autoscale on queue length.

**22. How did you test code that depends on a neural network?**
Unit-test pure math (IoU, NMS, Kalman, tracker, line logic) with hand-computed expectations. For the API,
generate a tiny fake ONNX model with the real I/O signature and known output → deterministic
end-to-end tests in CI without weights or GPU.

**23. SQLite vs PostgreSQL in your project?**
Same SQLAlchemy models; URL from env var. SQLite for zero-setup dev/tests, PostgreSQL in Docker for
concurrency and production-style aggregation queries.

**24. What would you monitor in production?**
Latency/FPS, error rates, GPU utilisation; model health: confidence-score distribution, class frequency,
detections per frame over time (drift — e.g. a camera moved, night vs day); periodically label a sample
and recompute mAP.

**25. What would you improve with two more weeks?**
Pick from section 12 of HOW_IT_WAS_BUILT.md: re-ID tracker, MOT evaluation, higher-res / tiled inference
for small objects, TensorRT, job queue.

**26. Tell me about a hard bug you fixed in this project.**
Use one story from section 11 of HOW_IT_WAS_BUILT.md, structured as symptom → diagnosis → fix → what you
changed so it can't happen silently again. The best one: ONNX Runtime silently falling back to CPU because
of a CUDA 12 vs 13 library mismatch — found by checking `session.get_providers()` and the ORT log, fixed by
installing matching runtime wheels, hardened by making the detector warn when the provider differs.
