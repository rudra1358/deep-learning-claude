### Error analysis (read from the yolo11s-visdrone normalised confusion matrix; columns = true class)

1. **Missed small objects dominate.** 41 % of pedestrians, 44 % of "people" and 45 % of bicycles end up
   as *background* (not detected at all), versus 14 % of cars. These are the smallest objects in the
   dataset (68.6 % of all boxes are < 32×32 px), which is also why AP_small (9.6) is a fraction of
   AP_large (48.4). → Biggest lever: higher input resolution or tiled inference (SAHI), a stride-4 (P2) head.
2. **Look-alike vehicles from above.** 51 % of vans are predicted as *car* — more often than as van (32 %);
   26 % of trucks → car, 16 % of buses → truck. From a top-down view the cues that separate them (height,
   side windows) are mostly invisible. → More van/truck data, or merge classes if the business question
   only needs "vehicle".
3. **Two-wheel / three-wheel confusion.** bicycle → motor 21 %, tricycle → motor 20 %,
   awning-tricycle → tricycle 17 %. Rare classes (awning-tricycle has 3,246 training boxes vs 144,867 cars)
   also get the lowest AP. → Class-balanced sampling or loss weighting.
4. **pedestrian vs people** (VisDrone's "walking" vs "standing/sitting" person): 14 % of *people* are labelled
   *pedestrian*. The difference is posture, which is hard to see at a few pixels tall; for counting use-cases
   the two can be merged into "person" (as the domain-gap comparison does).

### Optimisation findings (section 5)

* **GPU FP32 vs FP16:** FP16 was *not* faster for the nano models (19.0 → 21.6 ms) and only modestly faster for
  the small models (25.6 → 23.1 ms). At batch 1 the nano model is too small to be compute-bound — time goes to
  kernel launches, memory copies and CPU-side pre/post-processing (~6–8 ms) — so halving the arithmetic barely
  matters. FP16 pays off for bigger models / batches.
* **CPU dynamic INT8 was *slower*** than FP32 (e.g. 55.6 → 72.6 ms) because dynamic quantisation re-quantises
  activations at run time and ONNX Runtime falls back to slower kernels for many quantised conv patterns. Static
  INT8 with a calibration set (or OpenVINO on Intel CPUs) is the right tool — measured, not assumed.
* **Parity:** an early check matched only 92 % of boxes. The cause was the *reference*: Ultralytics' `predict()`
  letterboxes to a minimal rectangle (`auto=True`) while our static ONNX model needs a 640×640 square, so the two
  sides saw different pixels. Feeding both identical input gives 100 % box matches and raw-output differences
  below 0.008 px — the export and our post-processing are exact.
