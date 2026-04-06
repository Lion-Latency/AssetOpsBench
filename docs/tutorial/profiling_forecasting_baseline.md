# Profiling Baseline — TSFM Forecasting Pipeline

## 1. Overview
This document summarizes the initial profiling results for the TSFM MCP server forecasting pipeline.

The goal of this experiment is to:
- Validate the per-stage profiling instrumentation
- Establish a baseline latency breakdown
- Identify potential bottlenecks for optimization

---

## 2. Experimental Setup

- **Model:** TinyTimeMixer (TTM)
- **Workflow:** Forecasting
- **Dataset:** Synthetic data (for debugging and validation)
- **Environment:** Google Cloud VM (NVIDIA L4 GPU)
- **Repository:** AssetOpsBench (TSFM MCP server)
- **Runs:** 5 (1 cold start + 4 steady-state)

---

## 3. Pipeline Stages Profiled

The following stages were instrumented using `time.perf_counter`:

1. Data Loading
2. Preprocessing
3. Model Loading
4. Trainer Setup
5. Inference
6. Postprocessing / Serialization

---

## 4. Profiling Results (Steady-State Average)

| Stage                         | Latency (seconds) |
|------------------------------|------------------|
| Data Loading                 | ~0.0015 s        |
| Preprocessing                | ~0.0250 s        |
| Model Loading                | ~0.0154 s        |
| Trainer Setup                | ~0.0350 s        |
| Inference                    | ~0.1930 s        |
| Postprocessing / Serialization | ~0.2275 s     |

> Note: Values are averaged across steady-state runs (Runs 2–5).

---

## 5. End-to-End Latency

- **Cold-start latency:** ~4.01 s  
- **Steady-state latency (avg):** ~0.517 s :contentReference[oaicite:0]{index=0}  

---

## 6. Key Findings

- **Postprocessing / serialization is the dominant stage**, exceeding inference time.
- Inference (~0.19s) is relatively efficient compared to output handling (~0.23s).
- Cold-start latency is significantly higher due to initialization overhead.

---

## 7. Interpretation

- The pipeline is **not compute-bound** (model inference is not the bottleneck).
- Instead, it is **system-bound**, with overhead coming from:
  - output formatting
  - serialization
  - potential disk I/O

---

## 8. Hypotheses

Potential causes of the postprocessing bottleneck:
- JSON/CSV serialization overhead
- Disk I/O latency when writing outputs
- Inefficient data conversion or formatting

---

## 9. Limitations

- Results are based on synthetic data, which may not fully reflect real-world workloads.
- Only the forecasting pipeline has been profiled so far.
- Variance observed in preprocessing suggests input-dependent behavior.

---

## 10. Next Steps

- Validate results using AssetOpsBench datasets
- Extend profiling to:
  - Anomaly detection
  - Fine-tuning workflows
- Investigate optimization strategies:
  - Reduce serialization overhead
  - Optimize output formatting
- Build benchmarking harness for systematic comparisons

---

## 11. Notes

This baseline will be used for comparison against future optimizations.