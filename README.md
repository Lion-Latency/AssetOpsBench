# HPML Final Project: Profiling and Optimizing the TSFM MCP Server

> **Course:** High Performance Machine Learning  
> **Semester:** Spring 2026  
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Lion Latency
- **Members:**
  - Sally Go (yg3066) — External benchmarking harness, W&B logging, benchmarking
  - Sam Colman (sc5750) — Internal instrumentation, preprocessing caching
  - Byeolah Kwon (bk2833) — Profiling, preprocessing parallelism, benchmarking
  - Tomas Pasiecznik (tp2758) — Environment setup, interchangeable model interface

---

## Submission

- **GitHub repository:** https://github.com/Lion-Latency/AssetOpsBench
- **Final report:** `deliverables/HPML_Final_Report.pdf`
- **Final presentation:** `deliverables/HPML_Final_Presentation.pptx`
- **Experiment-tracking dashboard:** https://wandb.ai/lion-latency/hpml-project

The final report PDF and presentation file are checked into the `deliverables/` folder of this repository and uploaded to CourseWorks.

---

# 1. Problem Statement

This project profiles and optimizes the TSFM MCP server pipeline used in IBM AssetOpsBench. The original system lacked detailed end-to-end profiling, making it difficult to identify latency bottlenecks across forecasting, anomaly detection, and fine-tuning workflows.

Our work targets both inference and training performance by instrumenting the pipeline, benchmarking performance, and applying optimizations focused on preprocessing overhead, GPU utilization, and mixed precision execution. The goal is to improve latency and throughput while maintaining forecasting accuracy and anomaly detection quality.

---

# 2. Model/Application Description

- **Application:** TSFM MCP Server (IBM AssetOpsBench)
- **Models:** TinyTimeMixer (TTM) and Amazon Chronos
- **Framework:** PyTorch 2.x + Hugging Face Transformers
- **Dataset:** Synthetic Chiller 9 verification dataset and real HVAC asset datasets
- **Hardware Target:** NVIDIA L4 GPU on Google Cloud Platform
- **Optimization Targets:**
  - Preprocessing bottlenecks
  - GPU underutilization
  - Float32 inference/training overhead
  - Sequential request processing

---

# 3. Final Results Summary

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Forecasting Cold Start | 5.73 s | Reduced | Faster startup |
| Forecasting Steady State | 0.67 s | Reduced | Lower latency |
| Fine-Tuning Runtime | 2.76 s | Reduced | Faster training |
| Integrated TSAD Runtime | 1.35 s | Reduced | Faster pipeline |
| GPU Utilization | Lower | Improved | Better hardware efficiency |
| Preprocessing Overhead | High | Reduced | Parallelized + cached |

**Hardware:** NVIDIA L4 GPU, CUDA 12.8, Python 3.12, PyTorch 2.10

### Headline Result

We developed a reproducible benchmarking harness and interchangeable model interface for the TSFM MCP server, enabling profiling-driven optimizations including preprocessing parallelism, caching, AMP optimization, and Chronos integration while improving end-to-end pipeline efficiency.

---

# 4. Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── deliverables/
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── src/
│   └── servers/
├── tsfm_profiling/
│   ├── baseline_verification/
│   ├── functionality_verification/
│   ├── harness/
│   ├── results/
│   └── profiling/
├── configs/
├── scripts/
└── docs/
```

---

# 5. Reproducibility Instructions

## A. Environment Setup

```bash
# Clone repository
git clone https://github.com/Lion-Latency/AssetOpsBench.git
cd AssetOpsBench

# Activate environment
source .venv/bin/activate

# Install required versions
pip install "numpy==1.26.4" "transformers==4.45.2"
```

Verify installed versions:

```bash
python -c "import numpy as np; import transformers; \
print(np.__version__); print(transformers.__version__)"
```

---

## B. Environment Variables

Ensure the local `.env` file points to the shared TSFM directories:

```bash
PATH_TO_MODELS_DIR=/home/shared/tsfm_profiling_data/models
PATH_TO_DATASETS_DIR=/home/shared/tsfm_profiling_data/datasets
PATH_TO_OUTPUTS_DIR=/home/shared/tsfm_profiling_data/outputs
```

---

## C. Functionality Verification

### Generate Synthetic Verification Data

```bash
python tsfm_profiling/baseline_verification/create_synthetic_data.py
```

### Run Forecasting Check

```bash
python tsfm_profiling/functionality_verification/run_tsfm_forecasting_check.py
```

### Run Fine-Tuning Check

```bash
python tsfm_profiling/functionality_verification/run_tsfm_finetuning_check.py
```

### Run TSAD Check

```bash
python tsfm_profiling/functionality_verification/run_tsad_check.py
```

### Run Integrated TSAD Check

```bash
python tsfm_profiling/functionality_verification/run_integrated_tsad_check.py
```

### Run All Checks

```bash
python tsfm_profiling/functionality_verification/run_all_checks.py
```

---

## D. Benchmarking

### Run Baseline Benchmark

```bash
TSFM_BENCH_MODES=baseline python tsfm_profiling/harness/benchmark_runner.py
```

### Run Parallelism-Only Benchmark

```bash
TSFM_BENCH_MODES=parallelism_only python tsfm_profiling/harness/benchmark_runner.py
```

### Run Combined Optimizations

```bash
TSFM_BENCH_MODES=combined python tsfm_profiling/harness/benchmark_runner.py
```

---

## E. Experiment Tracking Dashboard

Public W&B dashboard:

🔗 https://wandb.ai/lion-latency/hpml-project

The dashboard includes:

- Baseline vs optimized comparisons
- Latency measurements
- GPU utilization metrics
- Profiling traces
- Workflow comparisons
- Chronos vs TTM benchmarking

---

## F. Quickstart: Reproduce the Headline Result

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install "numpy==1.26.4" "transformers==4.45.2"

# Run optimized benchmark
TSFM_BENCH_MODES=combined python tsfm_profiling/harness/benchmark_runner.py
```

---

# 6. Results and Observations

- Preprocessing parallelism reduced CPU-side bottlenecks by parallelizing preprocessing across asset groups.
- Preprocessing caching reduced repeated preprocessing work during repeated benchmark runs.
- AMP optimization improved GPU efficiency during inference and fine-tuning.
- The benchmarking harness enabled reproducible comparisons across workflows and optimization settings.
- Integrated TSAD workflows remained significantly more expensive than standalone workflows due to forecasting overhead.
- Chronos integration demonstrated that the benchmarking harness generalizes beyond TinyTimeMixer.

### Phase 1 Baseline Results

| Workflow | Cold Start (s) | Steady-State Avg (s) |
|---|---|---|
| Forecasting | 5.73 | 0.67 |
| Fine-Tuning | 3.98 | 2.76 |
| TSAD | 0.32 | 0.31 |
| Integrated TSAD | 2.05 | 1.35 |

---

# 7. Notes

- Source code lives under `src/`
- Benchmarking tools are under `tsfm_profiling/harness/`
- Functionality verification scripts are under `tsfm_profiling/functionality_verification/`
- Results and profiling artifacts are stored under `tsfm_profiling/results/`

---

# AI Use Disclosure

Per the HPML AI Use Policy.

## Did your team use AI tools?

Yes.

## Tools Used

- ChatGPT
- GitHub Copilot

## Specific Purposes

- Grammar checking and proofreading
- Debugging environment and dependency issues
- Clarifying profiling concepts and benchmarking workflows
- Improving documentation readability
- Assisting with README formatting

## Sections Affected

- README documentation
- Presentation wording and formatting
- Minor debugging support during development

## Verification Process

All experiments, profiling analysis, benchmarking results, and optimization decisions were independently implemented, verified, and interpreted by the team. AI-generated suggestions were reviewed, modified, and validated against actual experimental outputs and profiler traces.

---

# License

Released under the MIT License.

---

# Citation

```bibtex
@misc{lionlatency2026,
  title  = {Profiling and Optimizing the TSFM MCP Server},
  author = {Go, Sally and Colman, Sam and Kwon, Byeolah and Pasiecznik, Tomas},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/Lion-Latency/AssetOpsBench}
}
```

---

# Contact

Open a GitHub Issue on the repository for questions or discussions.

---

*HPML Spring 2026 — Columbia University*

# Upstream Repository Acknowledgment

This repository is forked from the original IBM AssetOpsBench repository and extends it with:

- profiling instrumentation
- benchmarking harnesses
- preprocessing optimizations
- AMP optimization
- Chronos integration
- reproducibility tooling

Original repository:
https://github.com/IBM/AssetOpsBench

We thank the original contributors for providing the TSFM benchmarking framework used in this project.




