# HPML Final Project: Profiling and Optimizing the TSFM MCP Server

> **Course:** High Performance Machine Learning  
> **Semester:** Spring 2026  
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team 19 Lion Latency
- **Members:**
  - Sally Go (yg3066) — External benchmarking harness, W&B logging, benchmarking
  - Sam Colman (sc5750) — Internal instrumentation, preprocessing caching, benchmarking
  - Byeolah Kwon (bk2833) — Profiling, preprocessing parallelism, benchmarking
  - Tomas Pasiecznik (tp2758) — Environment setup, interchangeable model interface, benchmarking

---

## Submission

- **GitHub repository:** https://github.com/Lion-Latency/AssetOpsBench
- **Final report:** `deliverables/HPML_Final_Report.pdf`
- **Final presentation:** `deliverables/HPML_Final_Presentation.pptx`
- **Experiment-tracking dashboard:** https://wandb.ai/lion-latency/hpml-project-final

The final report PDF and presentation file are checked into the `deliverables/` folder of this repository and uploaded to CourseWorks.

---

# 1. Problem Statement

This project profiles and optimizes the TSFM MCP server pipeline used in the IBM AssetOpsBench framework for time-series forecasting, anomaly detection, and fine-tuning workflows. The original system lacked detailed end-to-end instrumentation and reproducible benchmarking, making it difficult to identify latency bottlenecks across preprocessing, model loading, inference, and training stages.

Our work targets both inference and training performance by introducing a stage-level profiling system, external benchmarking harness, and interchangeable model interface for evaluating multiple time-series foundation models. We evaluated preprocessing parallelism, preprocessing caching, model caching, fast trainer configurations, and bfloat16 mixed precision execution to improve latency and throughput while maintaining forecasting accuracy and anomaly detection quality.

---

# 2. Model/Application Description

- **Model architecture:**
  - **Models Tested:** TinyTimeMixer (TTM) and Amazon Chronos
  - **Application:** TSFM MCP Server (IBM AssetOpsBench)
- **Framework:** PyTorch 2.x + Hugging Face Transformers
- **Dataset:** Synthetic Chiller 9 verification dataset and real HVAC asset datasets. (Relevant sample included in repo.)
- **Custom layers or modifications:**
  - Created an external benchmarking harness that produces reproducible benchmarks with optimizations for the AssetOpsBench TSFM MCP Server. The purpose of this harness is to drive experiments, collects measurements, and log results to Weights & Biases.
  - Created an internal instrumentation layer injected into the TSFM MCP server code for per-stage profiling.
  - Created an interchangeable model interface to introduce the ability to swap time-series models for performance comparisons of the AssetOpsBench TSFM MCP Server with models other than TTM.
- **Hardware Target:** NVIDIA L4 GPU on Google Cloud Platform

---

# 3. Final Results Summary

| Workflow | Baseline (s) | Best Optimized (s) | Best Optimization | Improvement | W&B Run Link |
|---|---:|---:|---|---:|---|
| Forecasting (TTM) | 46.95 | 31.31 | Fast Trainer | 33.3% faster | [https://wandb.ai/lion-latency/hpml-project-final/runs/cnroix9k]() |
| Forecasting (Chronos) | 46.95 | 3.67 | Combined Optimizations (1 & 2) | 12.8× faster | [https://wandb.ai/lion-latency/hpml-project-final/runs/2bckvz2x]() |
| Fine-tuning | 12.89 | 11.32 | Parallelism Only | 12.2% faster | [https://wandb.ai/lion-latency/hpml-project-final/runs/1wprdiyu]() |
| TSAD | 109.79 | 107.89 | Cache Only | 1.7% faster | [https://wandb.ai/lion-latency/hpml-project-final/runs/bp3p83rq](See Run) |
| Integrated TSAD | 2335.64 | 1527.30 | Combined Inference (3+4+5) | 34.6% faster | [https://wandb.ai/lion-latency/hpml-project-final/runs/jp9jpbos]() |

### Optimization Observations

| Optimization | Main Observation |
|---|---|
| Cache Only | Improved TSAD slightly but slowed forecasting and integrated TSAD |
| Parallelism Only | Produced the best fine-tuning result but regressed TSAD workloads |
| Cache + Parallelism | Improved fine-tuning slightly but increased integrated TSAD runtime |
| Model Cache | Regressed performance across all workflows |
| Fast Trainer | Produced the best TTM forecasting result and major integrated TSAD speedup |
| bf16 | Improved forecasting and integrated TSAD latency through mixed precision execution |
| Combined Inference (3+4+5) | Achieved the fastest integrated TSAD result overall |

**Hardware:** NVIDIA L4 GPU, CUDA 12.8, Python 3.12, PyTorch 2.10

### Headline Result

As part of the IBM AssetOpsBench project, our team developed a reproducible benchmarking harness, stage-level profiling system, and interchangeable model interface that identified preprocessing and inference bottlenecks, achieving up to **12.8× faster forecasting latency** with Chronos and approximately **34.6% lower integrated TSAD runtime** using Fast Trainer and bfloat16 optimizations.

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
│       └── tsfm/
│           └── interchangeable_model_interface/
│               ├── models/
│               │   ├── chronos.py
│               │   └── ttm.py
│               └── interchangeable_model_interface.py
├── tsfm_profiling/
│   ├── data/
│       └── sample.csv
│   ├── functionality_verification/
│       └── chronos/
│       │   ├── run_integrated_tsad_chronos_check.py
│       │   ├── run_tsad_chronos_check.py
│       │   ├── run_tsfm_finetuning_chronos_check.py
│       │   └── run_tsfm_forecasting_chronos_check.py
│       └── ttm/
│           ├── run_integrated_tsad_check.py
│           ├── run_tsad_check.py
│           ├── run_tsfm_finetuning_check.py
│           └── run_tsfm_forecasting_check.py
│   └── harness/
│       └── benchmark_runner.py
└── .env
```

---

# 5. Reproducibility Instructions

## A. Environment Setup

```bash
# Clone repository
git clone https://github.com/Lion-Latency/AssetOpsBench.git
cd AssetOpsBench

# Create environment
python3 -m venv .venv

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
PATH_TO_MODELS_DIR=${HOME}/AssetOpsBench/src/servers/tsfm/artifacts/tsfm_models
PATH_TO_OUTPUTS_DIR=${HOME}/AssetOpsBench/tsfm_profiling/harness/results
```

The dataset is included in the repository to make the experiments easy to reproduce.
---

## C. Benchmarking

### Run Baseline Benchmark

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py
```

#### Specify a mode with: --modes (baseline, cache_only, parallelism_only, combined, model_cache, fast_trainer, bf16, all_inference)

#### Specify a model with: --model (ttm, chronos)

#### Specify a workflow with: --workflows (forecasting, finetuning, tsad, integrated_tsad)
- Please append "_chronos" when running benchmarks using the Chronos model (Example: forecasting_chronos).

### Run Optimization 1 (Cache Only) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model ttm --workflows integrated_tsad
```

### Run Optimization 2 (Parallelism Only) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model ttm --workflows integrated_tsad
```

### Run Combined (Optimizations 1 & 2) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model ttm --workflows integrated_tsad
```

### Run Optimization 3 (Model Cache) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model ttm --workflows integrated_tsad
```

### Run Optimization 4 (Fast Trainer) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model ttm --workflows integrated_tsad
```

### Run Optimization 5 (bf16) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model ttm --workflows integrated_tsad
```

### Run Combined All Inference (Optimizations 3, 4, & 5) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model ttm --workflows forecasting
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model ttm --workflows finetuning
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model ttm --workflows tsad
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model ttm --workflows integrated_tsad
```

## Chronos Benchmarks

### Run Chronos - Optimization 1 (Cache Only) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes cache_only --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Optimization 2 (Parallelism Only) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes parallelism_only --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Combined (Optimizations 1 & 2) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes combined --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Optimization 3 (Model Cache) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes model_cache --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Optimization 4 (Fast Trainer) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Optimization 5 (bf16) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes bf16 --model chronos --workflows integrated_tsad_chronos
```

### Run Chronos - Combined All Inference (Optimizations 3, 4, & 5) Benchmarks

```bash
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model chronos --workflows forecasting_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model chronos --workflows finetuning_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model chronos --workflows tsad_chronos
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes all_inference --model chronos --workflows integrated_tsad_chronos
```


---

## D. Functionality Verification (Optional - Only used for debugging.)

### Run TTM Forecasting Check

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/ttm/run_tsfm_forecasting_check.py
```

### Run TTM Fine-Tuning Check

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/ttm/run_tsfm_finetuning_check.py
```

### Run TTM TSAD Check

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/ttm/run_tsad_check.py
```

### Run TTM Integrated TSAD Check

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/ttm/run_integrated_tsad_check.py
```

### Run Chronos Forecasting Check (Uses Interchangeable Model Interface)

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/chronos/run_tsfm_forecasting_chronos_check.py
```

### Run Chronos Fine-Tuning Check (Uses Interchangeable Model Interface)

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/chronos/run_tsfm_finetuning_chronos_check.py
```

### Run Chronos TSAD Check (Uses Interchangeable Model Interface)

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/chronos/run_tsad_chronos_check.py
```

### Run Chronos Integrated TSAD Check (Uses Interchangeable Model Interface)

```bash
python ~/AssetOpsBench/tsfm_profiling/functionality_verification/chronos/run_integrated_tsad_chronos_check.py
```
---

## E. Experiment Tracking Dashboard

Public W&B dashboard:

🔗 https://wandb.ai/lion-latency/hpml-project-final

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
# Clone repository
git clone https://github.com/Lion-Latency/AssetOpsBench.git
cd AssetOpsBench

# Create environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install required versions
pip install "numpy==1.26.4" "transformers==4.45.2"

# Run benchmark for TTM forecasting with Optimization 4 (Fast Trainer)
python ~/AssetOpsBench/tsfm_profiling/harness/benchmark_runner.py --modes fast_trainer --model ttm --workflows forecasting
```

---

# 6. Results and Observations

- Preprocessing parallelism reduced CPU-side bottlenecks and produced the best fine-tuning result, improving latency from 12.89 s to 11.32 s (12.2% faster).

- Cache-only preprocessing optimization slightly improved standalone TSAD performance but did not consistently improve forecasting or integrated TSAD workloads.

- Fast Trainer produced the best TinyTimeMixer (TTM) forecasting result, reducing forecasting latency from 46.95 s to 31.31 s (33.3% faster).

- bfloat16 mixed precision execution significantly improved integrated TSAD performance, reducing runtime from 2335.64 s to 1528.25 s (34.6% faster) while improving GPU efficiency.

- Model Cache optimization regressed performance across all workflows, suggesting that model loading was not the dominant bottleneck in repeated benchmark execution.

- Integrated TSAD remained substantially more expensive than standalone workflows because it includes both forecasting and anomaly detection stages in a single pipeline.

- Chronos integration demonstrated that the benchmarking harness generalizes beyond TinyTimeMixer and achieved the best overall forecasting result at 3.67 s using combined optimizations.

- The benchmarking harness and instrumentation system enabled reproducible comparisons across preprocessing, mixed precision, trainer, and model-level optimizations while logging metrics consistently to Weights & Biases.

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




