# SCIJuDAC: A Fine-Grained Benchmark for Appellate Legal Judgment Prediction in the Supreme Court of India

## Overview

This repository accompanies the paper:

> **SCIJuDAC: A Fine-Grained Benchmark for Appellate Legal Judgment Prediction in the Supreme Court of India**

<!-- **Authors:** Anurag Yadav, Durga Toshniwal -->

SCIJuDAC is a large-scale benchmark dataset for the Legal Judgment Prediction (LJP) task in the Supreme Court of India (SCI). Unlike existing appellate judgment prediction benchmarks that primarily employ binary outcome labels, SCIJuDAC introduces a **fine-grained three-label framework** that captures additional nuances in appellate court decisions.

The benchmark contains **53,548 appellate judgments** from the Supreme Court of India spanning **January 1950 to June 2025**. Through extensive classification, separability, and calibration analyses, we demonstrate that SCIJuDAC presents a more challenging and realistic evaluation setting for modern language models than existing benchmarks such as ILDC.

## Motivation

Legal Judgment Prediction benchmarks play a crucial role in evaluating the reasoning capabilities of language models in the legal domain. However, existing appellate benchmarks often simplify judicial outcomes into coarse-grained labels, potentially overlooking important distinctions in appellate decisions.

SCIJuDAC was developed to address the following research question:

> **Do current benchmark datasets sufficiently represent the nuances of appellate court cases?**

To answer this question, we introduce a refined labeling scheme that captures additional appellate outcome distinctions and evaluate a range of encoder-based and large language models on both SCIJuDAC and ILDC.

## Key Contributions

* Introduction of **SCIJuDAC**, a fine-grained appellate legal judgment prediction benchmark.
* Collection and processing of **53,548 Supreme Court of India appellate judgments**.
* Introduction of a **three-label classification framework** that captures additional appellate decision nuances.
* Comprehensive comparison against the ILDC benchmark.
* Evaluation of multiple BERT-based and instruction-tuned language models.
* Detailed **separability** and **calibration** analyses demonstrating increased benchmark difficulty.
* Public release of code, trained checkpoints, and evaluation pipelines.

## Dataset Construction

### Data Source

Judgments were collected from publicly available Supreme Court of India decisions available through:

https://www.courtkutchehry.com/
vg. doc. length (in words) 4
Largest and smallest doc. length 388,993, 51
Avg. # of tokens (GPT4) 
Largest and smallest token count 461,421, 56
### Dataset Statistics

| Property                | Value                                  |
| ----------------------- | -------------------------------------- |
| Jurisdiction            | Supreme Court of India                 |
| Case Type               | Appellate Judgments                    |
| Time Span               | January 1950 – June 2025               |
| Total Judgments         | 54,006                                 |
| Labels                  | 3 (allowed, dismissed, partly allowed) |
| Task                    | Legal Judgment Prediction              |
| Mean word count         | 2970.62                                |
| Std. dev. of word count | 4770.17                                |
| Mean token count (GPT4) | 3807.99                                |
| Largest word count      | 3,88,993                               |
| Smallest word count     | 51                                     |
| Largest token count     | 4,61,421                               |
| Smallest token count    | 56                                     |

### Preprocessing

The dataset construction pipeline includes:

* Collection of appellate judgments.
* Dataset cleaning and normalization.
* Removal of verdict information from document text to prevent label leakage.
* Extraction of final verdict from the verdict information

## Models Evaluated

### Encoder Models

* BERT-base
* LegalBERT
* InLegalBERT

### Large Language Models

* Llama 3.1 8B
* Qwen 2.5 7B
* Mistral 0.3 7B
* Saul 7B

## Evaluation

### Classification Metrics

* Accuracy
* Macro Recall
* Weighted Recall
* Macro F1
* Weighted F1

### Separability Metrics

* Silhouette Score
* Davies Bouldin Index (DBI)
* Top-2 Logit Margin Score

### Calibration Metrics

* Brier Score
* Expected Calibration Error (ECE)

## Repository Structure

```text
project/
├── checkpoints/
│   ├── ILDC/
│   └── SCIJuDAC/
│
├── data/
│   └── dataset/
│
├── logs/
│   ├── analysis/
│   ├── ILDC/
│   └── SCIJuDAC/
│
├── src/
│   ├── config/
│   ├── dataset_builder/
│   ├── labelling/
│   ├── models/
│   ├── tools/
│   ├── training/
│   └── utilities/
│
└── README.md
```

## Environment

### Software Requirements

* Python 3.13.12
* transformers 4.56.0
* PyTorch 2.8.0 + CUDA 12.8
* datasets 4.3.0
* numpy 2.3.5
* peft 0.18.1
* bitsandbytes 0.49.2
* scikit-learn 1.8.0

### Installation

```bash
pip install -r requirements.txt
```

## Training

Run model training from the `src/models/` directory in the command line. Change parameters to select different model/dataset:

```bash
python run_training.py \
    --name test \
    --model bert \
    --dataset SCIJuDAC \
    --batch_size 16 \
    --epochs 3 \
    --seed 42 \
    --load_ckpt n
    --collapse_labels n \
    --collapse_classifier n \
```
Replace the value after `--model` parameter with one of these: `bert`, `legalbert`, `inlegalbert`, `llama`, `qwen`, `mistral`, `saul` to select a model. Replace the value after `--dataset` parameter with one of these: `ILDC_single`, `ILDC_multi`, `SCIJuDAC` to select a dataset. Tune the model's batch size, training epochs, or initial seed with the given parameters. Toggle label collapse or classifier collapse ablations with the `--collapse_labels` or `--collapse_classifier` parameters.
Note: Only one ablation will be allowed at a time.

### Configuration

Project-wide directory paths and configuration variables are maintained in:

```text
src/config/
```

## Separability and Calibration Analysis

Run analysis using checkpoints stored in the `checkpoints/` directory. Works the same as the training execution command, with the `--load_cc_ckpt` and `--load_cl_ckpt` used to load the collapsed classifier and collapsed labels ablation checkpoints respectively.
Note: Only one checkpoint will be loaded at a time.

```bash
python run_analysis.py
    --name test \
    --model bert \
    --dataset SCIJuDAC \
    --batch_size 16 \
    --epochs 3 \
    --seed 42 \
    --load_ckpt n
    --collapse_labels n \
    --collapse_classifier n \
```

Analysis outputs are stored in:

```text
logs/analysis/
```

## Outputs

The framework generates:

* Training, validation, and test result logs
* Model checkpoints
* Classification metrics
* Separability analysis reports
* Calibration analysis reports
* UMAPs and confusion matrices for all runs

## Reproducibility

### Random Seed

```text
42
```

### Hardware Used

* NVIDIA RTX Pro 4000 Blackwell (24 GB VRAM)
* CUDA 12.8
* 96 GB RAM
* Intel Core Ultra 9 285K × 24 CPU

### Runtime

| Task                                              | Approximate Runtime |
| ------------------------------------------------- | ------------------- |
| BERT-based Training                               | ~3 hours            |
| LLM Fine-tuning (when trained on last 512 tokens) | ~1 day              |
| Separability & Calibration Analysis               | ~30 minutes         |

<!-- ## Citation

If you use SCIJuDAC in your research, please cite:

```bibtex
@article{yadav2025scijudac,
  title={SCIJuDAC: A Fine-Grained Benchmark for Appellate Legal Judgment Prediction in the Supreme Court of India},
  author={Yadav, Anurag and Toshniwal, Durga},
  journal={Under Review},
  year={2025}
}
```
-->
## License

### Code

Released under the MIT License.

### Dataset

Please refer to the dataset license and terms of use. Users are responsible for ensuring compliance with the terms governing the original judicial documents.

## Contact

For questions, issues, or collaborations, please open an issue in this repository.
