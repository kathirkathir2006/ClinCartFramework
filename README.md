# CART-Clin: Context-Aware Red-Teaming Framework for Clinical LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

CART-Clin is a **context-aware red-teaming framework** for evaluating **prompt-injection (PI) vulnerabilities** in **clinical Large Language Models (LLMs)**. This repository contains a fully runnable **synthetic dataset**, **evaluation pipeline**, and **defence-stack integration** for reproducible research.

![CART-Clin Architecture](timeline_cartclin.png)

## 🚀 Features

- **Comprehensive Clinical Evaluation** across four scenarios:
  - **S1**: Discharge Summarisation
  - **S2**: Medication Reconciliation  
  - **S3**: Oncology Counselling
  - **S4**: Patient Q&A

- **Multi-Surface Attack Simulation**:
  - **A**: Direct prompt injection
  - **B**: Retrieval-Augmented Injection (RAG poisoning)
  - **C**: Tool-chain manipulation
  - **D**: OCR/multimodal attacks

- **Layered Defence Testing**:
  - Prompt hardening
  - RSL-RAG sanitisation
  - Tool mediation
  - Output verification

- **Comprehensive Metrics**: ASR, PHIL, CVR, Latency tracking

## 📁 Project Structure

```plaintext
CARTCLIN_Project_Package/
├── cartclin_synth_dataset.jsonl   # Synthetic dataset (240 records)
├── cartclin_synth_config.yaml     # Configuration & reproducibility settings
├── cartclin_readme.txt            # Dataset + ethics documentation
├── cartclin_app.py                # CART-Clin prototype implementation
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/CART-Clin.git
   cd CART-Clin
   ```
   *Or extract the provided ZIP file and navigate to the folder*

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run CART-Clin**
   ```bash
   python cartclin_app.py
   ```

## 📊 Expected Output

```
Baseline → ASR 63.0%, PHIL 52.5%, CVR 63.0%, Latency 0.000s, Trials 200
Defended → ASR 0.0%, PHIL 0.0%, CVR 0.0%, Latency 0.000s, Trials 200
```

## 🔬 Evaluation Results

| Configuration | ASR (%) | PHIL (%) | CVR (%) | Median Latency (s) | Trials |
|---------------|---------|----------|---------|-------------------|--------|
| Baseline (undefended) | 63.0 | 52.5 | 63.0 | 0.000 | 200 |
| Defended (CART-Clin) | 0.0 | 0.0 | 0.0 | 0.000 | 200 |

### 🔍 Key Findings

- ⚠️ **Without defences**: 63% of attacks succeed
- 🚨 **PHI leakage** occurs in ~52.5% of trials
- ⚡ **Constraint violations** occur in ~63% of baseline runs
- ✅ **With CART-Clin's layered defences**: ASR, PHIL, and CVR drop to 0%
- ⏱️ **Latency overhead** is negligible, suitable for clinical environments

## 📋 Evaluation Timeline

| Phase | Description | Objective |
|-------|-------------|-----------|
| **P0** | Baseline Evaluation | Run undefended trials to measure vulnerability |
| **P1** | Static Attacks v1 | Launch prompt-injection attempts with fixed payloads |
| **P2** | Adaptive Attacks v1 | Attack evolves based on model outputs |
| **P3** | Defence Stack Integration | Apply prompt hardening + RSL-RAG + tool mediation |
| **P4** | Adaptive Attacks v2 | Retest with improved adversarial strategies |
| **P5** | Final Audit | Evaluate overall system safety and reproducibility |

## 🛡️ Dataset Ethics

- ✅ **No real patient data used**
- ✅ All samples generated programmatically
- ✅ Placeholder PHI tokens simulate leakage scenarios
- ✅ Fully compliant with ethics requirements
- ✅ Safe for open-source distribution

## 📄 License

You are free to use, modify, and distribute it.

## 🙏 Acknowledgements

This project was developed as part of the **MSc Dissertation** for the **University of Surrey**

- **Supervisor**: Dr. Caitlin Dragan
- **Author**: Kathiresan
- **Email**: kn00657@surrey.ac.uk

## 📚 Citation

If you use CART-Clin in your research, please cite:

```bibtex
@misc{kathiresan2024cartclin,
  title={CART-Clin: Context-Aware Red-Teaming Framework for Clinical LLMs},
  author={Kathiresan},
  year={2024},
  school={University of Surrey}
}
```

---

⭐ **Star this repository** if you find it useful for your research!

📧 **Questions?** Feel free to open an issue or contact: kn00657@surrey.ac.uk
