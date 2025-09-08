# CART-Clin: Context-Aware Red-Teaming Framework for Clinical LLMs

CART-Clin is a **context-aware red-teaming framework** for evaluating **prompt-injection (PI) vulnerabilities** in **clinical Large Language Models (LLMs)**.  
This repository contains a fully runnable **synthetic dataset**, **evaluation pipeline**, and **defence-stack integration** for reproducible research.

---

## **Features**
- Evaluate vulnerabilities across **four clinical scenarios**:
    - **S1**: Discharge Summarisation
    - **S2**: Medication Reconciliation
    - **S3**: Oncology Counselling
    - **S4**: Patient Q&A
- Simulate attacks across **four injection surfaces**:
    - **A:** Direct prompt injection
    - **B:** Retrieval-Augmented Injection (RAG poisoning)
    - **C:** Tool-chain manipulation
    - **D:** OCR/multimodal attacks
- Layered defences tested:
    - Prompt hardening
    - RSL-RAG sanitisation
    - Tool mediation
    - Output verification
- Metrics tracked: **ASR**, **PHIL**, **CVR**, **Latency**

---

## **Project Structure**
```plaintext
CARTCLIN_Project_Package/
├── cartclin_synth_dataset.jsonl   # Synthetic dataset (240 records)
├── cartclin_synth_config.yaml     # Config & reproducibility knobs
├── cartclin_readme.txt            # Dataset + ethics documentation
├── cartclin_app.py                # CART-Clin prototype implementation
├── requirements.txt              # Minimal dependencies
└── README.md                     # Project documentation

## **Installation**

Clone the repo or extract the CARTCLIN_Project_Package.zip into a local folder.
