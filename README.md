# CART-Clin: Context-Aware Red-Teaming Framework for Clinical LLMs

CART-Clin is a **context-aware red-teaming framework** for evaluating **prompt-injection (PI) vulnerabilities** in **clinical Large Language Models (LLMs)**.  
It simulates adversarial scenarios, evaluates vulnerabilities across multiple attack surfaces, and tests **layered defence strategies** to enhance patient safety and data privacy.

---

## **Features**
- **Synthetic dataset generation** (no PHI, ethics-safe, GitHub-friendly)
- Evaluation of **four clinical scenarios**:
  - S1: Discharge Summarisation  
  - S2: Medication Reconciliation  
  - S3: Oncology Counselling  
  - S4: Patient Q&A  
- Testing across **four attack surfaces**:
  - A: Direct Prompt Injection
  - B: Retrieval-Augmented Injection (RAG poisoning)
  - C: Tool-Chain Manipulation
  - D: OCR/Multimodal Injection
- Layered defence strategies:
  - Prompt hardening
  - RSL-RAG sanitisation
  - Tool mediation
  - Output verification
- Metrics tracked:
  - **ASR** – Attack Success Rate
  - **PHIL** – PHI Leakage Rate
  - **CVR** – Constraint Violation Rate
  - **Latency** – Overhead introduced by defences

---

## **Project Structure**
```plaintext
CARTCLIN_Project_Package/
├── cartclin_synth_dataset.jsonl    # Synthetic dataset (240 records)
├── cartclin_synth_config.yaml      # Config & reproducibility knobs
├── cartclin_readme.txt             # Ethics & dataset documentation
├── cartclin_app.py                 # Main CART-Clin prototype code
├── requirements.txt               # Minimal dependencies
└── README.md                      # (You are here)

