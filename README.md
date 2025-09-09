# CART-Clin: Context-Aware Red-Teaming Framework for Clinical LLMs

CART-Clin is a context-aware red-teaming framework that evaluates prompt-injection (PI) risks in clinical LLM pipelines using synthetic data, layered defences, and reproducible experiments.

---

## Installation

Follow these steps to set up and run the CART-Clin project locally:

### 1. Clone or Extract
```bash
git clone https://github.com/<your-username>/CART-Clin.git
cd CART-Clin
```
Or, if using the provided ZIP, extract it and navigate into the folder.

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Environment
- **Windows**
```bash
venv\Scripts\activate
```
- **Mac/Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Prototype
```bash
python cartclin_app.py
```

**Expected output (example):**
```text
Baseline → ASR ~55–70%, PHIL ~40–60%, CVR ~55–70%, Latency ~0.10–0.30s
Defended → ASR ~0–5%, PHIL ~0–2%, CVR ~0–5%, Latency ~0.10–0.30s
```

Optional flags:
```bash
python cartclin_app.py --baseline-only
python cartclin_app.py --defended-only
python cartclin_app.py --config cartclin_synth_config.yaml
```

---

## Project Structure
```text
CARTCLIN_Final_Project/
├─ cartclin_app.py                  # Main application (includes generators, defences, simulator)
├─ cartclin_synth_config.yaml       # Experiment configuration (YAML)
├─ cartclin_synth_dataset.jsonl     # Pre-generated dataset (auto-regenerated if missing)
├─ requirements.txt                 # Python dependencies
├─ timeline_cartclin.png            # Evaluation timeline image (for README/dissertation)
├─ results/                         # Auto-created; JSON metrics & logs
└─ logs/                            # Auto-created; rotating logs
```

---

## 4.6 Evaluation Results (Insert Your Numbers)
After running, copy the printed results into your dissertation's §4.6 table.

---

## 4.7 Evaluation Timeline
![Evaluation Timeline](timeline_cartclin.png)

> Ensure `timeline_cartclin.png` is alongside this README so GitHub renders it.

---

## Ethics
All data are synthetic and PHI-free. No ethical approval is required for replication or sharing.
