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
<img src="Screenshot%202025-09-16%20170614.png" width="715">

## Choose evaluation mode
<img src="Screenshot%202025-09-16%20170639.png" width="454">

## Direct Injection run (Baseline mode, 200 trials, Surfaces A–D)
<img src="Screenshot%202025-09-16%20170745.png" width="512">

## Results and logs saved locally
<img src="Screenshot%202025-09-16%20171042.png" width="944">
Select which model to run
<img width="715" height="260" alt="image" src="https://github.com/user-attachments/assets/6bf1e651-d0c1-4d39-922a-506eb9417f92" />

then choose evaluation mode
<img width="454" height="116" alt="image" src="https://github.com/user-attachments/assets/141ace5d-7df0-4347-9055-41aa3aa6c0f4" />

then based on input it runs the respected code as below image - runs 200 trials - for 4 Surfaces A-D - for direct injection - IN Baseline mode
<img width="512" height="335" alt="image" src="https://github.com/user-attachments/assets/6b719454-3470-4545-ade1-f3c314b62326" />

Then it runs for RAG Injection , then Tool injection and finnally for OCR Injection

Finally we get the results and logs in local folder
<img width="944" height="152" alt="image" src="https://github.com/user-attachments/assets/00cc47c5-9404-41d5-b10c-22b9d389a5a4" />

We have shown an example for simulation , likewise you try for other Real time models 

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

> Ensure `timeline_cartclin.png` is alongside this README so GitHub renders it.

---

## Ethics
All data are synthetic and PHI-free. No ethical approval is required for replication or sharing.
