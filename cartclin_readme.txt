CART-Clin Project Package
=========================
Generated: 2025-09-08T13:29:15.968182Z

Overview
--------
This package provides a runnable prototype for CART-Clin (Context-Aware Red-Teaming for clinical LLMs)
with a fully synthetic dataset. No real patient data is included.

Files
-----
- cartclin_synth_dataset.jsonl   : Synthetic dataset (240 records; 60 per scenario S1–S4)
- cartclin_synth_config.yaml     : Reproducibility knobs and notes
- cartclin_app.py                : Runnable prototype (baseline vs defended)
- requirements.txt               : Minimal Python dependencies

Ethics
------
- All content is synthetic and researcher-generated.
- No personally identifiable information (PHI) is present.
- Therefore, this artifact does not require ethics approval and may be shared on GitHub.

Usage
-----
1) Create a virtual environment (optional but recommended):
   python -m venv .venv && . .venv/bin/activate   # (Linux/macOS)
   # or on Windows:
   py -m venv .venv && .venv\Scripts\activate

2) Install requirements:
   pip install -r requirements.txt

3) Run the prototype:
   python cartclin_app.py

   You should see printed metrics, e.g.:
   Baseline → ASR ~60–65%, PHIL ~45–55%, CVR ~60–65%
   Defended → ASR ~0%, PHIL ~0%, CVR ~0%

Notes
-----
- The prototype is simulated and deterministic given seed values.
- To adjust trial sizes or seeds, edit the constants at the top of cartclin_app.py.
