# ClinCartFramework – Dissertation Repository (COMM067)

This repository contains the code, configuration files, datasets, and supporting materials for my MSc dissertation project **ClinCartFramework**.  

The framework implements clinical data cartography methods, supporting:
- Data preprocessing and synthetic dataset generation.
- Configurable clinical experiments.
- Analysis, figures, and results visualization.
- Automated pipelines for reproducible runs.

---

## 📂 Repository Structure

├── logs/ # Run logs
├── results/ # Output results (images, figures, tables)
├── cart_clin_final.py # Main implementation
├── cartclin_app_old_works.py # Previous version (archived)
├── cartclin_readme.txt # Original notes
├── cartclin_synth_config.yaml# Synthetic data configuration
├── cartclin_synth_dataset.jsonl # Example synthetic dataset
├── config.json # General configuration
├── requirements.txt # Python dependencies
├── run_bat.bat # Batch script for running
├── setup_bat.bat # Batch setup script
├── README.md # This file
└── (PNG figures) # Visual outputs

yaml
Copy code

---

## ⚙️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/kathirkathir2006/ClinCartFramework.git
   cd ClinCartFramework
Create a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
(Optional) Configure run settings in config.json or cartclin_synth_config.yaml.

▶️ Usage
To execute the main pipeline:

bash
Copy code
python cart_clin_final.py
To run with batch script (Windows):

bash
Copy code
run_bat.bat
📊 Results & Figures
Experimental Runs
<p align="center"> <img src="file1.png" alt="Run 1" width="48%"> <img src="file2.png" alt="Run 2" width="48%"> </p> <p align="center"> <img src="file3.png" alt="Run 3" width="48%"> <img src="file4.png" alt="Run 4" width="48%"> </p>
Project Timeline
<p align="center"> <img src="timeline_cartclin.png" alt="Project timeline" width="80%"> </p>
🧪 Configuration
cartclin_synth_config.yaml – Controls synthetic dataset generation.

cartclin_synth_dataset.jsonl – Pre-generated dataset.

config.json – General runtime configuration.

Adjust parameters in YAML/JSON files for new experiments.

📄 References
This dissertation follows IEEE citation style.
A corrected reference list (aligned with in-text citation numbers) is included in the submission files.

🙏 Acknowledgements
Supervised by Prof. Ala Marshan and Dr. Catalin.

Conducted as part of COMM067 Dissertation Module.

Special thanks to the academic staff and peers for feedback and support.

📌 Notes
The results (analysis, figures, tables) are unchanged between versions.

Only the reference list and in-text citation mapping have been corrected.

For details, see the included change log in the submitted files.
