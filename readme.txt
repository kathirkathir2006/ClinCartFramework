# CART-Clin: Clinical LLM Red-Teaming Framework

A simple framework for evaluating security vulnerabilities in clinical Large Language Models.

**Author:** Kathiresan  
**Supervisor:** Dr. Caitlin Dragan  
**University:** University of Surrey  
**Program:** MSc Computer Science (Dissertation)

## What This Does

Tests clinical AI systems for security problems like:
- Leaking patient information (PHI)
- Ignoring safety protocols
- Following malicious instructions
- Bypassing privacy controls

## Files Included

- `cart_clin.py` - Main framework (run this)
- `requirements.txt` - Python packages needed
- `config.json` - Settings and model configurations
- `setup.bat` - Windows setup script
- `run.bat` - Windows run script
- `README.md` - This file
- `.gitignore` - Git ignore rules

## Quick Start (Windows)

1. **Setup (first time only):**
   ```
   Double-click setup.bat
   ```

2. **Run the framework:**
   ```
   Double-click run.bat
   ```

## Manual Setup

1. **Install Python 3.8+** from https://python.org

2. **Create project folder:**
   ```
   mkdir cart-clin
   cd cart-clin
   ```

3. **Copy all files to the folder**

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Run:**
   ```
   python cart_clin.py
   ```

## How to Use

1. Run the program
2. Select a model to test (1-6)
3. Choose evaluation mode:
   - Baseline (no