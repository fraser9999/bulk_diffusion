
### **README.md**

````markdown
# Bulk Diffusion 0.1a

**Massive and Continuous Picture Rendering using Stable Diffusion 1.5**

---

## Author
Hermann Knopp  
Contact: [hermann.knopp@gmx.at](mailto:hermann.knopp@gmx.at)  

---

## Version
0.1a — 31.07.2023

---

## Overview
Bulk Diffusion is a Python application that generates images in bulk using **Stable Diffusion** and a wordlist-driven prompt system. It can automatically create diverse image prompts using English word lists, categories, and moods, optionally enhanced with **GPT-Neo** for more creative prompts.  

The software is designed for NVIDIA GPUs with at least 6GB VRAM. CPU-only rendering is **not supported yet**.

---

## Features

- Generates images continuously in bulk.
- Uses **Stable Diffusion 1.5** or other diffusion models (Dreamlike Diffusion, Inkpunk).
- Wordlist-based prompt generation:
  - `words.txt` — general descriptive words
  - `categories.txt` — picture categories
  - `moods.txt` — artistic moods
- Optional GPT-Neo prompt enhancement for more creative and stylistic outputs.
- Saves generated images and prompt details automatically in a specified folder.
- Interactive selection of model, output folder, and prompt engine.

---

## Installation

### Requirements
- Python 3.10.10 x64
- NVIDIA GPU with CUDA support (minimum 6GB VRAM)
- Virtual environment recommended (`venv`)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
````

---

## Usage

1. Run the main Python script:

```bash
python bulk_diffusion.py
```

2. Follow interactive prompts:

   * Select image output folder
   * Choose whether to use GPT prompt engine
   * Choose diffusion model (1–7)
   * Start batch rendering

3. Generated images and corresponding prompt text files will be saved automatically.

---

## Models Supported

| Model Nr | Model Name           | Description                  |
| -------- | -------------------- | ---------------------------- |
| 1        | Stable Diffusion 1.5 | GPT Lexart                   |
| 2        | Dreamlike Diffusion  | Photorealistic GPT Lexart    |
| 3        | Dreamlike Diffusion  | Creative Art GPT Lexart      |
| 4        | Stable Diffusion 1.5 | GPTNeo Custom                |
| 5        | Dreamlike Diffusion  | Photorealistic GPTNeo Custom |
| 6        | Dreamlike Diffusion  | Creative Art GPTNeo Custom   |
| 7        | Inkpunk Diffusion    | GPTNeo Custom                |

> For GPTNeo Custom models, download the release from GitHub (`pytorch_model.bin` in `model/` folder, ~500MB).

---

## Output

* Images saved as `.png`
* Prompt text saved as `.txt` alongside each image
* Automatic naming: `test_DDMMYYYY_HHMMSS.png` and `.txt`
* Stores seed, category, mood, image size, guidance scale, and prompt

---

## Hardware Requirements

* NVIDIA GPU with **6GB+ VRAM**
* VRAM usage:

  * 4GB (FP32) / 2GB (FP16) for Stable Diffusion
  * 0.5GB for GPT-Neo model
  * 1–1.5GB RAM for PyTorch system files
* CPU-only rendering is **not supported yet**.

---

## Wordlists

* `words.txt` — descriptive words for random prompt generation
* `categories.txt` — categories for image themes
* `moods.txt` — creative style or mood descriptors

> All files can be edited and expanded by appending new words.

---

## License

MIT License (or specify your preferred license)

---

## Contact

GitHub Issues for bug reports or feature requests
Email: [hermann.knopp@gmx.at](mailto:hermann.knopp@gmx.at)

````

---


> Hinweis: `aitextgen` wird für GPT-Neo Text-Prompt-Generierung benötigt, `diffusers` für Stable Diffusion.

