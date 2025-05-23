# <img src="docs/centurion_mascot.png" alt="Centurion mascot" width="120px" align="left" />

# Centurion · Julius&nbsp;⇄&nbsp;Foundation-Model Bridge

[![build](https://github.com/your-org/centurion/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/centurion/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/your-org/centurion)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/centurion)](https://pypi.org/project/centurion/)

> **Centurion** is a lightweight plug-in that lets the classic **[Julius](https://github.com/julius-speech/julius)** decoder speak the language of modern speech‑foundation models—**wav2vec 2.0, WavLM, HuBERT, Whisper**, and more.  Keep Julius’ blazing‑fast grammar search, add the acoustic power of giant self‑supervised encoders, no Kaldi lattice wrangling required.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **One‑line drop‑in** | Load any HuggingFace or local *.pt/bin* encoder and start streaming to Julius in minutes. |
| **Model Zoo** | Built‑in presets for wav2vec 2.0 (base/large), WavLM (base/large), HuBERT (base/large/xl), and Whisper (tiny‑en → large‑v3). |
| **Streaming & offline** | Works with Julius’ *vecnet* real‑time protocol **and** batch HTK logit dumping. |
| **Unit mapping** | Automatic projection from model vocab to Julius phone/state IDs (supports BPE, CTC units, or custom lists). |
| **GPU / CPU** | PyTorch ≥2.0, CUDA optional but loved. |
| **Tiny footprint** | Pure‑Python (+PyTorch) ≈ 1 k LOC; deploy on a Jetson Nano or a beefy A100 alike. |

---

## 🚀 Quick Start

```bash
# 1️⃣  Install (Python 3.9+)
pip install centurion

# 2️⃣  Download a model
env MODEL=wav2vec2-large-960h  # any HF repo or local path
auto-centrion-pull $MODEL  # helper script

# 3️⃣  Launch the Centurion server
centurion --model $MODEL --port 5532  # matches Julius vecnet default

# 4️⃣  Fire up Julius
julius -C main.jconf -C am-ctc.jconf \
       -charconv utf-8 utf-8 \
       -module -input vecnet
```
You should see something like:

```
[INFO] Model loaded (output dimension = 50457)
[INFO] Julius connected from 127.0.0.1:5531
[INFO] Ready. Listening for speech...
```

---

## 🏗️ Architecture

```
Audio → Centurion (PyTorch encoder + CTC layer) → Julius vecnet → Grammar / LM search → Final hypothesis
```
*Centurion* exposes Julius‑compatible HTK feature streams: frame‑level log‑probs (or logits) matching the phone/state order declared in `mlist`.  No source‑code patching on Julius’ side.

---

## 📚 Documentation

* **docs/usage.md** – detailed CLI flags & config files
* **docs/models.md** – how to add your own encoder, unit mapping, quantisation
* **examples/** – real‑time dictation demo, Japanese multi‑lingual grammar, batch decoding script

---

## 🛠️ Development

```bash
git clone https://github.com/your-org/centurion.git
cd centurion
pip install -e .[dev]
pre-commit install
pytest
```

---

## 🤝 Contributing
Pull requests are welcome!  Please open an issue to discuss big changes first.

### Contributors Covenant Code of Conduct
We pledge to foster an open and welcoming environment.

---

## 📜 License

Centurion is released under the MIT License – see [LICENSE](LICENSE) for details.

---

## 🏛️ Citation

If you use Centurion in academic work, please cite:

```bibtex
@software{centurion2025,
  author       = {Your Name and Contributors},
  title        = {Centurion: Julius \textbackslash{}leftrightarrow Foundation-Model Bridge},
  year         = 2025,
  url          = {https://github.com/your-org/centurion},
  note         = {Version 0.1.0}
}
```

---

## 💬 Acknowledgements
* Julius speech decoder – Nagoya Institute of Technology
* wav2vec 2.0, HuBERT, WavLM – Facebook AI, Microsoft Research
* Whisper – OpenAI

---

**Ave, decoder!**  🏛️  With Centurion, your old‑school Julius engine conquers the new frontiers of self‑supervised speech models.
