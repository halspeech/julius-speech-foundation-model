# <img src="picture.png" alt="Centurion mascot" width="120px" align="left" />

# Centurion (foundation model plugin for Julius decoder)

> **Centurion** is a lightweight plug-in that lets the classic **[Julius](https://github.com/julius-speech/julius)** decoder speak the language of modern speech‑foundation models—**wav2vec 2.0, WavLM, HuBERT, Whisper**, and more.  Keep Julius’ blazing‑fast decoding (no Kaldi lattice wrangling required), adding the acoustic power of giant self‑supervised encoders.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **One‑line drop‑in** | Load any HuggingFace or local *.pt/bin* encoder and start streaming to Julius in minutes. |
| **Model Zoo** | Built‑in presets for wav2vec 2.0 (base/large), WavLM (base/large), HuBERT (base/large/xl), and Whisper (large‑v3). |
| **Streaming & offline** | Works with Julius’ *vecnet* real‑time protocol **and** batch HTK logit dumping. |
| **GPU / CPU** | PyTorch ≥2.0, CUDA optional but loved. |

---

## 🚀 Quick Start

```bash
# 1️⃣  Install (Python 3.9+)
conda env create -f environment.yml
# To activate this environment, use
conda activate whisper_features
# To deactivate an active environment, use
conda deactivate

# 2️⃣  Download a model (we need to make some changes to the model, making the model work frame-by-frame)
wav2vec 2.0 (base/large),
WavLM (base/large),
HuBERT (base/large/xl),
and Whisper (large‑v3)

# 3️⃣  Launch the Centurion in two ways: online or offline
both follows Julius's way

# 4️⃣  Fire up Julius
julius -C main.jconf 
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
Audio → Centurion → Julius vecnet → LM rescoring → Final hypothesis
```
*Centurion* exposes Julius‑compatible feature streams: frame‑level log‑probs (or logits) matching the phone/state order declared in `mlist`.  
No source‑code patching on Julius’ side.

---

## 🤝 Contributing
Pull requests are welcome!  Please open an issue to discuss big changes first.

## 📜 License

Centurion is released under the MIT License – see [LICENSE](LICENSE) for details.

---

## 🏛️ Citation

If you use Centurion in academic work, please cite:

```bibtex
@software{centurion2025,
  author       = {Sheng Li},
  title        = {Centurion: foundation model plugin for Julius decoder},
  year         = 2025,
  url          = {https://github.com/halspeech/centurion},
  note         = {Version 0.1.0}
}
```

---

## 💬 Acknowledgements
* Julius speech decoder – Nagoya Institute of Technology and Kyoto University
* wav2vec 2.0, HuBERT, WavLM – Facebook AI, Microsoft Research
* Whisper – OpenAI

---

**Ave, decoder!**  🏛️  With Centurion, your old‑friend Julius engine conquers the new frontiers of speech foundation models.
