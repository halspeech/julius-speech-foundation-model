#!/usr/bin/env python3
"""
01get_logits.py — streaming wav2vec‑CTC client for Julius (offline model)
========================================================================

* Listens for **adintool** 16‑kHz/16‑bit/mono PCM packets (same socket
  protocol as the stock `dnnclient.py`).
* Runs a locally downloaded **wav2vec 2.0 / Whisper‑CTC** model **once**
  at start‑up and streams per‑frame **log₁₀ probability vectors**
  (3 503‑dim) to Julius in HTK format on port 5531.
* **No Internet needed** – the model is loaded with
  `local_files_only=True` so Hugging Face never triggers a download.
* All network/model settings can be overridden in an external cfg file
  that follows the *exact* syntax of `dnnclient.py` (add
  `--model_dir /path/to/local/model`).

Example cfg (save as *mysetup.cfg*)::

    # network
    --adinserver_host 0.0.0.0
    --adinserver_port 5532
    --julius_host      127.0.0.1
    --julius_port      5531

    # vector length (vocabulary)
    --num_output 3503

    # local model directory (contains config.json, pytorch_model.bin …)
    --model_dir /home/lisheng/models/whisper-large-v3-ctc

Run the client::

    ./client.py mysetup.cfg
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

# -----------------------------------------------------------------------------
# Defaults (can be changed via cfg)
# -----------------------------------------------------------------------------
adinserver_host: str = "localhost"
adinserver_port: int = 5532
julius_host: str = "localhost"
julius_port: int = 5531

num_output: int = 3503  # vocabulary size / vector dimension
log_base: int = 10      # Julius expects log10 probabilities

model_dir: str = "./wav2vec2-large-finetuned"  # can be local path
use_cuda: bool = torch.cuda.is_available()

# -----------------------------------------------------------------------------
# CFG parser (same syntax as Julius client.cfg)
# -----------------------------------------------------------------------------

def apply_cfg(path: Path):
    global adinserver_host, adinserver_port, julius_host, julius_port, num_output, model_dir  # noqa: E501
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, *vals = line.strip().split()
            val = vals[0]
            if key == "--adinserver_host":
                adinserver_host = val
            elif key == "--adinserver_port":
                adinserver_port = int(val)
            elif key == "--julius_host":
                julius_host = val
            elif key == "--julius_port":
                julius_port = int(val)
            elif key == "--num_output":
                num_output = int(val)
            elif key in ("--model", "--model_dir"):
                model_dir = val

# -----------------------------------------------------------------------------
# Argument parser (only cfg path is required)
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Streaming wav2vec2→Julius client")
    p.add_argument("cfg", type=Path, help="configuration file (same switches as dnnclient.py)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Networking helpers
# -----------------------------------------------------------------------------

def julius_handshake(sock: socket.socket):
    sock.sendall(struct.pack("=iiii", 12, num_output, log_base, 1))


def send_frame(sock: socket.socket, vec: np.ndarray):
    sock.sendall(struct.pack("=i", num_output * 4))
    sock.sendall(vec.astype(np.float32).tobytes())

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    apply_cfg(args.cfg)

    # 1. Load model **offline**
    print(f"[INFO] Loading model from '{model_dir}' (offline)…", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device).eval()
    print(f"[INFO] Model ready on {device}", file=sys.stderr)

    # 2. Prepare sockets
    adin_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    adin_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    adin_srv.bind((adinserver_host, adinserver_port))
    adin_srv.listen(1)
    print(f"[INFO] Waiting ADIN connection on {adinserver_host}:{adinserver_port}…", file=sys.stderr)
    adin_cli, _ = adin_srv.accept()
    print("[INFO] ADIN connected", file=sys.stderr)

    julius_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    julius_sock.connect((julius_host, julius_port))
    print(f"[INFO] Connected to Julius at {julius_host}:{julius_port}", file=sys.stderr)

    #first_handshake = False
    julius_handshake(julius_sock)
    pcm_buf = bytearray()

    # 3. Stream loop
    while True:
        header = adin_cli.recv(4)
        if not header:
            break  # lost connection
        (nbytes,) = struct.unpack("=i", header)

        # meta handshake from ADIN (we just forward the info once)
        if nbytes == 12:
            adin_cli.recv(12)
            #_meta = adin_cli.recv(12)
            #if not first_handshake:
            #    julius_handshake(julius_sock)
            #    first_handshake = True
            continue

        # End of stream
        if nbytes == -1:
            break

        # End of utterance → forward cached audio
        if nbytes == 0:
            if pcm_buf:
                #pcm_np = np.frombuffer(pcm_buf, dtype="<i2")
                pcm_np = np.frombuffer(pcm_buf, dtype="<i2").copy()
                pcm_buf.clear()
                with torch.no_grad():
                    inputs = processor(pcm_np.astype(np.float32) / 32768.0, sampling_rate=16000, return_tensors="pt", padding=False)
                    logits = model(inputs.input_values.to(device)).logits  # [1, T, V]
                    logp = torch.log_softmax(logits, dim=-1) / np.log(10)
                frames = logp.squeeze(0).cpu().numpy().astype(np.float32)
                for f in frames:
                    send_frame(julius_sock, f)
            # tell Julius sentence boundary
            julius_sock.sendall(struct.pack("=i", 0))
            continue

        # Normal PCM chunk
        chunk = bytearray()
        while len(chunk) < nbytes:
            part = adin_cli.recv(nbytes - len(chunk))
            if not part:
                break
            chunk.extend(part)
        pcm_buf.extend(chunk)

    # final terminator
    julius_sock.sendall(struct.pack("=i", 0))
    julius_sock.sendall(struct.pack("=i", -1))

    adin_cli.close()
    adin_srv.close()
    julius_sock.close()
    print("[INFO] Shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    main()

