#!/bin/bash
set -e
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip uninstall -y torchaudio
pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
