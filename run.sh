#!/usr/bin/env bash
set -e

PROJECT_DIR="$(pwd)"

echo "[INFO] Setting up project..."

# Install Python deps
if [ -f "requirements.txt" ]; then
	echo "[INFO] Installing Python requirements..."
	pip install -r requirements.txt
else
	echo "[WARN] No requirements.txt found!"
fi

# Install SUMO via apt
if ! command -v sumo >/dev/null 2>&1; then
	echo "[INFO] Installing SUMO via apt..."
	apt-get update -y
	apt-get install -y sumo sumo-tools sumo-doc
fi

# Set SUMO_HOME
export SUMO_HOME="$(dirname $(dirname $(which sumo)))"
echo "[INFO] SUMO_HOME set to: $SUMO_HOME"

echo "[INFO] Running training..."
python train.py
