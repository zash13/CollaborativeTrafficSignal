#!/usr/bin/env bash
set -e

PROJECT_DIR="$(pwd)"

echo "[INFO] Setting up project..."

if [ -f "requirements.txt" ]; then
	echo "[INFO] Installing Python requirements..."
	pip install -r requirements.txt
else
	echo "[WARN] No requirements.txt found!"
fi

if command -v sumo >/dev/null 2>&1; then
	echo "[INFO] System SUMO found, using it."
	export SUMO_HOME="$(dirname $(dirname $(which sumo)))"
else
	echo "[INFO] No system SUMO found, building from submodule..."
	echo "[INFO] Updating submodules..."
	git submodule update --init --recursive
	cd sumo_home
	cmake .
	make -j"$(nproc)"
	export SUMO_HOME="$(pwd)"
	cd "$PROJECT_DIR"
fi

echo "[INFO] SUMO_HOME set to: $SUMO_HOME"

echo "[INFO] Running training..."
python train.py
