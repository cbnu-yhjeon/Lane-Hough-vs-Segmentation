#!/usr/bin/env bash
# 전체 파이프라인 재현: manifest → L2 학습 · 추론 → 평가 → 집계 · 그림 · 속도
set -euo pipefail
PY="${PY:-python}"
$PY -m lanebench.data.manifest
$PY -m lanebench.segmentation.train --epochs 5
$PY -m lanebench.segmentation.infer --split val
$PY -m lanebench.eval.run --methods L0 L1 L1+ --tag classical
$PY -m lanebench.eval.run --methods L2 --tag L2
$PY -m lanebench.eval.by_condition
$PY -m lanebench.eval.figures
$PY -m lanebench.eval.qualitative
$PY -m lanebench.eval.fps
