#!/usr/bin/env bash
# BDD100K 이미지 · 라벨 내려받기 → ~/data
#
# 공식 서버(dl.cv.ethz.ch)가 닫혀 있어 공개 재배포본을 씁니다. BDD100K 라이선스
# (연구 · 교육 목적)를 따르며, 원본 데이터는 저장소에 넣지 않습니다.
#   이미지 train 70k : HF yiyi159/bdd100k-train (100k_images_train.zip, 4.0 GB)
#   이미지 val 10k   : HF hirundo-io/bdd100k-val (bdd100k_val_hirundo.zip, 0.57 GB)
#   라벨             : YOLOP 배포 det_annotations.zip (BDD100K v1 per-image JSON,
#                      박스 + 차선 poly2d + 주행가능영역 poly2d + 날씨/시간대/장면)
set -euo pipefail
DATA="${DATA:-$HOME/data}"
mkdir -p "$DATA/bdd100k_zips" "$DATA/bdd100k_yolop"
HF=https://huggingface.co/datasets

cd "$DATA/bdd100k_zips"
[ -f 100k_images_train.zip ] || curl -L -o 100k_images_train.zip "$HF/yiyi159/bdd100k-train/resolve/main/100k_images_train.zip"
[ -f bdd100k_val_hirundo.zip ] || curl -L -o bdd100k_val_hirundo.zip "$HF/hirundo-io/bdd100k-val/resolve/main/bdd100k_val_hirundo.zip"
unzip -q -n 100k_images_train.zip -d "$DATA"
unzip -q -n bdd100k_val_hirundo.zip "bdd100k/images/*" -d "$DATA"

cd "$DATA/bdd100k_yolop"
[ -f det_annotations.zip ] || uvx gdown 1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl -O det_annotations.zip
unzip -q -n det_annotations.zip

echo "images: $DATA/bdd100k/images/100k/{train,val}"
echo "labels: $DATA/bdd100k_yolop/data2/zwt/bdd/bdd100k/labels/100k/{train,val}"
