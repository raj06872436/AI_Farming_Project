#!/bin/bash
# Overnight GPU jobs for LesionNet (run from the project folder in Ubuntu with tf_gpu active).
cd "$(dirname "$0")"
source ~/tf_gpu/bin/activate 2>/dev/null
mkdir -p research/lesion_net/logs
python research/lesion_net/train_lesionnet.py --pool gap --lambda-seg 0 --bg-swap 0 --tag A0_plain 2>&1 | tee research/lesion_net/logs/train_A0_plain.log
python research/lesion_net/train_lesionnet.py --backbone MobileNetV2 --tag A5_mobilenet 2>&1 | tee research/lesion_net/logs/train_A5_mobilenet.log
python research/lesion_net/evaluate.py 2>&1 | tee research/lesion_net/logs/evaluate.log
echo "ALL DONE $(date)" | tee research/lesion_net/logs/DONE.txt
