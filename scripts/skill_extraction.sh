#!/bin/sh


FOLD_COUNT=4
SAMPLING_COUNT=6
MODEL_NAMES="Qwen2.5-7B-Instruct"
DATASETS="alfworld babyai sc"

##################################################
echo "Running skill extraction with:"
echo " - Fold count: $FOLD_COUNT"
echo " - Sampling count: $SAMPLING_COUNT"
echo " - Model names: $MODEL_NAMES"
echo " - Datasets: $DATASETS"

python ./skill_extraction/extract_skills.py \
  --fold_count "$FOLD_COUNT" \
  --sampling_count "$SAMPLING_COUNT" \
  --model_names $MODEL_NAMES \
  --datasets $DATASETS

##################################################
echo "Running node embedding with:"
echo " - Fold count: $FOLD_COUNT"
echo " - Sampling count: $SAMPLING_COUNT"
echo " - Model names: $MODEL_NAMES"
echo " - Datasets: $DATASETS"

python ./skill_extraction/embed_skills.py \
  --fold_count "$FOLD_COUNT" \
  --sampling_count "$SAMPLING_COUNT" \
  --model_names $MODEL_NAMES \
  --datasets $DATASETS


##################################################
echo "Running meta_info_embedding..."
python skill_extraction/meta_info_embedding.py
sleep 1.0
echo "Running retrieve_domain..."
python skill_extraction/retrieve_domain.py
sleep 1.0

echo "Running segment retrieve with:"
echo " - Fold count: $FOLD_COUNT"
echo " - Sampling count: $SAMPLING_COUNT"
echo " - Model names: $MODEL_NAMES"
echo " - Datasets: $DATASETS"

python ./skill_extraction/retrieve_segment.py \
  --fold_count "$FOLD_COUNT" \
  --sampling_count "$SAMPLING_COUNT" \
  --model_names $MODEL_NAMES \
  --datasets $DATASETS