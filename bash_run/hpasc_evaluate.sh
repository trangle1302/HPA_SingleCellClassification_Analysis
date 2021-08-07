ALL_GT_ANNOTATIONS=/home/trangle/HPA_SingleCellClassification/GT/all_annotations_nominus1_nodup.csv
LABEL_MAP=/home/trangle/HPA_SingleCellClassification/GT/hpa_single_cell_label_map.pbtxt
INPUT_PREDICTIONS=/home/trangle/HPA_SingleCellClassification/predictions/bestfitting/en_m11_4image_7cell_0509_complete590.csv
OUTPUT_METRICS=/home/trangle/HPA_SingleCellClassification/predictions/bestfitting/en_m11_4image_7cell_0509_complete590_outputmetrics_privatetest_kaggle.csv

python models/research/object_detection/metrics/oid_hpasc_challenge_evaluation_test.py \
    --all_annotations=${ALL_GT_ANNOTATIONS} \
    --input_class_labelmap=${LABEL_MAP} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --output_metrics=${OUTPUT_METRICS} \
