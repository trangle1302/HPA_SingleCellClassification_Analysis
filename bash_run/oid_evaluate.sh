HIERARCHY_FILE=/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-label300-segmentable-hierarchy.json
BOUNDING_BOXES=/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-bbox
IMAGE_LABELS=/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-labels
INPUT_PREDICTIONS=/home/trangle/HPA_SingleCellClassification/predictions/OID/sample_truncated_submission.csv
OUTPUT_METRICS=/home/trangle/HPA_SingleCellClassification/predictions/OID/sample_outputmetrics.csv

python models/research/object_detection/metrics/oid_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES}_expanded.csv \
    --input_annotations_labels=${IMAGE_LABELS}_expanded.csv \
    --input_class_labelmap=models/research/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
    --input_predictions=${INPUT_PREDICTIONS} \
    --output_metrics=${OUTPUT_METRICS} \
