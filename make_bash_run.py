import os
import argparse

parser = argparse.ArgumentParser(description='Creating bash script')
parser.add_argument('-folder', type=str, nargs='+', help='path to submission folder')

args = parser.parse_args()
d = "/home/trangle/HPA_SingleCellClassification/"
f = open(FLAGS.input_predictions.replace(".csv","_formatted.csv"), "a+")
f.write(f"ALL_GT_ANNOTATIONS={d}/GT/all_annotations_privatetest.csv\n")
f.write(f"LABEL_MAP={d}/GT/all_annotations_privatetest.csv\n")
f.write(f"INPUT_PREDICTIONS={d}/{args.folder}\n")
f.write(f"OUTPUT_METRICS={d}/{args.folder}\n")

f.write("python models/research/object_detection/metrics/oid_hpasc_challenge_evaluation.py \ \n")
f.write("--all_annotations=${ALL_GT_ANNOTATIONS} \ \n")
    --input_class_labelmap=${LABEL_MAP} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --output_metrics=${OUTPUT_METRICS} \
