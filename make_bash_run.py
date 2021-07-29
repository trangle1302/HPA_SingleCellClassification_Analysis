import os
import argparse
import re

parser = argparse.ArgumentParser(description="Creating bash script")
parser.add_argument("-folder", type=str, help="path to submission folder")

args = parser.parse_args()
d = "/home/trangle/HPA_SingleCellClassification/"

file = os.listdir(args.folder)[0]
name = args.folder.split("/")[-2]
name = re.search(r"\d+", name)[0]

ALL_GT_ANNOTATIONS = f"{d}/GT/all_annotations_nominus1_nodup.csv"
LABEL_MAP = f"{d}/GT/hpa_single_cell_label_map.pbtxt"
INPUT_PREDICTIONS = os.path.join(d, args.folder, file)
OUTPUT_METRICS = os.path.join(
    d, args.folder, file.replace(".csv", "_outputmetrics.csv")
)

# Writing to bash file
bash_file = f"{d}/bash_run/hpasc_evaluate_{name}.sh"
if os.path.exists(bash_file):
    print("found an existing bash file, removing...")
    os.remove(bash_file)
f = open(bash_file, "a+")
f.write(
    f"python models/research/object_detection/metrics/oid_hpasc_challenge_evaluation.py --all_annotations={ALL_GT_ANNOTATIONS} --input_class_labelmap={LABEL_MAP} --input_predictions={INPUT_PREDICTIONS} --output_metrics={OUTPUT_METRICS} \n"
)
f.close()
