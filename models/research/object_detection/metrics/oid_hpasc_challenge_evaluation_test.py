# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs evaluation using OpenImages groundtruth and predictions.

Uses Open Images Challenge 2018, 2019 metrics

Example usage:
python models/research/object_detection/metrics/oid_od_challenge_evaluation.py \
    --input_annotations_boxes=/path/to/input/annotations-human-bbox.csv \
    --input_annotations_labels=/path/to/input/annotations-label.csv \
    --input_class_labelmap=/path/to/input/class_labelmap.pbtxt \
    --input_predictions=/path/to/input/predictions.csv \
    --output_metrics=/path/to/output/metric.csv \
    --input_annotations_segm=[/path/to/input/annotations-human-mask.csv] \

If optional flag has_masks is True, Mask column is also expected in CSV.

CSVs with bounding box annotations, instance segmentations and image label
can be downloaded from the Open Images Challenge website:
https://storage.googleapis.com/openimages/web/challenge.html
The format of the input csv and the metrics itself are described on the
challenge website as well.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from absl import app
from absl import flags
import pandas as pd
from google.protobuf import text_format

from object_detection.metrics import io_utils
from object_detection.metrics import oid_challenge_evaluation_utils as utils
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import object_detection_evaluation
import time
import tqdm

flags.DEFINE_string('all_annotations', None,
                    'File with groundtruth boxes and label annotations.')
flags.DEFINE_string(
    'input_predictions', None,
    """File with detection predictions; NOTE: no postprocessing is applied in the evaluation script."""
)
flags.DEFINE_string('input_class_labelmap', None,
                    'Open Images Challenge labelmap.')
flags.DEFINE_string('output_metrics', None, 'Output file with csv metrics.')
flags.DEFINE_string(
    'input_annotations_segm', None,
    'File with groundtruth instance segmentation annotations [OPTIONAL].')

FLAGS = flags.FLAGS
FLAGS.all_annotations="/home/trangle/HPA_SingleCellClassification/GT/all_annotations.csv"
FLAGS.input_class_labelmap="/home/trangle/HPA_SingleCellClassification/GT/hpa_single_cell_label_map.pbtxt"
FLAGS.input_predictions="/home/trangle/HPA_SingleCellClassification/predictions/bestfitting/en_m11_4image_7cell_0509_complete590.csv"
FLAGS.output_metrics="/home/trangle/HPA_SingleCellClassification/predictions/bestfitting/en_m11_4image_7cell_0509_complete590_output_metrics.csv"

def _load_labelmap(labelmap_path):
  """Loads labelmap from the labelmap path.

  Args:
    labelmap_path: Path to the labelmap.

  Returns:
    A dictionary mapping class name to class numerical id
    A list with dictionaries, one dictionary per category.
  """

  label_map = string_int_label_map_pb2.StringIntLabelMap()
  with open(labelmap_path, 'r') as fid:
    label_map_string = fid.read()
    text_format.Merge(label_map_string, label_map)
  labelmap_dict = {}
  categories = []
  for item in label_map.item:
    labelmap_dict[item.name] = item.id
    categories.append({'id': item.id, 'name': item.name})
  return labelmap_dict, categories


def main(unused_argv):
  s = time.time()
  flags.mark_flag_as_required('all_annotations')
  flags.mark_flag_as_required('input_predictions')
  flags.mark_flag_as_required('input_class_labelmap')
  flags.mark_flag_as_required('output_metrics')

  is_instance_segmentation_eval = False
  """
  all_location_annotations = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-bbox_expanded.csv")
  all_label_annotations = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-labels_expanded.csv")
  all_label_annotations.rename(
      columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

  if False: #FLAGS.input_annotations_segm:
    is_instance_segmentation_eval = True
    all_segm_annotations = pd.read_csv(FLAGS.input_annotations_segm)
    # Note: this part is unstable as it requires the float point numbers in both
    # csvs are exactly the same;
    # Will be replaced by more stable solution: merge on LabelName and ImageID
    # and filter down by IoU.
    all_location_annotations = utils.merge_boxes_and_masks(
        all_location_annotations, all_segm_annotations)
  all_annotations = pd.concat([all_location_annotations, all_label_annotations])
  """ 
  
  if False: #Formatting the solution files, only need to be done once!!
      all_annotations_hpa = pd.read_csv("/home/trangle/HPA_SingleCellClassification/GT/_solution.csv_")
      # Testing with 10 images
      #imlist = list(set(all_annotations_hpa.ID))[:10]
      #all_annotations_hpa = all_annotations_hpa[all_annotations_hpa.ID.isin(imlist)]
      all_annotations = pd.DataFrame()
      for i, row in all_annotations_hpa.iterrows():
          pred_string = row.PredictionString.split(" ")
          for k in range(0, len(pred_string), 7):
              boxes = utils._get_bbox(pred_string[k+6], row.ImageWidth,row.ImageWidth) # ymin, xmin, ymax, xmax
              line = {
                      "ImageID":row.ID,
                      "ImageWidth":row.ImageWidth,
                      "ImageHeight":row.ImageHeight,
                      "ConfidenceImageLabel": 1, 
                      "LabelName": pred_string[k], 
                      "XMin":boxes[1],
                      "YMin":boxes[0],
                      "XMax":boxes[3],
                      "YMax":boxes[2],
                      "IsGroupOf":pred_string[k+5],
                      "Mask": pred_string[k+6]
                }
              """
              binary_mask = decodeToBinaryMask(pred_string[k+6], row.ImageWidth, row.ImageHeight)
              plt.figure()
              plt.imshow(binary_mask)
              """
              all_annotations = all_annotations.append(line, ignore_index=True)
      all_annotations_hpa.to_csv("/home/trangle/HPA_SingleCellClassification/all_annotations.csv", index=False)
  all_annotations = pd.read_csv(FLAGS.all_annotations)
  all_annotations['ImageHeight'] = all_annotations['ImageHeight'].astype(int)
  all_annotations['ImageWidth'] = all_annotations['ImageWidth'].astype(int)
  
  class_label_map, categories = _load_labelmap(FLAGS.input_class_labelmap)
  challenge_evaluator = (
      object_detection_evaluation.OpenImagesChallengeEvaluator(
          categories, evaluate_masks=is_instance_segmentation_eval, matching_iou_threshold=0.6))
 
  submissions = pd.read_csv(FLAGS.input_predictions)      
  # Testing with 10 images
  #submissions = submissions[submissions.ID.isin(all_annotations.ImageID)]
  print(f"{FLAGS.input_predictions} prediction loaded")
  all_predictions = pd.DataFrame()
  f = open(FLAGS.input_predictions.replace(".csv","_formatted.csv"), "a+")
  f.write("ImageID,ImageWidth,ImageHeight,LabelName,Score,Mask\n")
  for i, row in tqdm.tqdm(submissions.iterrows(),total=submissions.shape[0]):
      try:
          pred_string = row.PredictionString.split(" ")
          for k in range(0, len(pred_string), 3):
                label = pred_string[k]
                conf = pred_string[k + 1]
                rle = pred_string[k + 2]
                line = f"{row.ID},{row.ImageWidth},{row.ImageHeight},{str(label)},{conf},{rle}\n"
                f.write(line)
                """
                line = {
                        "ImageID":row.ID,
                        "ImageWidth":row.ImageWidth,
                        "ImageHeight":row.ImageHeight,
                        "LabelName":str(label),
                        "Score":conf,
                        "Mask":rle,
                }
                all_predictions = all_predictions.append(line, ignore_index=True)
                """
      except:
          continue

  
  #all_predictions.to_csv(FLAGS.input_predictions.replace(".csv","_formatted.csv"), index=False)
  
  """
  images_processed = 0
  for _, groundtruth in enumerate(all_annotations.groupby('ImageID')):
    logging.info('Processing image %d', images_processed)
    image_id, image_groundtruth = groundtruth
    groundtruth_dictionary = utils.build_groundtruth_dictionary(
        image_groundtruth, class_label_map)
    challenge_evaluator.add_single_ground_truth_image_info(
        image_id, groundtruth_dictionary)

    prediction_dictionary = utils.build_predictions_dictionary(
        all_predictions.loc[all_predictions['ImageID'] == image_id],
        class_label_map)
    challenge_evaluator.add_single_detected_image_info(image_id,
                                                       prediction_dictionary)
    images_processed += 1

  metrics = challenge_evaluator.evaluate()

  with open(FLAGS.output_metrics, 'w') as fid:
    io_utils.write_csv(fid, metrics)
  """
  print(f'Finished in {(time.time() - s)/3600} hour')

if __name__ == '__main__':
  app.run(main)