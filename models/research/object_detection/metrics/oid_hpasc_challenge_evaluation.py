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
  flags.mark_flag_as_required('all_annotations')
  flags.mark_flag_as_required('input_predictions')
  flags.mark_flag_as_required('input_class_labelmap')
  flags.mark_flag_as_required('output_metrics')

  all_location_annotations = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-bbox_expanded.csv")
  all_label_annotations = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/OID/challenge-2019-validation-segmentation-labels_expanded.csv")
  all_label_annotations.rename(
      columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

  is_instance_segmentation_eval = False
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
  
  # Testing with 10 images
  if False: #FOrmatting the solution files, only need to be done once!
      all_annotations_hpa = pd.read_csv("/home/trangle/HPA_SingleCellClassification/GT/_solution.csv_")
      imlist = list(set(all_annotations_hpa.ID))[:10]
      all_annotations_hpa = all_annotations_hpa[all_annotations_hpa.ID.isin(imlist)]
      all_annotations = pd.DataFrame()
      for i, row in all_annotations_hpa.iterrows():
          pred_string = row.PredictionString.split(" ")
          for k in range(0, len(pred_string), 7):
              compressed_mask = base64.b64decode(pred_string[k+6])
              rle_encoded_mask = zlib.decompress(compressed_mask)
              decoding_dict = {
                  'size': [im_height, im_width],
                  'counts': rle_encoded_mask
              }
              mask_tensor = coco_mask.decode(decoding_dict)
              boxes = utils._to_normalized_box(mask_tensor) # ymin, xmin, ymax, xmax
              line = {
                      "ImageID":row.ID,
                      "ImageWidth":row.ImageWidth,
                      "ImageHeight":row.ImageHeight,
                      "ConfidenceImageLabel": 1, 
                      "LabelName": pred_string[k], 
                      "XMin":pred_string[k+1],
                      "YMin":pred_string[k+2],
                      "XMax":pred_string[k+3],
                      "YMax":pred_string[k+4],
                      "IsGroupOf":pred_string[k+5],
                      "Mask": np.expand_dims(mask_tensor, 0), #pred_string[k+6]
                }
              print(line)
              """
              binary_mask = decodeToBinaryMask(pred_string[k+6], row.ImageWidth, row.ImageHeight)
              plt.figure()
              plt.imshow(binary_mask)
              """
              #all_annotations = all_annotations.append(line, ignore_index=True)
      all_annotations_hpa.to_csv("/home/trangle/HPA_SingleCellClassification/all_annotations.csv", index=False)
  all_annotations = pd.read_csv("/home/trangle/HPA_SingleCellClassification/GT/all_annotations.csv")
  all_annotations['ImageHeight'] = all_annotations['ImageHeight'].astype(int)
  all_annotations['ImageWidth'] = all_annotations['ImageWidth'].astype(int)
  all_annotations['Mask'] = all_annotations['ImageWidth'].astype(int)
  class_label_map, categories = _load_labelmap("/home/trangle/HPA_SingleCellClassification/GT/hpa_single_cell_label_map.pbtxt")
  #class_label_map, categories = _load_labelmap("/home/trangle/HPA_SingleCellClassification/models/research/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt")
  challenge_evaluator = (
      object_detection_evaluation.OpenImagesChallengeEvaluator(
          categories, evaluate_masks=is_instance_segmentation_eval))

  #submissions = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/OID/sample_truncated_submission.csv")
  submissions = pd.read_csv("/home/trangle/HPA_SingleCellClassification/predictions/bestfitting/bestfitting_submission.csv")
  submissions = submissions[submissions.ID.isin(imlist)]
  all_predictions = pd.DataFrame()
  for i, row in submissions.iterrows():
      try:
          pred_string = row.PredictionString.split(" ")
          for k in range(0, len(pred_string), 3):
                label = pred_string[k]
                conf = pred_string[k + 1]
                rle = pred_string[k + 2]
                line = {
                        "ImageID":row.ID,
                        "ImageWidth":row.ImageWidth,
                        "ImageHeight":row.ImageHeight,
                        "LabelName":str(label),
                        "Score":conf,
                        "Mask":rle,
                }
                all_predictions = all_predictions.append(line, ignore_index=True)
      except:
          continue
      
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

  with open("/home/trangle/HPA_SingleCellClassification/examples/OID/sample_outputmetrics.csv", 'w') as fid:
    io_utils.write_csv(fid, metrics)


if __name__ == '__main__':
  app.run(main)
def _get_bbox(segment, image_widths,image_heights):
     compressed_mask = base64.b64decode(pred_string[k+6])
     rle_encoded_mask = zlib.decompress(compressed_mask)
     decoding_dict = {
                  'size': [im_height, im_width],
                  'counts': rle_encoded_mask
    }
    mask_tensor = coco_mask.decode(decoding_dict)
    boxes = utils._to_normalized_box(mask_tensor)