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
"""Common IO utils used in offline metric computation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import pickle
import os

def write_csv(fid, metrics):
  """Writes metrics key-value pairs to CSV file.

  Args:
    fid: File identifier of an opened file.
    metrics: A dictionary with metrics to be written.
  """
  metrics_writer = csv.writer(fid, delimiter=',')
  for metric_name, metric_value in metrics.items():
    metrics_writer.writerow([metric_name, str(metric_value)])

def save_obj(obj, dir, name):
    if not os.path.isdir(dir + '/obj/'):
      os.makedirs(dir + '/obj/')
    with open(dir + '/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, name):
    with open(dir + '/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)