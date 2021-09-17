import pandas as pd
from sklearn.metrics import f1_score
import collections
import re
import string

def _normalize_answer(s):
  """Lowers text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score(prediction, ground_truth):
  """Computes F1 score by comparing prediction to ground truth."""
  prediction_tokens = _normalize_answer(prediction).split()
  ground_truth_tokens = _normalize_answer(ground_truth).split()
  prediction_counter = collections.Counter(prediction_tokens)
  ground_truth_counter = collections.Counter(ground_truth_tokens)
  common = prediction_counter & ground_truth_counter
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def conf_to_label(pred_str, conf_threshold = 0.5):
    d = eval(pred_str)
    labels = None
    for k, v in d.items():
        if float(v) > conf_threshold:
            if labels == None:
                labels = k
            else:
                labels = "|".join([labels, k])
    return labels


if __name__ == "__main__":
    bestfitting_match = pd.read_csv("/data/kaggle-dataset/mAPscoring/bestfitting/IOU_p_merged.csv")
    bestfitting_match["Predicted_cell_label_formatted"] = "" 
    for i, row in bestfitting_match.iterrows():
        pred_str = row.Predicted_cell_label
        #print(pred_str)
        if pred_str != 'None':
            formatted_str = conf_to_label(pred_str, conf_threshold=0.2)
            if formatted_str != None:
                bestfitting_match.loc[i,"Predicted_cell_label_formatted"] = formatted_str
    
    bestfitting_match = bestfitting_match[bestfitting_match.GT_cell_label!='None']
    n_GT = bestfitting_match.GT_cell_label.value_counts().sum()
    n_matched = sum(bestfitting_match.Predicted_cell_label!= "None")
    print(f"Predicted {n_matched}/{n_GT} cells")
    
    num_class = len(eval(bestfitting_match.Predicted_cell_label[0]))
    tp_fp_labels_per_class = [[] for _ in range(num_class)]
    f1 = 0
    df = bestfitting_match[["Image","Predicted_cell_label_formatted", "locations_reindex"]]
    for im_df in df.groupby("Image") :
        f1_img = 0 
        for i, row in im_df[1].iterrows(): 
            f1_img += _f1_score(str(row.Predicted_cell_label_formatted), str(row.locations_reindex))
        print(im_df[0], f1_img/len(im_df[1]))
        f1 += f1_img/len(im_df[1])
    print("Avg of avg F1 per cell per image", f1/df.Image.nunique())
    
    #macrof1 = f1_score(y_true, y_pred, average='macro')
    #microf1 = f1_score(y_true, y_pred, average='micro')