import pandas as pd
from scipy.signal import butter, lfilter 
import numpy as np

feat_an_rgb = pd.read_pickle(r"C:/Users/Nunzi/OneDrive/Desktop/aml/aml23-ego/labels_AN/labels_S04/S04_test.pkl")
print(feat_an_rgb)
# print(np.size(feat["features"][0]["features_RGB"][0]))
# f_tot = []
# for features in feat["features"]:
#     feat_clip = features.get(f"features_RGB")
#     f_tot.append(feat_clip)
# aggregated_features = []
# for f in f_tot:
#     aggregated_features.append(np.mean(f, axis=0))
# print(aggregated_features[0])
# dataset = pd.read_pickle(r"C:\\Users\\Nunzi\\OneDrive\\Desktop\\aml\\aml23-ego\\train_val\\D1_test.pkl")

# print(dataset.keys())
# print(f"video_id: {dataset["video_id"][1]}")
# print(f"uid: {dataset["uid"][1]}")
# print(f"start_frame: {dataset["start_frame"][1]}")
# print(f"stop_frame: {dataset["stop_frame"][1]}")
# print(f"narration: {dataset["narration"][1]}")
# print(f"verb: {dataset["verb"][1]}")
# print(f"verb_class: {dataset["verb_class"][1]}")
# print(f"uid: {feat["features"][1]["uid"]}")
# print(f"video_name: {feat["features"][1]["video_name"]}")
# unique_verb_classes = list(set(dataset["verb_class"]))
# print(unique_verb_classes)