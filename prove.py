import pandas as pd
from scipy.signal import butter, lfilter 
import numpy as np
'''
def load_data(filename):
    emg_data = pd.read_pickle(f"/Users/federicobuccellato/Desktop/aml23-ego/EMG_data/{filename}")
    return emg_data

def data_loader(emg_ann):
    distinct_files = list(map(lambda x: x.split('.')[0].split('_'), emg_ann['file'].unique()))
    final_df = pd.DataFrame()

    for file in distinct_files:
        subject_id, video = file
        file_name = f'{subject_id}_{video}.pkl'

        df_curr_file = emg_ann.query(f"file == '{file_name}'")

        indexes = list(df_curr_file['index'])
        data_byKey = load_data(file_name).loc[indexes] #get data of subject and id
        data_byKey['subject_id']=subject_id
        data_byKey['video']=video

        final_df = pd.concat([final_df, data_byKey], ignore_index=True)

    return final_df

def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

splits = ['train', 'test']
resampled_Fs = 10  # Define rate for all sensors to interpolate
segment_duration_s = 10
save_dir = '//Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed'
x=0
for split in splits:
    emg_annotations = pd.read_pickle(f'/Users/federicobuccellato/Desktop/aml23-ego/action-net/ActionNet_test.pkl')
    emg = data_loader(emg_annotations)

    for key in ['myo_right', 'myo_left']:
        for i, row in emg.iterrows():
            # Estrazione e rettifica dei dati EMG
            data = abs(row[key + '_readings'])
            t = row[key + '_timestamps']
            
            # Calcolo della frequenza di campionamento
            Fs = (t.size - 1) / (t[-1] - t[0])
            
            # Applicazione del filtro passa-basso
            filtered_data = lowpass_filter(data, 5, Fs)
            
            # Calcolo del minimo e massimo per la normalizzazione
            min_val = np.min(filtered_data)
            max_val = np.max(filtered_data)
            
            # Applicazione della normalizzazione
            data_normalized = 2 * (filtered_data - min_val) / (max_val - min_val) - 1
            
            # Aggiornamento del DataFrame con i dati normalizzati
            emg.at[i, key + '_readings'] = data_normalized


        new_emg_samples = []
        for _, row in emg.iterrows():
            start, end = row['start'], row['stop']
            duration_s = end - start

            if duration_s < 2 * segment_duration_s:
                new_emg_samples.append(row)
            else:
                num_segments = int(duration_s // segment_duration_s)
                segment_times = np.linspace(start, end - segment_duration_s, num=num_segments, endpoint=True)
                total_readings = resampled_Fs * segment_duration_s

                for segment_start in segment_times:
                    segment_end = segment_start + segment_duration_s
                    subaction = row.copy()

                    for key in ['myo_right', 'myo_left']:
                        data = np.array(row[f'{key}_readings'])
                        time_s = np.array(row[f'{key}_timestamps'])

                        time_indexes = np.where((time_s >= segment_start) & (time_s <= segment_end))[0]

                        while len(time_indexes) < total_readings:
                            if time_indexes[0] > 0:
                                time_indexes = np.insert(time_indexes, 0, time_indexes[0] - 1)
                            elif time_indexes[-1] < len(time_s) - 1:
                                time_indexes = np.append(time_indexes, time_indexes[-1] + 1)
                            else:
                                time_indexes = np.append(time_indexes, time_indexes[-1])
                                break

                        time_indexes = time_indexes[:total_readings]

                        subaction[f'{key}_timestamps'] = time_s[time_indexes]
                        subaction[f'{key}_readings'] = data[time_indexes, :]

                    new_emg_samples.append(subaction)

        processed_emg = pd.DataFrame(new_emg_samples).reset_index(drop=True)
        processed_emg.to_pickle(f'{save_dir}/ActionNet_EMG_{split}.pkl')


a = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/ActionNet_EMG_test.pkl')
b = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/action-net/ActionNet_test.pkl')
print(a)
print(b)'''


'''rgb_annotations = pd.read_pickle(f'/Users/federicobuccellato/Desktop/aml23-ego/saved_features/feat_5_True_D1_train.pkl')

# Visualizza il contenuto di una specifica chiave
for key in rgb_annotations.keys():
    print(f"Chiave: {key}")

print(rgb_annotations['features'][0]['features_RGB'])'''
'''
center_points = np.linspace(5,95, 10,  dtype=int)
frame_indices = []
print(center_points)
for center in center_points:
    frames = [(center + 2 * (x - 10 // 2)) 
            for x in range(10)]
    
    frames = [max(0, min(f, 100 - 1)) for f in frames]
    frame_indices.extend(frames)

print(frame_indices)
'''
# feat = pd.read_pickle(r"C:\\Users\\Nunzi\\OneDrive\\Desktop\\aml\\aml23-ego\\saved_features\\feat_16_EK_D1_test_dense.pkl")
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
import torch

print('cuda' if torch.cuda.is_available() else 'cpu')
