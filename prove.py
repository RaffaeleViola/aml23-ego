import pandas as pd

data = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/labels_AN/labels_S04/S04_train.pkl')
emg = pd.read_pickle('//Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/EMG_ActionNet_S04_train.pkl')
rgb = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/EMG_data/S02_3.pkl')

print(data['start_frame'].iloc[400])
print(data['stop_frame'].iloc[400])
print(data['verb'].iloc[400])