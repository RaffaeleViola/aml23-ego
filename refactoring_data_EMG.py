import pickle
import pandas as pd

def filter_refactor_emg_data(input_path, output_path, filter_value):
    with open(input_path, 'rb') as file:
        data = pickle.load(file)

    transformed_data = {'features': []}

    for i in range(len(data)):
        #if data['file'][i] == filter_value: #da inserire quando si vuole prendere il dataset solo per S04
            record = {
                'uid': data['uid'][i],
                'features_EMG': data['myo_combined_readings'][i]
            }
            transformed_data['features'].append(record)

    with open(output_path, 'wb') as file:
        pickle.dump(transformed_data, file)

    print(f"Filtraggio completato e file salvato: {output_path}")


def create_labels_data(input_path, output_path):
    with open(input_path, 'rb') as file:
        data = pickle.load(file)

    records = []

    for i in range(len(data)):
        record = {
            'uid': data['uid'][i],
            'description': data['description'][i],
            'verb_class': data['labels'][i]  
        }
        records.append(record)

    labels_data = pd.DataFrame(records)
    labels_data.to_pickle(output_path)

    print(f"DataFrame delle etichette creato e file salvato: {output_path}")


def create_S04_labels_data(timestamps_path, emg_data_path, output_dir):
    timestamps_data = pd.read_pickle(timestamps_path)
    calibration_val = timestamps_data['start'].iloc[0]

    emg_data = pd.read_pickle(emg_data_path)
    emg_data = emg_data[emg_data['file'] == 'S04_1.pkl']

    emg_data = emg_data.rename(columns={
        'file': 'video_id',
        'description': 'narration',
        'labels': 'verb_class'
    })
    emg_data['participant_id'] = 'S04'
    emg_data['video_id'] = 'S04_1'

    FPS = 30  
    emg_data['start_frame'] = ((emg_data['start'] - calibration_val) * FPS).astype(int)
    emg_data['stop_frame'] = ((emg_data['stop'] - calibration_val) * FPS).astype(int)

    emg_data['verb'] = emg_data['narration']
    emg_data = emg_data.rename(columns={
        'start': 'start_timestamp',
        'stop': 'stop_timestamp'
    })

    emg_data = emg_data.drop(['myo_combined_readings'], axis=1)
    emg_data = emg_data[
        ['uid', 'participant_id', 'video_id', 'narration', 'start_timestamp', 'stop_timestamp', 'start_frame',
         'stop_frame', 'verb', 'verb_class']]

    emg_data.to_pickle(output_dir)

    print(f"Annotazioni RGB create e file salvato: {output_dir}")


if __name__ == "__main__":
    '''filter_refactor_emg_data(
        input_path='/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/ActionNet_EMG_train_raw.pkl',
        output_path='/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/EMG_ActionNet_train_S04.pkl',
        filter_value='S04_1.pkl'
    )'''

    '''create_labels_data(
        input_path='/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/ActionNet_EMG_train_raw.pkl',
        output_path='/Users/federicobuccellato/Desktop/aml23-ego/labels_AN/labels_full_dataset/ActionNet_train.pkl'
    )
    '''
    '''create_S04_labels_data(
        timestamps_path='/Users/federicobuccellato/Desktop/aml23-ego/EMG_data/S04_1.pkl',
        emg_data_path='/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/ActionNet_EMG_test_raw.pkl',
        output_dir='/Users/federicobuccellato/Desktop/aml23-ego/labels_AN/labels_S04/S04_test.pkl'
    )'''
