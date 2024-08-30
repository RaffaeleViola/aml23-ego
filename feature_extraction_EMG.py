import pandas as pd
from scipy.signal import butter, lfilter
from scipy import interpolate
import numpy as np

activity_labels = {
        'Get/replace items from refrigerator/cabinets/drawers': 0,
        'Peel a cucumber': 1,
        'Clear cutting board': 2,
        'Slice a cucumber': 3,
        'Peel a potato': 4,
        'Slice a potato': 5,
        'Slice bread': 6,
        'Spread almond butter on a bread slice': 7,
        'Spread jelly on a bread slice': 8,
        'Open/close a jar of almond butter': 9,
        'Pour water from a pitcher into a glass': 10,
        'Clean a plate with a sponge': 11,
        'Clean a plate with a towel': 12,
        'Clean a pan with a sponge': 13,
        'Clean a pan with a towel': 14,
        'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
        'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
        'Stack on table: 3 each large/small plates, bowls': 17,
        'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
        'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}

def load_data(filename):
    emg_data = pd.read_pickle(f"/Users/federicobuccellato/Desktop/aml23-ego/EMG_data/{filename}")
    return emg_data

def data_loader(emg_ann):
    final_df = pd.DataFrame()

    for file in emg_ann['file'].unique():
        df_curr_file = emg_ann.query(f"file == '{file}'")
        indexes = df_curr_file['index']
        data_byKey = load_data(file).loc[indexes]
        data_byKey['file'] = file
        final_df = pd.concat([final_df, data_byKey], ignore_index=True)

    return final_df

def lowpass_filter(signal_data, cutoff_frequency, sampling_rate, filter_order=5):
    nyquist_rate = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_rate
    filter_coefficients_b, filter_coefficients_a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(filter_coefficients_b, filter_coefficients_a, signal_data.T).T
    return filtered_signal

def filter_and_normalize(signal_data, timestamps, cutoff_frequency, filter_order=5):
    sampling_rate = (timestamps.size - 1) / (timestamps[-1] - timestamps[0])
    
    nyquist_rate = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_rate
    filter_coefficients_b, filter_coefficients_a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(filter_coefficients_b, filter_coefficients_a, signal_data.T).T
    
    min_val = np.min(filtered_signal)
    max_val = np.max(filtered_signal)
    
    normalized_signal = 2 * (filtered_signal - min_val) / (max_val - min_val) - 1
    
    return normalized_signal, timestamps

def process_emg_data(emg_data):
    for key in ['myo_right', 'myo_left']:
        for i, row in emg_data.iterrows():
            signal_data = abs(row[key + '_readings'])
            timestamps = row[key + '_timestamps']
            
            normalized_signal, timestamps = filter_and_normalize(signal_data, timestamps, cutoff_frequency=5)
            
            emg_data.at[i, key + '_readings'] = normalized_signal
            emg_data.at[i, key + '_timestamps'] = timestamps

def resample_emg_data(emg_data, target_sampling_rate):
    for key in ['myo_right', 'myo_left']:
        for i, row in emg_data.iterrows():
            signal_data = row[key + '_readings']
            timestamps = row[key + '_timestamps']
            
            resampled_timestamps = np.linspace(timestamps[0], timestamps[-1],
                                               num=int(round(1 + target_sampling_rate * (timestamps[-1] - timestamps[0]))),
                                               endpoint=True)
            
            interpolate_function = interpolate.interp1d(timestamps, signal_data, axis=0, kind='linear', fill_value='extrapolate')
            
            resampled_signal = interpolate_function(resampled_timestamps)
            
            emg_data.at[i, key + '_readings'] = resampled_signal
            emg_data.at[i, key + '_timestamps'] = resampled_timestamps

def augment_emg_data(emg_data, segment_duration_s, resampled_Fs, num_segments_per_subject=10):
    augmented_data = []
    
    for i, row in emg_data.iterrows():
        timestamps_left = row['myo_left_timestamps']
        readings_left = row['myo_left_readings']
        timestamps_right = row['myo_right_timestamps']
        readings_right = row['myo_right_readings']
        
        num_readings_per_segment = int(segment_duration_s * resampled_Fs)
        
        if len(timestamps_left) < num_readings_per_segment:
            num_subactions = 1
        else:
            num_subactions = num_segments_per_subject
        
        if len(timestamps_left) < num_readings_per_segment:
            padded_readings_left = np.pad(readings_left, ((0, num_readings_per_segment - len(timestamps_left)), (0, 0)), mode='constant')
            padded_readings_right = np.pad(readings_right, ((0, num_readings_per_segment - len(timestamps_right)), (0, 0)), mode='constant')
            combined_readings = np.concatenate((padded_readings_left, padded_readings_right), axis=1)
            
            new_row = {
                'description': row['description'],
                'file': row['file'],
                'myo_combined_readings': combined_readings,
                'start': timestamps_left[0],
                'stop': timestamps_left[-1]
            }
            augmented_data.append(new_row)
        else:
            segment_start_times_s = np.linspace(timestamps_left[0], timestamps_left[-1] - segment_duration_s, num=num_subactions)
            
            for start_time in segment_start_times_s:
                start_index = np.where(timestamps_left >= start_time)[0][0]
                end_index = start_index + num_readings_per_segment
                
                if end_index > len(timestamps_left):
                    end_index = len(timestamps_left)
                
                left_segment_readings = readings_left[start_index:end_index]
                if len(left_segment_readings) < num_readings_per_segment:
                    left_segment_readings = np.pad(left_segment_readings, ((0, num_readings_per_segment - len(left_segment_readings)), (0, 0)), mode='constant')
                
                right_segment_readings = readings_right[start_index:end_index]
                if len(right_segment_readings) < num_readings_per_segment:
                    right_segment_readings = np.pad(right_segment_readings, ((0, num_readings_per_segment - len(right_segment_readings)), (0, 0)), mode='constant')
                
                combined_segment_readings = np.concatenate((left_segment_readings, right_segment_readings), axis=1)
                
                new_row = {
                    'description': row['description'],
                    'file': row['file'],
                    'myo_combined_readings': combined_segment_readings,
                    'start': timestamps_left[start_index],
                    'stop': timestamps_left[end_index-1],
                }
                
                augmented_data.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_data).reset_index(drop=True)
    return augmented_df

if __name__=='__main__':

    save_dir = '/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed'

    emg_annotations_train = pd.read_pickle(f'/Users/federicobuccellato/Desktop/aml23-ego/action-net/ActionNet_train.pkl')
    emg_annotations_test = pd.read_pickle(f'/Users/federicobuccellato/Desktop/aml23-ego/action-net/ActionNet_test.pkl')

    emg_train = data_loader(emg_annotations_train)
    emg_test = data_loader(emg_annotations_test)

    process_emg_data(emg_train)
    process_emg_data(emg_test)

    resample_emg_data(emg_train, target_sampling_rate=10)
    resample_emg_data(emg_test, target_sampling_rate=10)

    emg_train = augment_emg_data(emg_train, segment_duration_s=10, resampled_Fs=10, num_segments_per_subject=20)
    emg_test = augment_emg_data(emg_test, segment_duration_s=10, resampled_Fs=10, num_segments_per_subject=20)

    emg_train['description'] = emg_train['description'].map(lambda x: 'Get/replace items from refrigerator/cabinets/drawers' if x == 'Get items from refrigerator/cabinets/drawers' else x)
    emg_test['description'] = emg_test['description'].map(lambda x: 'Get/replace items from refrigerator/cabinets/drawers' if x == 'Get items from refrigerator/cabinets/drawers' else x)
    emg_train['description'] = emg_train['description'].map(lambda x: 'Open/close a jar of almond butter' if x == 'Open a jar of almond butter' else x)
    emg_test['description'] = emg_test['description'].map(lambda x: 'Open/close a jar of almond butter' if x == 'Open a jar of almond butter' else x)
    
    emg_train['labels'] = emg_train['description'].map(activity_labels).astype(int)
    emg_test['labels'] = emg_test['description'].map(activity_labels).astype(int) 

    emg_train['uid']=emg_train.index
    emg_test['uid']=emg_test.index

    emg_train.to_pickle(f'{save_dir}/ActionNet_EMG_train_raw.pkl')
    emg_test.to_pickle(f'{save_dir}/ActionNet_EMG_test_raw.pkl')



