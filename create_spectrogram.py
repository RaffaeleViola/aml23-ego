import torch
import torchaudio.transforms as T
import pandas as pd

n_fft = 32
win_length = None
hop_length = 4

spectrogram_transform = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True
)

def convert_emg_to_spectrogram(emg_raw):
    results = {"features": []}
    print(emg_raw.keys())

    for idx in range(len(emg_raw)):
        signal = emg_raw['myo_combined_readings'][idx]
        
        signal_tensor = torch.from_numpy(signal).float()
        
        spectrograms = [spectrogram_transform(signal_tensor[:, i]) for i in range(signal_tensor.shape[1])]
        
        results['features'].append({
            'uid': emg_raw['uid'][idx],
            'features_spectrogram': spectrograms
        })
    
    return results

if __name__ == '__main__':
    emg_raw = pd.read_pickle("/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/ActionNet_EMG_test_raw.pkl")
    miao = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/EMG_spectrograms.pkl')
    maio = pd.read_pickle('/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/EMG_ActionNet_test.pkl')

    '''# Converte i dati EMG in spettrogrammi
    spectrograms_dict = convert_emg_to_spectrogram(emg_raw)

    output_file = "/Users/federicobuccellato/Desktop/aml23-ego/EMG_spectrograms_test.pkl"
    with open(output_file, 'wb') as f_pickle:
        pickle.dump(spectrograms_dict, f_pickle)'''
    


