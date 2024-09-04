import pandas as pd

# Percorsi ai file a, b, c
rgb_data = pd.read_pickle("/Users/federicobuccellato/Desktop/aml23-ego/RGB_preprocessed/S04_10_dense_test.pkl")
emg_data = pd.read_pickle("/Users/federicobuccellato/Desktop/aml23-ego/EMG_preprocessed/EMG_S04_test.pkl")

# Verifica che le lunghezze siano uguali
assert len(rgb_data['features']) == len(emg_data['features']), "I file a e b non hanno lo stesso numero di UID!"

# Creazione della nuova lista di feature
new_features = []

# Per ogni UID, combina i dati di RGB e EMG
for i in range(len(rgb_data['features'])):
    uid = rgb_data['features'][i]['uid']
    features_RGB = rgb_data['features'][i]['features_RGB']
    features_EMG = emg_data['features'][i]['features_EMG']
    
    # Crea un dizionario per ciascuna riga
    new_row = {
        'uid': uid,
        'features_RGB': features_RGB,
        'features_EMG': features_EMG
    }
    
    new_features.append(new_row)

# Creare il nuovo dataframe
new_data = {
    'features': new_features
}

# Salvare il nuovo file .pkl
new_file_path = "/Users/federicobuccellato/Desktop/aml23-ego/Fusion_preprocessed/midlevel_feat_10_dense_S04_test.pkl"
with open(new_file_path, 'wb') as f:
    pd.to_pickle(new_data, f)

print(f"Il nuovo file è stato salvato correttamente in {new_file_path}")



