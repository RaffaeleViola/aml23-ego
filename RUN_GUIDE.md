

# 1. Addestramento su Dati EMG

```bash
python3 train_classifier_AN.py name=train_emg_an \
  config=configs/training_emg_an.yaml \
  dataset.EMG.features_name=EMG_preprocessed \
  models.EMG.model=EmgLSTMNet
```
 
## 2. Addestramento su Dati RGB
```bash
python3 train_classifier_AN.py name=train_rgb_an\
  config=configs/training_rgb_an.yaml \
  models.RGB.model=LSTM_Classifier \
  dataset.name_dataset_rgb=S04_10_dense\
```
```bash
python3 train_classifier_AN.py name=train_rgb_an\
  config=configs/training_rgb_an.yaml \
  models.RGB.model=MultiScaleTRN \
  dataset.name_dataset_rgb=S04_10_dense\
```

## 2. Addestramento su Dati Multimodal
```bash
python3 train_classifier_AN.py name=train_multimodal _an\
  config=configs/training_multimodal_an.yaml \
  models.RGB.model=Lstm_classifier \
  models.EMG.model=EmgLSTMNet \
  dataset.name_dataset_rgb=S04_10_dense\
```
```bash
python3 train_classifier_AN.py name=train_multimodal _an\
  config=configs/training_multimodal_an.yaml \
  models.RGB.model=MultiScaleTRN \
  models.EMG.model=EmgLSTMNet \
  dataset.name_dataset_rgb=S04_10_dense\
```