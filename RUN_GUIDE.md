
# Epic Kitchens

## EK Data Extraction
```bash
python3 save_feat.py \
  config=configs/I3D_save_feat.yaml \
  dataset.shift=D1-D1 \
  save.num_frames_per_clip.RGB=5 \
  save.dense_sampling.RGB=True\
  split=train \
  dataset.RGB.data_path=./ek_data/frames 
```

## EK Training
```bash
python3 training.py name=training_RGB_EK \
  config=configs/default.yaml \
  train.num_frames_per_clip.RGB=25 \
  train.dense_sampling.RGB=False \
  models.RGB.model=MLP_Classifier
```

# Action NET

## AN RGB Data Extraction
```bash
python3 save_feat_AN.py \
  config=configs/feature_rgb_extraction.yaml \
  save.num_frames_per_clip.RGB=5 \
  save.dense_sampling.RGB=True \
  split=test
```

## AN EMG Data Extraction
Run the file feature_extraction_EMG.py and then refactoring_data_EMG.py to properly extract and set up the data.

## Training on EMG Data Only

```bash
python3 train_classifier_AN.py name=train_emg_an \
  config=configs/training_emg_an.yaml \
  dataset.EMG.features_name=EMG_preprocessed \
  models.EMG.model=EmgLSTMNet
```
 
## Training on RGB Data Only
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

## Training on Multimodal Data Late Fusion
```bash
python3 train_classifier_AN.py name=train_multimodal _an\
  config=configs/training_multimodal_an.yaml \
  models.RGB.model=MultiScaleTRN \
  models.EMG.model=EmgLSTMNet \
  dataset.name_dataset_rgb=S04_10_dense\
```

## Training on Multimodal Data Middle Fusion
```bash
python3 train_mid_classifier_AN.py  name=train_mid_multimodal_an\
  config=configs/training_mid_an.yaml \
```
## Training on Spectrogram Data
```bash
python3 train_classifier_AN.py name=train_spectrogram_an \
  config=configs/training_spectrogram_an.yaml \
  dataset.EMG.features_name=EMG_preprocessed \
  models.EMG.model=SqueezeNet \
```

