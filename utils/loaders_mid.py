import glob
import math
from abc import ABC
from random import randint

import numpy as np
import pandas as pd
from .action_net_record import ActionNetRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger


class MidDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):

        self.modalities = modalities  # considered modalities [RGB, EMG])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.annotation_list = [ActionNetRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            # load features for each modality

            sampling_mode = "dense" if self.dense_sampling['RGB'] else "uniform"
            self.model_features = pd.DataFrame(pd.read_pickle(os.path.join(str(self.dataset_conf['MID_FUSION'].data_path),
                                                                          self.dataset_conf['MID_FUSION'].features_name + "_" +
                                                                          str(self.num_frames_per_clip['RGB']) + "_" +
                                                                          sampling_mode + "_" +
                                                                          pickle_name))['features'])[
                    ["uid", "features_RGB", "features_EMG"]]

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def __getitem__(self, index):
        record = self.annotation_list[index]
        
        sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
        
        # Aggiungi una gestione degli errori
        if len(sample_row) != 1:
            raise ValueError(f"Expected exactly 1 row for UID {record.uid}, but got {len(sample_row)}")
        
        sample = {}
        sample['MID_FUSION'] = sample_row["features_RGB"].values[0], sample_row["features_EMG"].values[0]
        if self.additional_info:
            return sample, record.label, record.uid
        else:
            return sample, record.label

    def __len__(self):
        return len(self.annotation_list)
