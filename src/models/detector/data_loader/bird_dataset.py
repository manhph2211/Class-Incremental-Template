#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import json
from .dataset_base import DatasetBase, DatasetConfigBase


class BirdDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(BirdDatasetConfig, self).__init__()

        self.CLASSES = [
            "bird",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_bird_config = BirdDatasetConfig()


class BirdDataset(DatasetBase):
    __name__ = "bird_dataset"

    def __init__(
        self,
        data_path,
        classes=_bird_config.CLASSES,
        colors=_bird_config.COLORS,
        phase="test",
        transform=None,
        shuffle=True,
        random_seed=2000,
        normalize_bbox=False,
        bbox_transformer=None,
        multiscale=False,
        resize_after_batch_num=10,
    ):
        super(BirdDataset, self).__init__(
            data_path,
            classes=classes,
            colors=colors,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
            multiscale=multiscale,
            resize_after_batch_num=resize_after_batch_num,
        )

        assert os.path.isdir(data_path)
        self._transform = transform
        self._image_paths = sorted(
            [os.path.join(data_path, image_path) for image_path in os.listdir(data_path)]
        )
