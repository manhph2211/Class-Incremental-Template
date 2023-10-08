import os
import logging
from src.models.detector.data_loader.bird_dataset import BirdDataset


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

__all__ = ["dataset_params"]

dataset_params = {
    "bird_dataset": {
        "anchors": [
            [0.013333333333333334, 0.013666666666666667],
            [0.016923076923076923, 0.027976190476190477],
            [0.022203947368421052, 0.044827615904163357],
            [0.025833333333333333, 0.016710875331564987],
            [0.034375, 0.028125],
            [0.038752362948960305, 0.07455104993043463],
            [0.05092592592592592, 0.04683129325109843],
            [0.06254458977407848, 0.0764872521246459],
            [0.07689655172413794, 0.14613778705636743],
            [0.11500570776255709, 0.09082682291666666],
            [0.162109375, 0.18448023426061494],
            [0.26129166666666664, 0.3815],
        ],
        "anchor_masks": [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]],
        "num_classes": 1,
        "img_h": 608,
        "img_w": 608,
    },
}


class Config(object):
    DATASETS = {
        "bird_dataset": BirdDataset
    }

    DATASET_PARAMS = dataset_params

    def __init__(self):
        self.SAVED_MODEL_PATH = os.path.join("src/models/detector", "saved_models")

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def _config_logging(log_file, log_level=logging.DEBUG):
    import sys

    format_line = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    custom_formatter = CustomFormatter(format_line)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    stream_handler.setFormatter(custom_formatter)

    logging.basicConfig(handlers=[file_handler, stream_handler], level=log_level, format=format_line)


class CustomFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        import copy

        LOG_COLORS = {
            logging.INFO: "\x1b[33m",
            logging.DEBUG: "\x1b[36m",
            logging.WARNING: "\x1b[31m",
            logging.ERROR: "\x1b[31;1m",
            logging.CRITICAL: "\x1b[35m",
        }

        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname, color_begin=LOG_COLORS[new_record.levelno], color_end="\x1b[0m",
            )
        return super(CustomFormatter, self).format(new_record, *args, **kwargs)
