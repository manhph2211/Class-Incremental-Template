import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models.detector.config import Config
from src.models.detector.models.yolo_v3 import YoloNet
from src.models.detector.models.functions.detect import DetectHandler


def test_one_image(
    args, total_config, dataset_class, dataset_params,
):

    model_path = args.snapshot
    dataset = args.dataset
    dataset_params = total_config.DATASET_PARAMS[dataset]
    input_size = (dataset_params["img_h"], dataset_params["img_w"])

    dataset_instance = dataset_class(data_path='data/raw/public_test/')
    model = YoloNet(dataset_config=dataset_params)
    model.load_state_dict(torch.load(model_path, map_location='cpu')["state_dict"])
    model.eval()

    img = cv2.imread(args.image_path)
    orig_img = np.copy(img)

    ori_h, ori_w = img.shape[:2]
    h_ratio = ori_h / dataset_params["img_h"]
    w_ratio = ori_w / dataset_params["img_w"]

    img = cv2.resize(img, input_size)

    img = img / 255.0
    input_x = torch.tensor(img.transpose(2, 0, 1)[np.newaxis, :]).float()

    predictions = model(input_x)

    detect_handler = DetectHandler(
        num_classes=dataset_params["num_classes"],
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        h_ratio=h_ratio,
        w_ratio=w_ratio,
    )

    bboxes, scores, classes = detect_handler(predictions)

    result = dataset_class.visualize_one_image_util(
        orig_img, dataset_instance.classes, dataset_instance.colors, bboxes, classes,
    )

    return result


def main(args):
    total_config = Config()
    if not args.dataset or args.dataset not in total_config.DATASETS.keys():
        raise Exception("specify one of the datasets to use in {}".format(list(total_config.DATASETS.keys())))
    if not args.image_path or not os.path.isfile(args.image_path):
        raise Exception("invalid image path")

    dataset = args.dataset
    dataset_class = total_config.DATASETS[dataset]
    dataset_params = total_config.DATASET_PARAMS[dataset]

    result = test_one_image(args, total_config, dataset_class, dataset_params,)

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=str)
    parser.add_argument(
        "--dataset", type=str, required=True, help="name of the dataset to use",
    )
    parser.add_argument("--image_path", required=True, type=str, help="path to the test image")
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parsed_args = parser.parse_args()

    main(parsed_args)
