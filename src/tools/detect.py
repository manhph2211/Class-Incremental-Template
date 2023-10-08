import sys
sys.path.append(".")
from src.models.detector.inference import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default='src/models/detector/saved_models/detector.pth', type=str)
    parser.add_argument(
        "--dataset", default='bird_dataset', type=str, help="name of the dataset to use",
    )
    parser.add_argument("--image_path", required=True, type=str, help="path to the test image")
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.1)
    parsed_args = parser.parse_args()

    main(parsed_args)