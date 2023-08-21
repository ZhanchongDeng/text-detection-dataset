import argparse
import logging
from pathlib import Path
import json
import cv2
import numpy as np

import constants

def visualize_all():
    # Converting all text detection dataset to COCO format (simplified)
    # [
    #     {
    #         "image_path": str,
    #         "boxes": [
    #             {
    #                 corners: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
    #                 "text": str
    #             }
    #         ]
    # ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to inspect")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--silent", action="store_true", help="No logging")
    args = parser.parse_args()

    if not args.silent:
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    with open(args.config, "r") as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        inspect_dataset(dataset_config["name"], args.num_images)


def inspect_dataset(dataset_name:str, num_images):
    '''Randomly select some images, visualize its text boxes and text.
    
    Args:
        dataset_name (str): name of the dataset
        num_images (int): number of images to visualize
    '''
    label_json_fp = Path(constants.JSON_DIR) / f"{dataset_name}.json"
    with label_json_fp.open("r") as f:  
        label_json = json.load(f)

    num_total_images = len(label_json)
    random_indices = np.random.choice(num_total_images, num_images, replace=False)
    for i in random_indices:
        image_path = label_json[i]["image_path"]
        logging.info("gt_path: %s", label_json[i]["gt_path"])
        boxes = label_json[i]["boxes"]
        image = cv2.imread(image_path)
        for box in boxes:
            corners = np.array(box["corners"], dtype=np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)
            if "text" in box:
                cv2.putText(image, box["text"], (corners[0][0], corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(str(Path(image_path).name), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_all()