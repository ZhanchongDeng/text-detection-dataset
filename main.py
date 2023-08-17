import logging
from pathlib import Path
import json

from PIL import Image, ImageDraw
import cv2
import numpy as np

import coco_text_api.coco_text as coco_text
import constants

def main():
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
    logging.basicConfig(level=logging.INFO)
    # create_label_json(constants.IC13_DIR)
    inspect_dataset(constants.IC13_DIR, 20)
    # create_label_json(constants.IC15_DIR)
    # inspect_dataset(constants.IC15_DIR, 20)
    return

def create_label_json(dataset:str):
    '''Create label json for dataset and save it to dataset.json.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = dataset_dir / "dataset.json"
    
    match(dataset):
        case constants.IC13_DIR:
            # initialize empty json
            label_json = []
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                train_or_test = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{train_or_test}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")
                with gt_fp.open("r", encoding="utf-8-sig") as f:
                    gt = f.read()
                    lines = gt.split("\n")
                # write in in for loop, more interpretable
                boxes = []
                for line in lines:
                    if line == "":
                        continue
                    
                    if train_or_test == "test":
                        sep = ","
                    else:
                        sep = " "
                    box = line.split(sep)
                    # first 4 is corners, rest should be regrouped as text
                    coordinates = [int(n) for n in box[:4]]
                    text = sep.join(box[4:]).strip()[1:-1]
                    try:
                        int(box[0])
                    except:
                        logging.exception("Error with file %s", gt_fp)
                        exit(-1)

                    # Transform from horizontal bounding box (top left, bottom right) to 4 corners
                    x_tl, y_tl, x_br, y_br = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
                    # clockwise from top left
                    corners = [
                        [x_tl, y_tl],
                        [x_br, y_tl],
                        [x_br, y_br],
                        [x_tl, y_br]
                    ]
                    box_dict = {
                        "corners": corners,
                        "text": text,
                    }
                    boxes.append(box_dict)

                label_json.append({"image_path": str(image_path.absolute()), "boxes": boxes, "group": train_or_test, "gt_path": str(gt_fp.absolute())})
            
        case constants.IC15_DIR:
            # initialize empty json
            label_json = []
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                train_or_test = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{train_or_test}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")
                with gt_fp.open("r", encoding="utf-8-sig") as f:
                    gt = f.read()
                    lines = gt.split("\n")
                # write in in for loop, more interpretable
                boxes = []
                for line in lines:
                    if line == "":
                        continue

                    sep = ","
                    box = line.split(sep)
                    # first 8 is corners, rest should be regrouped as text
                    corners =[]
                    for i in range(4):
                        corners.append([int(box[2*i]), int(box[2*i+1])])
                    text = sep.join(box[8:]).strip()
                    try:
                        int(box[0])
                    except:
                        logging.exception("Error with file %s", gt_fp)
                        exit(-1)
                    
                    box_dict = {
                        "corners": corners,
                        "text": text,
                    }
                    boxes.append(box_dict)

                label_json.append({"image_path": str(image_path.absolute()), "boxes": boxes, "group": train_or_test, "gt_path": str(gt_fp.absolute())})
        
        case constants.COCO_TEXT:
            boxes = []

    with label_json_fp.open("w") as f:
        json.dump(label_json, f)


def inspect_dataset(dataset:str, num_images):
    '''Randomly select some images, visualize its text boxes and text.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
        num_images (int): number of images to visualize
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = dataset_dir / "dataset.json"
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
            cv2.putText(image, box["text"], (corners[0][0], corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()