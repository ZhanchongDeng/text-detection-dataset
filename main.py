import logging
from pathlib import Path
import json

from PIL import Image, ImageDraw
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
    #                 "x0": int,
    #                 "y0": int,
    #                 "x1": int,
    #                 "y1": int,
    #                 "text": str
    #             }
    #         ]
    # ]
    create_label_json(constants.IC13_DIR, "gt_")
    inspect_dataset("ic13", 5)
    # create_label_json(constants.IC15_DIR, "gt_")
    return

def create_label_json(dataset:str, label_suffix:str):
    '''Create label json for IC dataset and save it to dataset.json.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = dataset_dir / "dataset.json"
    # initialize empty json
    label_json = []
    for gt_fp in (dataset_dir / "labels").iterdir():
        image_path = dataset_dir / 'images' / (gt_fp.stem.replace(label_suffix, "") + ".jpg")
        with gt_fp.open("r") as f:
            gt = f.read()
            lines = gt.split("\n")
        # write in in for loop, more interpretable
        boxes = []
        for line in lines:
            if line == "":
                continue

            box = line.split(" ")
            box_dict = {
                "x0": int(box[0]),
                "y0": int(box[1]),
                "x1": int(box[2]),
                "y1": int(box[3]),
                "text": box[4][1:-1] # remove "" around text
            }
            boxes.append(box_dict)

        label_json.append({"image_path": str(image_path), "boxes": boxes})

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
        boxes = label_json[i]["boxes"]
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle([(box["x0"], box["y0"]), (box["x1"], box["y1"])], outline="red")
            draw.text((box["x0"], box["y0"]), box["text"], fill="red")
        image.show()

if __name__ == "__main__":
    main()