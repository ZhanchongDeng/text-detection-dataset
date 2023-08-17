import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate label json for all datasets")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to inspect")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.generate:
        create_label_json(constants.IC13_DIR)
        create_label_json(constants.IC15_DIR)
        create_label_json(constants.COCO_TEXT)

    if args.inspect:
        inspect_dataset(constants.IC13_DIR, args.num_images)
        inspect_dataset(constants.IC15_DIR, args.num_images)
        inspect_dataset(constants.COCO_TEXT, args.num_images)

def create_label_json(dataset:str):
    '''Create label json for dataset and save it to dataset.json.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = Path(constants.JSON_DIR) / f"{dataset}.json"
    num_text_instances = 0
    
    match(dataset):
        case constants.IC13_DIR:
            # initialize empty json
            label_json = []
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                set_name = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{set_name}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")\
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)
                    exit(-1)

                with gt_fp.open("r", encoding="utf-8-sig") as f:
                    gt = f.read()
                    lines = gt.split("\n")
                # write in in for loop, more interpretable
                boxes = []
                for line in lines:
                    if line == "":
                        continue
                    
                    if set_name == "test":
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
                    
                num_text_instances += len(boxes)
                label_json.append(
                    {
                        "image_path": str(image_path.absolute()),
                        "boxes": boxes,
                        "set": set_name,
                        "gt_path": str(gt_fp.absolute())
                    }
                )
            
        case constants.IC15_DIR:
            # initialize empty json
            label_json = []
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                set_name = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{set_name}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)
                    exit(-1)

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
                    
                num_text_instances += len(boxes)
                label_json.append(
                    {
                        "image_path": str(image_path.absolute()),
                        "boxes": boxes,
                        "set": set_name,
                        "gt_path": str(gt_fp.absolute())
                    }
                )

        case constants.COCO_TEXT:
            # initialize empty json
            label_json = []
            ct = coco_text.COCO_Text(dataset_dir / "COCO_Text.json")
            images = ct.loadImgs(ct.getImgIds())
            coco_images_dir = dataset_dir / "images"

            num_no_annotations = 0
            num_non_english = 0
            num_illegible = 0
            num_all_non_english = 0
            
            for img in images:
                id = img["id"]
                image_path = coco_images_dir / img["file_name"]
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)
                    exit(-1)

                set_name = img["set"]
                annIds = ct.getAnnIds(imgIds=id)
                anns = ct.loadAnns(annIds)
                if len(anns) == 0:
                    num_no_annotations += 1
                    continue
                    
                boxes = []
                for ann in anns:
                    # skip if non english
                    if ann["language"] != "english":
                        num_non_english += 1
                        continue
                    # polygon is a list of 8 numbers
                    corners = []
                    for i in range(4):
                        corners.append(ann['polygon'][2*i:2*i+2])
                    # if illegible, put label as ###
                    if ann["legibility"] == "illegible":
                        text = "###"
                        num_illegible += 1
                    else:
                        text = ann["utf8_string"]
                    box_dict = {
                        "corners": corners,
                        "text": text,
                    }
                    boxes.append(box_dict)

                if len(boxes) == 0:
                    num_all_non_english += 1
                    continue
                
                num_text_instances += len(boxes)
                label_json.append(
                    {
                        "image_path": str(image_path.absolute()), 
                        "boxes": boxes, 
                        "set": set_name, 
                        "gt_path": str(dataset_dir / "COCO_Text.json")
                    }
                )

            logging.info("Number of images with no annotations: %d", num_no_annotations)
            logging.info("Number of non-english annotations: %d", num_non_english)
            logging.info("Number of images without english annotations: %d", num_all_non_english)
            logging.info("Number of illegible annotations: %d", num_illegible)
            
    logging.info(f"{dataset} has {len(label_json)} images")
    logging.info(f"{dataset} has {num_text_instances} text instances")
    
    with label_json_fp.open("w") as f:
        json.dump(label_json, f, indent=4, sort_keys=True)


def inspect_dataset(dataset:str, num_images):
    '''Randomly select some images, visualize its text boxes and text.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
        num_images (int): number of images to visualize
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = Path(constants.JSON_DIR) / f"{dataset}.json"
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
        cv2.imshow(str(Path(image_path).name), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()