import argparse
import logging
from pathlib import Path
import json
import xml.etree.ElementTree as ET

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
    parser.add_argument("--silent", action="store_true", help="No logging")
    args = parser.parse_args()

    if not args.silent:
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    if args.generate:
        create_label_json(constants.IC13)
        create_label_json(constants.IC15)
        create_label_json(constants.COCO_TEXT)
        create_label_json(constants.MSRA_TD500)
        create_label_json(constants.SVT)

    if args.inspect:
        inspect_dataset(constants.IC13, args.num_images)
        inspect_dataset(constants.IC15, args.num_images)
        inspect_dataset(constants.COCO_TEXT, args.num_images)
        inspect_dataset(constants.MSRA_TD500, args.num_images)
        inspect_dataset(constants.SVT, args.num_images)

def create_label_json(dataset:str):
    '''Create label json for dataset and save it to dataset.json.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
    '''
    dataset_dir = Path(constants.RAW_DIR) / dataset
    label_json_fp = Path(constants.JSON_DIR) / f"{dataset}.json"
    num_text_instances = 0

    # initialize empty json
    label_json = []
    
    match(dataset):
        case constants.IC13:
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                set_name = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{set_name}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")\
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)

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
                        logging.error("Error with file %s", gt_fp)

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
            
        case constants.IC15:
            for gt_fp in (dataset_dir).glob("*/*.txt"):
                set_name = gt_fp.parent.name.split("_")[0]
                image_path = dataset_dir / f"{set_name}_images" / (gt_fp.stem.replace("gt_", "") + ".jpg")
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)

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
                        logging.error("Error with file %s", gt_fp)
                    
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

            logging.debug("Number of images with no annotations: %d", num_no_annotations)
            logging.debug("Number of non-english annotations: %d", num_non_english)
            logging.debug("Number of images without english annotations: %d", num_all_non_english)
            logging.debug("Number of illegible annotations: %d", num_illegible)

        case constants.MSRA_TD500:
            for gt_fp in (dataset_dir).glob("*/*.gt"):
                set_name = gt_fp.parent.name
                image_path = gt_fp.with_suffix(".JPG")
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)
                # each line separated by " "
                # index, difficulty level, x, y, width, height, angle
                # no text, only for detection, do not include text as key
                with gt_fp.open("r", encoding="utf-8-sig") as f:
                    gt = f.read()
                    lines = gt.split("\n")

                boxes = []
                # write in in for loop, more interpretable
                for line in lines:
                    if line == "":
                        continue

                    box = line.split(" ")
                    # manually calculate 4 corners given center_x, center_y, w, h, angle
                    x_tl, y_tl = int(box[2]), int(box[3])
                    w, h = int(box[4]), int(box[5])
                    theta = float(box[6])
                    # Define rotation from image frame to global frame
                    R = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ])
                    center = np.array([x_tl + w/2, y_tl + h/2])
                    # Define 4 corners in image frame
                    pt_tl = R @ np.array([-w/2, -h/2]) + center
                    pt_tr = R @ np.array([w/2, -h/2]) + center
                    pt_br = R @ np.array([w/2, h/2]) + center
                    pt_bl = R @ np.array([-w/2, h/2]) + center

                    corners = [
                        pt_tl.round().tolist(),
                        pt_tr.round().tolist(),
                        pt_br.round().tolist(),
                        pt_bl.round().tolist()
                    ]
                    
                    box_dict = {
                        "corners": corners,
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

        case constants.SVT:
            # test.xml and train.xml
            for xml_fp in (dataset_dir).glob("*.xml"):
                set_name = xml_fp.stem

                # each xml contains multiple images
                root = ET.parse(xml_fp).getroot()
                for image in root.findall("image"):
                    image_path = dataset_dir / image.find("imageName").text
                    # sanity check if image exists
                    if not image_path.exists():
                        logging.error("Image %s does not exist", image_path)

                    boxes = []
                    for box in image.findall("taggedRectangles/taggedRectangle"):
                        
                        # each box has 4 corners
                        x = int(box.attrib["x"])
                        y = int(box.attrib["y"])
                        w = int(box.attrib["width"])
                        h = int(box.attrib["height"])
                        corners = [
                            [x, y],
                            [x+w, y],
                            [x+w, y+h],
                            [x, y+h]
                        ]
                        text = box.find("tag").text
                        
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
                            "gt_path": str(xml_fp.absolute())
                        }
                    )

            
            
    logging.info(f"{dataset} has {len(label_json)} images")
    logging.info(f"{dataset} has {num_text_instances} text instances")
    
    if not label_json_fp.parent.exists():
        label_json_fp.parent.mkdir(parents=True)

    with label_json_fp.open("w") as f:
        json.dump(label_json, f, indent=4, sort_keys=True)


def inspect_dataset(dataset:str, num_images):
    '''Randomly select some images, visualize its text boxes and text.
    
    Args:
        dataset (str): dataset name, e.g. ic13, ic15
        num_images (int): number of images to visualize
    '''
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
            if "text" in box:
                cv2.putText(image, box["text"], (corners[0][0], corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(str(Path(image_path).name), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()