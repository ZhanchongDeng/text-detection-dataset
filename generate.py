import argparse
import logging
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np

import coco_text_api.coco_text as coco_text
import constants

def generate_all():
    # Converting all text detection dataset to COCO format (simplified)
    # [
    #     {
    #         "image_path": str,
    #         "boxes": [
    #             {
    #                 corners: [[x0, y0], ..., [xn, yn]],
    #                 "text": str
    #             }
    #         ]
    # ]
    parser = argparse.ArgumentParser("Generate label json for datasets in COCO format")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, nargs="*", help="List of dataset names to generate")
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

    if args.dataset is None:
        args.dataset = list(config["datasets"].keys())
    
    for dataset_name in args.dataset:
        if dataset_name not in config['datasets']:
            logging.error("Dataset name %s not in config", dataset_name)
            continue
        else:
            dataset_config = config["datasets"][dataset_name]
            generate_label_json(dataset_name, dataset_config)

def generate_label_json(dataset_name:str, dataset_config:dict):
    '''Create label json for dataset and save it to dataset.json.
    
    Args:
        dataset_config(dict): dictionary containing dataset path and other metadata
            should at least contain:
                path
    '''
    dataset_dir = Path(dataset_config['path'])
    label_json_fp = Path(constants.JSON_DIR) / f"{dataset_name}.json"
    num_text_instances = 0

    # initialize empty json
    label_json = []
    
    match(dataset_name):
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
            ct = coco_text.COCO_Text(dataset_config['json_path'])
            images = ct.loadImgs(ct.getImgIds())
            coco_images_dir = dataset_dir

            num_no_text = 0
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
                    num_no_text += 1
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

            logging.debug("Number of images with no text: %d", num_no_text)
            logging.debug("Number of non-english text instances: %d", num_non_english)
            logging.debug("Number of images without english text instances: %d", num_all_non_english)
            logging.debug("Number of illegible text instances: %d", num_illegible)

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

        case constants.ART:
            dataset_dir = Path(dataset_config['path'])
            with open(dataset_config['json_path'], "r") as f:
                train_json = json.load(f)

            num_non_english = 0
            num_all_non_english = 0
            
            for img_id in train_json:
                image_path = dataset_dir / (img_id + ".jpg")
                # sanity check if image exists
                if not image_path.exists():
                    logging.error("Image %s does not exist", image_path)
                
                boxes = []
                for box in train_json[img_id]:
                    # Skip Chinese
                    if box["language"] == "Chinese":
                        num_non_english += 1
                        continue
                    # each box has n points
                    corners = box["points"]
                    text = box["transcription"]
                    box_dict = {
                        "corners": corners,
                        "text": text,
                    }
                    boxes.append(box_dict)

                # Skip if no english text
                if len(boxes) == 0:
                    num_all_non_english += 1
                    continue
                
                num_text_instances += len(boxes)
                label_json.append(
                    {
                        "image_path": str(image_path.absolute()),
                        "boxes": boxes,
                        "set": "train",
                        "gt_path": str(dataset_dir / "ArT.json")
                    }
                )
            
            logging.debug("Number of images without english text instances: %d", num_all_non_english)
            logging.debug("Number of non-english text instances: %d", num_non_english)

            
    logging.info(f"{dataset_name} has {len(label_json)} images")
    logging.info(f"{dataset_name} has {num_text_instances} text instances")
    
    if not label_json_fp.parent.exists():
        label_json_fp.parent.mkdir(parents=True)

    with label_json_fp.open("w") as f:
        json.dump(label_json, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    generate_all()