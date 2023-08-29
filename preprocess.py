import argparse
import json
from pathlib import Path

import cv2
import logging
import numpy as np

import constants

def main():
    parser = argparse.ArgumentParser("Resize images and bounding polygon to 640 x 640")
    parser.add_argument('--data-dir', type=str, required=True, help="Directory to save the new data")
    parser.add_argument('--build-dir', type=str, default='build', help="Directory to save created files")
    parser.add_argument("--dataset", type=str, default=None, nargs="*", help="List of dataset names to generate")

    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = ['ArT', 'COCO_Text', 'ICDAR2013', 'ICDAR2015', 'MSRA-TD500', 'SVT', 'TextOCR', 'UberText']
    
    for dataset in args.dataset:
        preprocess_dataset(args.data_dir, args.build_dir, dataset)

def preprocess_dataset(data_dir, build_dir, dataset):
    to_width = 640
    to_height = 640

    json_path = Path(build_dir) / constants.JSON_DIR / f'{dataset}.json'
    with json_path.open() as f:
        json_data = json.load(f)

    # new json path
    new_json_path = Path(data_dir) / constants.JSON_DIR / f'{json_path.stem}.json'

    idx = 0
    for entry in json_data:
        old_path = Path(entry['image_path'])
        image = cv2.imread(str(old_path))
        from_height, from_width = image.shape[:2]

        image_scaled = cv2.resize(image, (to_width, to_height))
        # save image to new path
        new_path = Path(data_dir) / json_path.stem / f'{idx}.jpg'
        new_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(new_path), image_scaled)

        new_boxes = []
        for box in entry['boxes']:
            corners = np.array(box['corners'])
            corners_rescaled = (corners * np.array([to_width / from_width, to_height / from_height])).tolist()
            box['corners'] = corners_rescaled
            new_boxes.append(box)
        
        # modify json_data
        entry['image_path'] = str(new_path.absolute())
        entry['boxes'] = new_boxes

        idx += 1
    
    # save the new json
    new_json_path.parent.mkdir(parents=True, exist_ok=True)
    with new_json_path.open('w') as f:
        json.dump(json_data, f, indent=4)
    # # show imagge with polygon before rescaling
    # cv2.polylines(image, np.array(boxes), True, (0, 255, 0), 2)
    # cv2.imshow('before', image)
    # cv2.waitKey(0)
    # # draw polygon on image
    # cv2.polylines(image_scaled, np.array(boxes_rescaled), True, (0, 255, 0), 2)
    # cv2.imshow('after', image_scaled)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()