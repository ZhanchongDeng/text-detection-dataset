import argparse
import json
from pathlib import Path
import copy

import cv2
import logging
import numpy as np

import constants

def main():
    parser = argparse.ArgumentParser("Resize images and bounding polygon to 640 x 640")
    parser.add_argument('--data-dir', type=str, required=True, help="Directory to save the new data")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument('--build-dir', type=str, default='build', help="Directory to save created files")
    parser.add_argument("--dataset", type=str, default=None, nargs="*", help="List of dataset names to generate")
    parser.add_argument("--train-size", type=float, default=0.8, help="Percentage of data to use for training")
    parser.add_argument("--fixed-crop", action="store_true", default =False, help="Crop to 640 x 640")
    parser.add_argument("--visualize", action="store_true", default = False, help="Visualize the crop box")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")

    args = parser.parse_args()

    config_path = Path(constants.CONFIG_DIR) / args.config
    with open(config_path, "r") as f:
        config = json.load(f)

    if args.dataset is None:
        args.dataset = list(config["datasets"].keys())
    
    for dataset in args.dataset:
        preprocess_dataset(args.data_dir, args.build_dir, dataset, args.train_size, args.seed, args.fixed_crop, args.visualize)

def preprocess_dataset(data_dir, build_dir, dataset, train_size, seed = 0, fixed_crop = False, visualize = False):
    to_width = 640
    to_height = 640
    min_box_edge = 15

    json_path = Path(build_dir) / constants.JSON_DIR / f'{dataset}.json'
    with json_path.open() as f:
        json_data = json.load(f)

    # new json path
    new_json_path = Path(data_dir) / constants.JSON_DIR / f'{json_path.stem}.json'

    # randomly select 80% of the data for training
    np.random.seed(seed)
    np.random.shuffle(json_data)
    train_size = int(len(json_data) * train_size)

    new_json_data = []
    missed_boxes = 0
    img_idx = 0
    for entry_idx, entry in enumerate(json_data):
        old_path = Path(entry['image_path'])
        image = cv2.imread(str(old_path))
        from_height, from_width = image.shape[:2]
        crop_box_idx = 0

        # image smaller than 640 x 640, skip
        if from_height < to_height or from_width < to_width:
            continue
        
        '''
        Sliding Window crop
        '''
        # if 640crop
        # if fixed_crop:
        #     max_crop_size = np.array([to_width, to_height])
        #     crop_step_size = (max_crop_size / 2).astype(int)
        # else:
        #     # automatically crop & resize
        #     # image larger than 640 x 640, check each box
        #     ## if box is smaller than 15 x 15 in original resolution, use the entire image because we don't upsample
        #     ## if the box is larger than 15 x 15, use the largest cropping size while maintaining the bounding box to be larger than 15 x 15
        #     box_sizes = []
        #     for box in entry['boxes']:
        #         corners = np.array(box['corners'])
        #         box_size = np.max(corners, axis=0) - np.min(corners, axis=0)
        #         box_sizes.append(box_size)
        #     min_box_size = np.min(box_sizes, axis=0)
        #     # assume the model will resize the image into to_width x to_height
        #     # find the largest cropping size which will maintain the bounding box to be larger than min_box_edge x min_box_edge
        #     max_crop_size = (np.array([to_width, to_height]) / min_box_edge * min_box_size)
        #     # limit max_crop_size
        #     max_crop_size = np.where(max_crop_size > np.array([from_width, from_height]), np.array([from_width, from_height]), max_crop_size)
        #     crop_step_size = (max_crop_size / 2).astype(int)
        #     max_crop_size = max_crop_size.astype(int) 
        
        # # crop and append
        # # crop images with step size of crop_step_size, box size of max_crop_size
        # box_id = set()
        # for i in range(0, from_height, crop_step_size[1]):
        #     for j in range(0, from_width, crop_step_size[0]):
        #         if i + max_crop_size[1] > from_height:
        #             # offset from edge, maintain box size
        #             i = from_height - max_crop_size[1]
        #         if j + max_crop_size[0] > from_width:
        #             # offset from edge, maintain box size
        #             j = from_width - max_crop_size[0]
        #         # crop image
        #         image_cropped = image[i:i+max_crop_size[1], j:j+max_crop_size[0]]

        #         new_boxes = []
        #         for k, box in enumerate(entry['boxes']):
        #             corners = np.array(box['corners'])
        #             # limit x and y to be within the image
        #             corners = np.where(corners > np.array([from_width, from_height]), np.array([from_width, from_height]), corners)
        #             corners = np.where(corners < np.array([0, 0]), np.array([0, 0]), corners)
        #             # if any corners are outside the cropped image, skip
        #             if np.any(corners < np.array([j, i])) or np.any(corners > np.array([j+max_crop_size[0], i+max_crop_size[1]])):
        #                 continue
        #             # shift corners
        #             corners_shifted = corners - np.array([j, i])
        #             new_box = copy.deepcopy(box)
        #             new_box['corners'] = corners_shifted.tolist()
        #             box_id.add(k)
        #             new_boxes.append(new_box)

        #         # if no boxes are in the cropped image, skip
        #         if len(new_boxes) == 0:
        #             continue

        #         # save image to new path
        #         set_name = 'train' if img_idx < train_size else 'test'
        #         new_path = Path(data_dir) / set_name / f'{json_path.stem}_{img_idx}_{crop_box_idx}.jpg'
        #         new_path.parent.mkdir(parents=True, exist_ok=True)
        #         cv2.imwrite(str(new_path), image_cropped)
                
        #         # copy and append json_data
        #         new_entry = copy.deepcopy(entry)
        #         new_entry['old_image_path'] = str(old_path.absolute())
        #         new_entry['image_path'] = str(new_path.absolute())
        #         new_entry['boxes'] = new_boxes
        #         new_entry['set'] = set_name
        #         new_json_data.append(new_entry)
        #         crop_box_idx += 1
         
        # check if we miss any box
        # if len(box_id) != len(entry['boxes']):
        #     missed_boxes += len(entry['boxes']) - len(box_id)
        #     if visualize:
        #         # visualize crop box, all bounding box, and missing bounding box
        #         image = cv2.imread(str(old_path))
        #         for i, box in enumerate(entry['boxes']):
        #             if i in box_id:
        #                 color = (0, 255, 0)
        #             else:
        #                 color = (0, 0, 255)
        #             corners = np.array(box["corners"], dtype=np.int32)
        #             cv2.polylines(image, [corners], True, color, 2)

        #         # crop box top left
        #         for i in range(0, from_height, crop_step_size[1]):
        #             for j in range(0, from_width, crop_step_size[0]):
        #                 if i + max_crop_size[1] > from_height:
        #                     # offset from edge, maintain box size
        #                     i = from_height - max_crop_size[1]
        #                 if j + max_crop_size[0] > from_width:
        #                     # offset from edge, maintain box size
        #                     j = from_width - max_crop_size[0]
        #                 cv2.rectangle(image, (j, i), (j+max_crop_size[0], i+max_crop_size[1]), (255, 0, 0), 2)

        #         # save images
        #         p = Path(data_dir) / 'missing_box' / f'{json_path.stem}_{img_idx}.jpg'
        #         p.parent.mkdir(parents=True, exist_ok=True)
        #         cv2.imwrite(str(p), image)
        
        '''
        BBO Crop
        '''
        # generate a queue of box_idx sorted by box_size
        box_sizes = []
        rectangle_boxes = []
        # convert corners into rectangle box
        rectangle_boxes = []
        for box in entry['boxes']:
            corners = np.array(box['corners'])
            # corners should not exceed image size or negative
            corners = np.where(corners > np.array([from_width, from_height]), np.array([from_width, from_height]), corners)
            corners = np.where(corners < np.array([0, 0]), np.array([0, 0]), corners)
            # collect box size and box stats for later use
            box_size = np.max(corners, axis=0) - np.min(corners, axis=0)
            box_sizes.append(box_size)
            # append top left, top right, bottom right, bottom left to rectangle
            tl = np.min(corners, axis=0)
            br = np.max(corners, axis=0)
            rectangle_boxes.append([[tl[0], tl[1]], [br[0], tl[1]], [br[0], br[1]], [tl[0], br[1]]])

        box_sizes = np.array(box_sizes)
        box_size_id = np.argsort(box_sizes[:, 0] * box_sizes[:, 1])
        box_size_id = box_size_id[::-1].tolist()


        while len(box_size_id) > 0:
            cur_box_id = box_size_id.pop(0)
            # find max crop size
            cur_box_size = box_sizes[cur_box_id]
            cur_corners = rectangle_boxes[cur_box_id]
            max_crop_size = (np.array([to_width, to_height]) / min_box_edge * cur_box_size).astype(int)
            # randomly offset crop box to the top left of the box, limit to be within the image
            crop_box_top_left_min = cur_corners[0] - (max_crop_size - cur_box_size)
            crop_box_top_left_max = cur_corners[0]
            crop_box_top_left_min = np.where(crop_box_top_left_min < 0, -1, crop_box_top_left_min)
            crop_box_top_left = np.random.randint(crop_box_top_left_min, crop_box_top_left_max)
            # limit crop_box_top_left to be above 0
            crop_box_top_left = np.where(crop_box_top_left < np.array([0, 0]), np.array([0, 0]), crop_box_top_left)
            # create the image
            image_cropped = image[crop_box_top_left[1]:crop_box_top_left[1]+max_crop_size[1], crop_box_top_left[0]:crop_box_top_left[0]+max_crop_size[0]]
            new_boxes = []
            # add ids of boxes that are in the crop, remove from queue
            for i, box in enumerate(rectangle_boxes):
                in_crop = np.all(box[0] >= crop_box_top_left) and np.all(box[1] <= crop_box_top_left + max_crop_size)
                large_enough = np.all((box_sizes[i] / max_crop_size) >= (min_box_edge / np.array([to_width, to_height])))
                if in_crop and large_enough:
                    if i in box_size_id:
                        box_size_id.remove(i)
                    new_corners = box - crop_box_top_left
                    new_box = copy.deepcopy(entry['boxes'][i])
                    new_box['corners'] = new_corners.tolist()
                    new_boxes.append(new_box)
                    
            if len(new_boxes) == 0:
                logging.warning("No box in image %s", old_path)
                # visualize bounding boxes
                image = cv2.imread(str(old_path))
                for box in entry['boxes']:
                    corners = np.array(box["corners"], dtype=np.int32)
                    cv2.polylines(image, [corners], True, (0, 255, 0), 2)
                cv2.imshow(str(Path(old_path).name), image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return

            # save image
            set_name = 'train' if img_idx < train_size else 'test'
            new_path = Path(data_dir) / set_name / f'{json_path.stem}_{img_idx}_{crop_box_idx}.jpg'
            new_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(new_path), image_cropped)
            # copy and append json_data
            new_entry = copy.deepcopy(entry)
            new_entry['old_image_path'] = str(old_path.absolute())
            new_entry['image_path'] = str(new_path.absolute())
            new_entry['boxes'] = new_boxes
            new_entry['set'] = set_name
            crop_box_idx += 1
            new_json_data.append(new_entry)

        img_idx += 1
        if img_idx > 500:
            break
        
        # husam's top left algorithm
        # find top left box
        # offset random amount of pixel from top left box, make crop box
        # repeat until all box are cropped, or no more box can be cropped

        # resize image
        # image_scaled = cv2.resize(image, (to_width, to_height))
        # # save image to new path
        # set_name = 'train' if idx < train_size else 'test'
        # new_path = Path(data_dir) / set_name / f'{json_path.stem}_{idx}.jpg'
        # new_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(new_path), image_scaled)

        # new_boxes = []
        # for box in entry['boxes']:
        #     corners = np.array(box['corners'])
        #     corners_rescaled = (corners * np.array([to_width / from_width, to_height / from_height]))
        #     # limit x and y to be within the image
        #     corners_rescaled = np.where(corners_rescaled > np.array([to_width, to_height]), np.array([to_width, to_height]), corners_rescaled)
        #     box['corners'] = corners_rescaled.tolist()
        #     new_boxes.append(box)
    
    # save the new json
    logging.warning("Missed %d boxes", missed_boxes)
    new_json_path.parent.mkdir(parents=True, exist_ok=True)
    with new_json_path.open('w') as f:
        json.dump(new_json_data, f, indent=4)

if __name__ == '__main__':
    main()