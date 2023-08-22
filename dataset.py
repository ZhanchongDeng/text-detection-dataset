from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset

import constants

class CustomTextDetectionDataset(Dataset):
    def __init__(self, build_dir="build"):
        self.image_fps = []
        self.points = []

        json_dir = Path(build_dir) / constants.JSON_DIR

        for json_file in json_dir.iterdir():
            with json_file.open("r") as f:
                json_data = json.load(f)
            
            for image in json_data:
                self.image_fps.append(image["image_path"])
                boxes = []
                for box in image["boxes"]:
                    boxes.append(box["corners"])
                self.points.append(boxes)

    def __len__(self):
        return len(self.image_fps)

    def __getitem__(self, idx):
        image_fp = self.image_fps[idx]
        image = Image.open(image_fp).convert("RGB")
        points = self.points[idx]

        return image, points
    

if __name__ == "__main__":
    dataset = CustomTextDetectionDataset()
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][1])