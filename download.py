import argparse
from pathlib import Path
import json
import wget
import zipfile
import logging

import constants

def download_all():
    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--build-dir", type=str, default="build", help="Path to build directory")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    config_fp = Path(constants.CONFIG_DIR) / args.config
    if not config_fp.exists():
        # TODO: create example config file
        # Warn user to fill out required fields
        pass
    
    with config_fp.open("r") as f:
        config = json.load(f)
    
    for dataset in config["datasets"]:
        # ICDAR datasets require registration, need mauall download
        if "ICDAR" in dataset["name"]:
            pass
        # COCO Dataset is too large to download, better specify path to dataset
        elif dataset["name"] == "COCO-Text":
            pass
        elif dataset["url"] is not None:
            download_dataset(args.build_dir, dataset)
        else:
            logging.exception(f"Dataset {dataset['name']} does not have a download url and not handled")
    

def download_dataset(build_dir, dataset):
    dataset_name = dataset["name"]
    dataset_fp = Path(dataset["path"])
    dataset_fp.mkdir(parents=True, exist_ok=True)
    # Download using wget
    zip_tmp_fp = Path(build_dir) / "tmp" / f"{dataset_name}.zip"
    print(zip_tmp_fp)
    if not zip_tmp_fp.exists():
        zip_tmp_fp.parent.mkdir(parents=True, exist_ok=True)
        wget.download(dataset["url"], out=str(zip_tmp_fp))
    # Unzip
    with zipfile.ZipFile(zip_tmp_fp, "r") as zip_fp:
        zip_fp.extractall(dataset_fp.parent)
    # rename
    dataset_fp.rename(dataset_fp.parent / dataset_name)


if __name__ == "__main__":
    download_all()