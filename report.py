import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import constants

def generate_report():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=str, default="build", help="Path to build directory")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    json_dir = build_dir / constants.JSON_DIR
    report_fp = build_dir / "report.csv"
    report_md = build_dir / "report.md"

    dataset_summary = []

    for dataset_json_fp in json_dir.iterdir():
        with dataset_json_fp.open("r") as f:
            dataset_json = json.load(f)
        
        dataset_name = dataset_json_fp.stem
        num_images = len(dataset_json)
        num_text_instances = 0
        num_illegible = 0
        num_partial_illegible = 0 # only available for Uber-Text

        for image in dataset_json:
            num_text_instances += len(image["boxes"])
            for box in image["boxes"]:
                # MSRA_TD500 has no recognition, ignore illegible counts
                if dataset_name != constants.MSRA_TD500 and box['text'] == "###":
                    num_illegible += 1
                # Uber-Text has partial illegible
                if dataset_name == constants.UBERTEXT and ("###" in box['text']) and (box['text'] != "###"):
                    num_partial_illegible += 1
                
                # check box coordinates should never exceed image size
                # use pillow to find image size
                # img = Image.open(image['image_path'])
                image_size = np.array([640, 640])

                if (np.array(box['corners']).max(axis=0) > image_size).any():
                    print(f"image: {image['image_path']}")
                    print(f"box: {box}")

        dataset_summary.append({
            "dataset_name": dataset_name,
            "num_images": num_images,
            "num_text_instances": num_text_instances,
            "num_illegible": num_illegible,
            "num_partial_illegible": num_partial_illegible,
        })

    df_report = pd.DataFrame(dataset_summary).sort_values(by="num_images")
    # set dataset_name as index
    df_report.set_index("dataset_name", inplace=True)
    # sum each column except average
    df_report.loc["Total"] = df_report.sum(numeric_only=True)
    #percent_legible = (num_text_instances - num_illegible - num_partial_illegible) / num_text_instances
    #avg_text_per_image = num_text_instances / num_images
    df_report['percent_legible'] = (df_report['num_text_instances'] - df_report['num_illegible'] - df_report['num_partial_illegible']) / df_report['num_text_instances']
    df_report['avg_text_per_image'] = df_report['num_text_instances'] / df_report['num_images']
    # write produce csv
    df_report.to_csv(report_fp, index=True)
    # write produce markdown
    with report_md.open("w") as f:
        f.write(df_report.to_markdown(index=True))


if __name__ == "__main__":
    generate_report()