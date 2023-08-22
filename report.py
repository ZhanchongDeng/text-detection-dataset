import argparse
import json
from pathlib import Path

import pandas as pd

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

        for image in dataset_json:
            num_text_instances += len(image["boxes"])
            for box in image["boxes"]:
                # MSRA_TD500 has no recognition, ignore illegible counts
                if dataset_name != constants.MSRA_TD500 and box['text'] == "###":
                    num_illegible += 1
        
        percent_legible = (num_text_instances - num_illegible) / num_text_instances
        avg_text_per_image = num_text_instances / num_images

        dataset_summary.append({
            "dataset_name": dataset_name,
            "num_images": num_images,
            "num_text_instances": num_text_instances,
            "num_illegible": num_illegible,
            "percent_legible": percent_legible,
            "avg_text_per_image": avg_text_per_image
        })

    df_report = pd.DataFrame(dataset_summary).sort_values(by="num_images")
    df_report.to_csv(report_fp, index=False)
    # write produce markdown
    with report_md.open("w") as f:
        f.write(df_report.to_markdown(index=False))


if __name__ == "__main__":
    generate_report()