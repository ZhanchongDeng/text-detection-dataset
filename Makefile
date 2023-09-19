BUILD_DIR=build
CONFIG_FILE=config.json

generate:
	. venv/bin/activate && \
	python3 generate.py --build-dir $(BUILD_DIR) --verbose --config $(CONFIG_FILE) --dataset COCO_Text

visualize:
	. venv/bin/activate && \
	python3 visualize.py --build-dir $(BUILD_DIR) --verbose --num-images 20 --config $(CONFIG_FILE) --dataset TextOCR

report:
	. venv/bin/activate && \
	python3 report.py --build-dir $(BUILD_DIR)

preprocess:
	. venv/bin/activate && \
	python3 preprocess.py --build-dir $(BUILD_DIR) --data-dir build/uber_crop --config $(CONFIG_FILE) --dataset UberText

download:
	. venv/bin/activate && \
	python3 download.py --verbose --build-dir $(BUILD_DIR)

configure:
	python3 -m venv venv
	. venv/bin/activate && \
	pip3 install -r requirements.txt

unconfigure:
	rm -rf venv

clean:
	rm -rf $(BUILD_DIR)