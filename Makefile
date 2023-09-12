BUILD_DIR=/workspace/bobby/build/
CONFIG_FILE=runpod.json

generate:
	. venv/bin/activate && \
	python3 generate.py --build-dir $(BUILD_DIR) --verbose --config $(CONFIG_FILE) --dataset COCO_Text

visualize:
	. venv/bin/activate && \
	python3 visualize.py --build-dir $(BUILD_DIR) --silent --num-images 10 --config $(CONFIG_FILE) --dataset MSRA-TD500

report:
	. venv/bin/activate && \
	python3 report.py --build-dir $(BUILD_DIR)

preprocess:
	. venv/bin/activate && \
	python3 preprocess.py --build-dir $(BUILD_DIR) --data-dir /workspace/bobby/resized_wh --config $(CONFIG_FILE) --dataset ICDAR2013 ICDAR2015 ArT SVT MSRA-TD500

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