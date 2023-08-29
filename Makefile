BUILD_DIR=build

generate:
	. venv/bin/activate && \
	python3 generate.py --build-dir $(BUILD_DIR) --silent --config config.json

visualize:
	. venv/bin/activate && \
	python3 visualize.py --build-dir $(BUILD_DIR) --silent --num-images 10 --config config.json

report:
	. venv/bin/activate && \
	python3 report.py --build-dir $(BUILD_DIR)

preprocess:
	. venv/bin/activate && \
	python3 preprocess.py --build-dir $(BUILD_DIR) --data-dir build/tmp

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