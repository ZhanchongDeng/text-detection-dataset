BUILD_DIR=build

generate:
	. venv/bin/activate && \
	python3 generate.py --build-dir $(BUILD_DIR) --silent

visualize:
	. venv/bin/activate && \
	python3 visualize.py --build-dir $(BUILD_DIR) --verbose --num-images 10 --dataset UberText

report:
	. venv/bin/activate && \
	python3 report.py --build-dir $(BUILD_DIR)

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