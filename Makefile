BUILD_DIR=build

generate:
	. venv/bin/activate && \
	python3 generate.py --verbose

visualize:
	. venv/bin/activate && \
	python3 visualize.py --verbose --num-images 20 --dataset ArT

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
	rm -rf json_data