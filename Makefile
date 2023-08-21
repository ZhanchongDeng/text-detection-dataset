BUILD_DIR=build

generate:
	. venv/bin/activate && \
	python3 generate.py --verbose

inspect:
	. venv/bin/activate && \
	python3 visualize.py --verbose --num-images 1

download:
	. venv/bin/activate && \
	python3 download.py --verbose --build-dir $(BUILD_DIR)

configure:
	python3 -m venv venv
	. venv/bin/activate && \
	pip3 install -r requirements.txt

test:
	echo 'Please test'

unconfigure:
	rm -rf venv

clean:
	rm -rf json_data