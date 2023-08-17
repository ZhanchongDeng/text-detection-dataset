BUILD_DIR=build

generate:
	. venv/bin/activate && \
	python3 main.py --generate --verbose

inspect:
	. venv/bin/activate && \
	python3 main.py --inspect --verbose --num-images 5

configure:
	python3 -m venv venv
	. venv/bin/activate && \
	pip3 install -r requirements.txt

test:
	@echo 'Please test'

unconfigure:
	rm -rf venv

clean:
	rm -rf $(BUILD_DIR)