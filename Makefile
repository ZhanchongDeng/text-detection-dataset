BUILD_DIR=build

configure:
	python3 -m venv venv
	. venv/bin/activate && \
	pip3 install -r requirements.txt

generate:
	. venv/bin/activate && \
	python3 main.py

test:
	@echo 'Please test'


unconfigure:
	rm -rf venv

clean:
	rm -rf $(BUILD_DIR)
