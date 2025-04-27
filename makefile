# Setup Commands
clean:
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .DS_Store */.DS_Store */*/.DS_Store

install:
	venv/bin/pip install -r requirements.txt

venv:
	python3.11 -m venv venv/
	venv/bin/pip install --upgrade pip setuptools wheel

# Run Commands
dataset:
	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
	unzip DIV2K_train_LR_bicubic_X4.zip
	rm -rf DIV2K_train_LR_bicubic_X4.zip

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
	unzip DIV2K_valid_LR_bicubic_X4.zip
	rm -rf DIV2K_valid_LR_bicubic_X4.zip

train:
	venv/bin/python -m src.train
