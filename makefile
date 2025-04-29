# Setup Commands
clean:
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .DS_Store */.DS_Store */*/.DS_Store

install:
	venv/bin/pip install -r requirements.txt

venv:
	python3.11 -m venv venv/
	venv/bin/pip install --upgrade pip setuptools wheel

dataset:
	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
	unzip DIV2K_train_LR_bicubic_X4.zip
	rm -rf DIV2K_train_LR_bicubic_X4.zip

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
	unzip DIV2K_valid_LR_bicubic_X4.zip
	rm -rf DIV2K_valid_LR_bicubic_X4.zip

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip
	unzip DIV2K_valid_LR_unknown_X4.zip
	rm -rf DIV2K_valid_LR_unknown_X4.zip
	mv DIV2K_valid_LR_unknown DIV2K_test_LR_unknown

# Data Analysis Commands
metrics:
	venv/bin/python metrics.py --model_path=./models/augmented/DenseSteganoGAN/1/epoch_32.pth --visualize

# Training Commands
train:
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=6

train-basic:
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=6

train-residual:
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=6

train-dense:
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=6

train-all:
	make train-basic
	make train-residual
	make train-dense

train-basic-extra:
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=32 --data_depth=12

train-residual-extra:
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=32 --data_depth=12

train-dense-extra:
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=32 --data_depth=12

train-all-extra:
	make train-basic-extra
	make train-residual-extra
	make train-dense-extra

train-all-both:
	make train-all
	make train-all-extra
