# Setup Commands
clean:
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .DS_Store */.DS_Store */*/.DS_Store

install:
	venv/bin/pip install -r requirements.txt

venv:
	python3.9 -m venv venv/
	venv/bin/pip install --upgrade pip setuptools wheel

dataset:
	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
	unzip DIV2K_train_LR_bicubic_X4.zip
	rm -rf DIV2K_train_LR_bicubic_X4.zip
	mv DIV2K_train_LR_bicubic data/DIV2K_train_LR_bicubic

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
	unzip DIV2K_valid_LR_bicubic_X4.zip
	rm -rf DIV2K_valid_LR_bicubic_X4.zip
	mv DIV2K_valid_LR_bicubic data/DIV2K_valid_LR_bicubic

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip
	unzip DIV2K_valid_LR_unknown_X4.zip
	rm -rf DIV2K_valid_LR_unknown_X4.zip
	mv DIV2K_valid_LR_unknown data/DIV2K_valid_LR_unknown

run:
	venv/bin/python run.py --model_type="dense" --data_depth=6 --model_path=models/DenseSteganoGAN/6/epoch_32.pth --image_path=data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"	

# Data Analysis Commands
metrics:
	venv/bin/python metrics.py --model_path=./models/archived/augmented/DenseSteganoGAN/1/epoch_32.pth --visualize

run:
	venv/bin/python run.py --model_type="dense" --data_depth=6 --model_path=models/DenseSteganoGAN/6/epoch_32.pth --image_path=DIV2K_valid_LR_bicubic/X4/0801x4.png --text="Hello World!"

steg-expose:
	java -jar StegExpose.jar save_and_div default default steganalysis


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

train-basic-leaky:
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=6

train-residual-leaky:
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=6

train-dense-leaky:
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m src.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=6

train-all-leaky:
	make train-basic-leaky
	make train-residual-leaky
	make train-dense-leaky

train-basic-long:
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m src.train --model=BasicSteganoGAN --epochs=64 --data_depth=6

train-residual-long:
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m src.train --model=ResidualSteganoGAN --epochs=64 --data_depth=6

train-dense-long:
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m src.train --model=DenseSteganoGAN --epochs=64 --data_depth=6

train-all-long:
	make train-basic-long
	make train-residual-long
	make train-dense-long


metrics-basic:
		venv/bin/python metrics.py --model_type=basic --data_depth=1 --model_path=models/basic/depth1_epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-residual:

metrics-dense:
		venv/bin/python metrics.py --model_type=dense --data_depth=3 --model_path=models/archived/augmented/DenseSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv


metrics-all:
	make metrics-basic
	make metrics-residual
	make metrics-dense

metrics-basic-extra:

metrics-residual-extra:

metrics-dense-extra:

metrics-all-extra:
	make metrics-basic-extra
	make metrics-residual-extra
	make metrics-dense-extra

metrics-all-both:
	make metrics-all
	make metrics-all-extra