# Setup Commands
clean:
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .DS_Store */.DS_Store */*/.DS_Store

install:
	venv/bin/pip install -r requirements.txt

venv3.10:
	python3.10 -m venv venv/
	venv/bin/pip install --upgrade pip setuptools wheel
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
	venv/bin/python run.py --visualizer="message_decode_full" --model_type="dense" --data_depth=6 --model_path=models/normal/DenseSteganoGAN/6/epoch_32.pth --image_path=data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"	

# Data Analysis Commands
metrics:
	venv/bin/python metrics.py --model_path=./models/archived/augmented/DenseSteganoGAN/1/epoch_32.pth --visualize

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

.PHONY: setup-steg-expose steg-expose

setup-steg-expose:
	@if [ ! -d "steg-expose" ]; then \
		git clone https://github.com/b3dk7/StegExpose.git steg-expose; \
	else \
		echo "StegExpose already installed"; \
	fi

save_images:
	rm -rf data/save_and_div
	venv/bin/python -m trad_visual --model_path models/archived/norm-normal/DenseSteganoGAN/6/epoch_32.pth --save --model_type dense --save_path data/save_and_div --data_depth 6 --num_examples 100 --dataset_path data/COCO_val_2017
steg-expose:
	java -jar steg-expose/StegExpose.jar data/save_and_div default default steganalysis
steg-visualize:
	venv/bin/python -m trad_visual --model_path models/archived/norm-normal/DenseSteganoGAN/6/epoch_32.pth --model_type dense --dataset_path data/COCO_val_2017 --visualize --csv_path steganalysis --data_depth 6
analyze-stego:
	make setup-steg-expose
	make save_images
	make steg-expose
	make steg-visualize

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
		venv/bin/python metrics.py --model_type=basic --data_depth=1 --model_path=models/normal/BasicSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=2 --model_path=models/normal/BasicSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=3 --model_path=models/normal/BasicSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=4 --model_path=models/normal/BasicSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=5 --model_path=models/normal/BasicSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=6 --model_path=models/normal/BasicSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-residual:
		venv/bin/python metrics.py --model_type=residual --data_depth=1 --model_path=models/normal/ResidualSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=2 --model_path=models/normal/ResidualSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=3 --model_path=models/normal/ResidualSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=4 --model_path=models/normal/ResidualSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=5 --model_path=models/normal/ResidualSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=6 --model_path=models/normal/ResidualSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv


metrics-dense:
		venv/bin/python metrics.py --model_type=dense --data_depth=1 --model_path=models/normal/DenseSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=2 --model_path=models/normal/DenseSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=3 --model_path=models/normal/DenseSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=4 --model_path=models/normal/DenseSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=5 --model_path=models/normal/DenseSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=6 --model_path=models/normal/DenseSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv


metrics-all:
	make metrics-basic
	make metrics-residual
	make metrics-dense

metrics-basic-extra:
		venv/bin/python metrics.py --model_type=basic --data_depth=7 --model_path=models/normal/BasicSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=8 --model_path=models/normal/BasicSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=9 --model_path=models/normal/BasicSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=10 --model_path=models/normal/BasicSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=11 --model_path=models/normal/BasicSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=basic --data_depth=12 --model_path=models/normal/BasicSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-residual-extra:
		venv/bin/python metrics.py --model_type=residual --data_depth=7 --model_path=models/normal/ResidualSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=8 --model_path=models/normal/ResidualSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=9 --model_path=models/normal/ResidualSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=10 --model_path=models/normal/ResidualSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=11 --model_path=models/normal/ResidualSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=residual --data_depth=12 --model_path=models/normal/ResidualSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-dense-extra:
		venv/bin/python metrics.py --model_type=dense --data_depth=7 --model_path=models/normal/DenseSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=8 --model_path=models/normal/DenseSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=9 --model_path=models/normal/DenseSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=10 --model_path=models/normal/DenseSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=11 --model_path=models/normal/DenseSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
		venv/bin/python metrics.py --model_type=dense --data_depth=12 --model_path=models/normal/DenseSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-all-extra:
	make metrics-basic-extra
	make metrics-residual-extra
	make metrics-dense-extra

metrics-all-both:
	make metrics-all
	make metrics-all-extra