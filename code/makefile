.PHONY: setup-steg-expose steg-expose


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
	mv DIV2K_train_LR_bicubic ../data/DIV2K_train_LR_bicubic

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
	unzip DIV2K_valid_LR_bicubic_X4.zip
	rm -rf DIV2K_valid_LR_bicubic_X4.zip
	mv DIV2K_valid_LR_bicubic ../data/DIV2K_valid_LR_bicubic

	curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip
	unzip DIV2K_valid_LR_unknown_X4.zip
	rm -rf DIV2K_valid_LR_unknown_X4.zip
	mv DIV2K_valid_LR_unknown ../data/DIV2K_valid_LR_unknown

	curl -O http://images.cocodataset.org/zips/val2017.zip
	unzip val2017.zip
	rm -rf val2017.zip
	mv val2017 ../data/COCO_val_2017


# Run Commands
run:
	venv/bin/python run.py --visualizer="message_decode_full" --model_type="dense" --data_depth=6 --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --image_path=../data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"

run-full:
	venv/bin/python run.py --visualizer="message_decode_full" --model_type="dense" --data_depth=6 --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --image_path=../data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"

run-small:
	venv/bin/python run.py --visualizer="message_decode_small" --model_type="dense" --data_depth=6 --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --image_path=../data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"

run-accuracy:
	venv/bin/python run.py --visualizer="rsbpp_accuracy" --model_type="dense" --data_depth=6 --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --image_path=../data/DIV2K_valid_LR_bicubic/X4 --text="I Love Deep Learning!"


# Steganalysis Commands
setup-steg-expose:
	@if [ ! -d "steg-expose" ]; then \
		git clone https://github.com/b3dk7/StegExpose.git steg-expose; \
	else \
		echo "StegExpose already installed"; \
	fi

save_images:
	rm -rf save_and_div
	venv/bin/python -m traditional --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --model_type=dense --save_path=save_and_div --data_depth=6 --num_examples=100 --dataset_path=../data/DIV2K_test_LR_unknown/X4 --save

steg-expose:
	java -jar steg-expose/StegExpose.jar save_and_div default default steganalysis

steg-visualize:
	venv/bin/python -m traditional --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --model_type=dense --dataset_path=../data/DIV2K_test_LR_unknown/X4  --csv_path=steganalysis --data_depth=6 --visualize

analyze-stego:
	make setup-steg-expose
	make save_images
	make steg-expose
	make steg-visualize


# Training Commands
train:
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=6

train-basic:
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=6

train-residual:
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=6

train-dense:
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=6

train-all:
	make train-basic
	make train-residual
	make train-dense

train-basic-depth:
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=32 --data_depth=12

train-residual-depth:
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=32 --data_depth=12

train-dense-depth:
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=7
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=8
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=9
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=10
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=11
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=32 --data_depth=12

train-all-depth:
	make train-basic-depth
	make train-residual-depth
	make train-dense-depth

train-basic-leaky:
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=LeakyBasicSteganoGAN --epochs=32 --data_depth=6

train-residual-leaky:
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=LeakyResidualSteganoGAN --epochs=32 --data_depth=6

train-dense-leaky:
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=1
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=2
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=3
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=4
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=5
	venv/bin/python -m network.train --model=LeakyDenseSteganoGAN --epochs=32 --data_depth=6

train-all-leaky:
	make train-basic-leaky
	make train-residual-leaky
	make train-dense-leaky

train-basic-long:
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m network.train --model=BasicSteganoGAN --epochs=64 --data_depth=6

train-residual-long:
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m network.train --model=ResidualSteganoGAN --epochs=64 --data_depth=6

train-dense-long:
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=1
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=2
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=3
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=4
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=5
	venv/bin/python -m network.train --model=DenseSteganoGAN --epochs=64 --data_depth=6

train-all-long:
	make train-basic-long
	make train-residual-long
	make train-dense-long


# Metrics Commands
metrics:
	venv/bin/python metrics.py --data_depth=2 --model_path=../results/models/normal/DenseSteganoGAN/2/epoch_32.pth --visualize

metrics-basic:
	venv/bin/python metrics.py --model_type=basic --data_depth=1 --model_path=../results/models/normal/BasicSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=2 --model_path=../results/models/normal/BasicSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=3 --model_path=../results/models/normal/BasicSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=4 --model_path=../results/models/normal/BasicSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=5 --model_path=../results/models/normal/BasicSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=6 --model_path=../results/models/normal/BasicSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-residual:
	venv/bin/python metrics.py --model_type=residual --data_depth=1 --model_path=../results/models/normal/ResidualSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=2 --model_path=../results/models/normal/ResidualSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=3 --model_path=../results/models/normal/ResidualSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=4 --model_path=../results/models/normal/ResidualSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=5 --model_path=../results/models/normal/ResidualSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=6 --model_path=../results/models/normal/ResidualSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-dense:
	venv/bin/python metrics.py --model_type=dense --data_depth=1 --model_path=../results/models/normal/DenseSteganoGAN/1/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=2 --model_path=../results/models/normal/DenseSteganoGAN/2/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=3 --model_path=../results/models/normal/DenseSteganoGAN/3/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=4 --model_path=../results/models/normal/DenseSteganoGAN/4/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=5 --model_path=../results/models/normal/DenseSteganoGAN/5/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=6 --model_path=../results/models/normal/DenseSteganoGAN/6/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-all:
	make metrics-basic
	make metrics-residual
	make metrics-dense

metrics-basic-depth:
	venv/bin/python metrics.py --model_type=basic --data_depth=7 --model_path=../results/models/normal/BasicSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=8 --model_path=../results/models/normal/BasicSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=9 --model_path=../results/models/normal/BasicSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=10 --model_path=../results/models/normal/BasicSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=11 --model_path=../results/models/normal/BasicSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=basic --data_depth=12 --model_path=../results/models/normal/BasicSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-residual-depth:
	venv/bin/python metrics.py --model_type=residual --data_depth=7 --model_path=../results/models/normal/ResidualSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=8 --model_path=../results/models/normal/ResidualSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=9 --model_path=../results/models/normal/ResidualSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=10 --model_path=../results/models/normal/ResidualSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=11 --model_path=../results/models/normal/ResidualSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=residual --data_depth=12 --model_path=../results/models/normal/ResidualSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-dense-depth:
	venv/bin/python metrics.py --model_type=dense --data_depth=7 --model_path=../results/models/normal/DenseSteganoGAN/7/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=8 --model_path=../results/models/normal/DenseSteganoGAN/8/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=9 --model_path=../results/models/normal/DenseSteganoGAN/9/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=10 --model_path=../results/models/normal/DenseSteganoGAN/10/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=11 --model_path=../results/models/normal/DenseSteganoGAN/11/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv
	venv/bin/python metrics.py --model_type=dense --data_depth=12 --model_path=../results/models/normal/DenseSteganoGAN/12/epoch_32.pth --visualize --save_path=results/visualizations/ --csv_path=results/steganogan_results.csv

metrics-all-depth:
	make metrics-basic-depth
	make metrics-residual-depth
	make metrics-dense-depth
