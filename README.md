# HAW


## About

This is the source code for paper _Peserving Potential Neighbors for Low-Degree
Nodes via Reweighting in Link Prediction_.

The source code is based on [WALKPOOL](https://github.com/DaDaCheng/WalkPooling/) 

## Requirements

python>=3.9

torch>=1.9.0

torch-cluster>=1.5.9

torch-geometric>=2.0.0

torch-scatter>=2.0.8

torch-sparse>=0.6.11

setuptools=58.0.4(<60)

tqdm

## Run

### Quick start

For evaluate HAW-enhanced GAEs:

	python ./src/eva_gae.py --data-name cora --model gae --haw 1 --non-uni 0

For evaluate HAW-enhanced WALKPOOL:

	python ./src/eva_wp.py --init-representation argva --data-name cora --haw 1 --non-uni 0

## Reference


If you find our work useful in your research, please cite our paper in the future:



