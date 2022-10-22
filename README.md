# GARL
This work "Air-Ground Spatial Crowdsourcing with UAV Carriers by Geometric Graph Convolutional Multi-Agent Deep Reinforcement Learning" has been submitted in ICDE 2023.
## :page_facing_up: Description
GARL is a  novel MADRL model, which consists of a multi-center attention-based graph convolutional network (GCN) to accurately extract UGV specific features from UGV stop network called "MC-GCN", and a novel GNN-based communication mechanism called "E-Comm" to make the cooperation among UGVs adaptive to constant changing of geometric shapes formed by UGVs.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- NVIDIA GPU (RTX A6000) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/Richard19980527/Air-Ground-SC-with-UAV-Carriers.git
    cd Air-Ground-SC-with-UAV-Carriers
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```
Then the usage information will be shown as following
```
usage: main.py [-h] dataset_name method_name mode

positional arguments:
  dataset_name     the name of dataset (KAIST or UCLA)
  method_name  the name of method (GARL)
  mode         train or test
 
optional arguments:
  -h, --help   show this help message and exit
```
Test the trained models provided in [Air-Ground-SC-with-UAV-Carriers/log](https://github.com/Richard19980527/Air-Ground-SC-with-UAV-Carriers/tree/main/log).
```
python main.py KAIST GARL test
python main.py UCLA GARL test
```
## :computer: Training

We provide complete training codes for GARL.<br>
You could adapt it to your own needs.

1. If you don't have NVIDIA RTX A6000, you should comment these two lines in file
[Air-Ground-SC-with-UAV-Carriers/code/util.py](https://github.com/Richard19980527/Air-Ground-SC-with-UAV-Carriers/tree/main/code/util.py).
	```
	[24]  torch.backends.cuda.matmul.allow_tf32 = False
	[25]  torch.backends.cudnn.allow_tf32 = False
	```
2. Training
	```
	python main.py KAIST GARL train
	python main.py UCLA GARL train
	```
	The log files will be stored in [Air-Ground-SC-with-UAV-Carriers/log](https://github.com/Richard19980527/Air-Ground-SC-with-UAV-Carriers/tree/main/log).
## :checkered_flag: Testing
1. Testing
	```
	python main.py KAIST GARL test
	python main.py UCLA GARL test
	```
## :e-mail: Contact

If you have any question, please email `2656886245@qq.com`.
