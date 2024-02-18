# TMT
The official implementation of “Token Masking Transformer for Weakly Supervised Object Localization”

To do：
- [x] Trained model for TMT
- [x] Training code (Coming soon)
- [x] Testing code (Coming soon)

### Installation
```
conda env create -f environment.yml
conda activate TMT
```
### Dataset
To prepare the datasets, you can download CUB and ISVRC from the links below. 
```
CUB: https://www.vision.caltech.edu/datasets/cub_200_2011/
ISVRC: https://www.image-net.org/challenges/LSVRC/2012/
```

For instance in CUB/deit_tmt_scm_small_patch16_224.yaml:
```
DATA:
 DATASET: CUB
 DATADIR: /root/Datasets/CUB_200_2011

TRAIN:
 BATCH_SIZE: 256

TEST:
 BATCH_SIZE: 256
```
You could change the DATADIR for the absolute path of the dataset or the BATCH_SIZE depending on your memory.

### Trained Models
[Google Drive](https://drive.google.com/drive/folders/1S4aXhPRpOmQjIop7tkZxC5FD-hkXBHbj?usp=drive_link)

### Traing
Please use the commands below to launch TMT. Replace {config_file} with the config file path you set up and learning rate {lr}.
```
python tools_cam/train_cam.py --config_file {config_file} --lr {lr}
```
```
# CUB
python tools_cam/train_cam.py --config_file ./configs/CUB/deit_scm_tmt_small_patch16_224.yaml --lr 5e-5
# ILSVRC
python tools_cam/train_cam.py --config_file ./configs/ILSVRC/deit_scm_tmt_small_patch16_224.yaml --lr 1e-6
```
Since only CUBV2 dataset is available for [MaxboxAcc](https://github.com/clovaai/wsolevaluation).
Please prepare the dataset CUBV2 at the root directory of CUB_200_2011 if you want to test it.
```
CUB_200_2011/CUBV2
```
We provide the default False option at tools_cam/train_cam.py
```
CUBV2=False
```
You can test MaxboxAcc when setting it to True.

### Resume Training
Add the ${pth_file_path} to continue the training if needed.
```
 --resume ${pth_file_path}
```

### Testing
Specify the path of .pth file for testing and visualization. 
```
# CUB
python tools_cam/test.py --config_file ./configs/CUB/deit_scm_tmt_small_patch16_224.yaml --resume ${pth_file_path}
# ILSVRC
python tools_cam/test.py --config_file ./configs/ILSVRC/deit_scm_tmt_small_patch16_224.yaml --resume ${pth_file_path}
```
Set up testing at tools_cam/test.py


