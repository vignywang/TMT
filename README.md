# TMT
The official implementation of “Token Masking Transformer for Weakly Supervised Object Localization”

To do：
- [x] Trained model for TMT
- [ ] Training code (Coming soon)
- [ ] Testing code (Coming soon)

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

## Trained Models
[Google Drive](https://drive.google.com/drive/folders/1S4aXhPRpOmQjIop7tkZxC5FD-hkXBHbj?usp=drive_link)
