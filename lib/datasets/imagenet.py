import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np


def get_transforms(cfg):
    
    train_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE)),
        transforms.RandomCrop((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize(list(map(float, cfg.DATA.IMAGE_MEAN)), list(map(float, cfg.DATA.IMAGE_STD)))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(list(map(float, cfg.DATA.IMAGE_MEAN)), list(map(float, cfg.DATA.IMAGE_STD)))
    ])

    test_tencrops_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE)),
        transforms.TenCrop(cfg.DATA.CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
                 (transforms.ToTensor()(crop)) for crop in crops])),
    ])
    return train_transform, test_transform, test_tencrops_transform


class ImageNetDataset(Dataset):
    """ 'ImageNet <https://image-net.org/index.php>'

        Args:
            root (string): Root directory of dataset where directory "ImageNet_ILSVRC2012" exists.
            cfg (dict): Hyperparameter configuration.
            is_train (bool): If True. create dataset from training set, otherwise creates from test set.
        """
    def __init__(self, root, cfg, is_train, val=False):
        self.root = root
        self.cfg = cfg
        self.is_train = is_train
        self.resize_size = cfg.DATA.RESIZE_SIZE
        self.crop_size = cfg.DATA.CROP_SIZE

        if self.is_train:
            datalist = os.path.join(self.root, 'ILSVRC2012_list', 'train.txt')
            self.image_dir = os.path.join(self.root, 'train')
            datalist = [datalist]

        else:
            datalist = os.path.join(self.root, 'ILSVRC2012_list', 'val_folder_new.txt')
            self.image_dir = os.path.join(self.root, 'val')

        names = []
        labels = []
        bboxes = []
        if self.is_train:
            for datalist in datalist_new:
                with open(datalist) as f:
                    for line in f:
                        info = line.strip().split()
                        names.append(info[0][:-5])
                        labels.append(int(info[1]))
                        if self.is_train is False:
                            bboxes.append(np.array(list(map(float, info[2:]))).reshape(-1, 4))

        with open(datalist) as f:
            for line in f:
                info = line.strip().split()
                names.append(info[0][:-5])
                labels.append(int(info[1]))
                if self.is_train is False:
                    bboxes.append(np.array(list(map(float, info[2:]))).reshape(-1,4))
                    # bboxes.append([float(info[i]) for i in range(2, 6)])
        self.names = names
        self.labels = labels
        if self.is_train is False:
            self.bboxes = bboxes

        self.train_transform, self.onecrop_transform, self.tencrops_transform = get_transforms(cfg)
        if cfg.TEST.TEN_CROPS:
            self.test_transform = self.tencrops_transform
        else:
            self.test_transform = self.onecrop_transform

        self.val = val
        

    def __getitem__(self, idx):

        if self.val:
            name = self.val2_names[idx]
            label = self.val2_class_labels[name]
            image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
            bbox = self.val2_boxes[name]
        else:        
            name = self.names[idx]
            label = self.labels[idx]

            self.image_dir = os.path.join(self.root, 'train')
            image = Image.open(os.path.join(self.image_dir, name + '.JPEG')).convert('RGB')
            
        image_size = list(image.size)
            
        if self.is_train:
            image = self.train_transform(image)
            return image, label

        else:
            image = self.test_transform(image)
            bbox = self.bboxes[idx]
            
            [x1, y1, x2, y2] = np.split(bbox, 4, 1)

            resize_size = self.crop_size
            crop_size = self.crop_size
            shift_size = 0
            [image_width, image_height] = image_size
            left_bottom_x = np.maximum(x1 / image_width * resize_size - shift_size, 0).astype(int)
            left_bottom_y = np.maximum(y1 / image_height * resize_size - shift_size, 0).astype(int)
            right_top_x = np.minimum(x2 / image_width * resize_size - shift_size, crop_size - 1).astype(int)
            right_top_y = np.minimum(y2 / image_height * resize_size - shift_size, crop_size - 1).astype(int)

            gt_bbox = np.concatenate((left_bottom_x, left_bottom_y, right_top_x, right_top_y),axis=1).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            return image, label, gt_bbox, name+'.jpg', image, image

    def __len__(self):
        return len(self.names)


