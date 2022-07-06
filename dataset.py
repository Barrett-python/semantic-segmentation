import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg
import random
import torch
import cv2


class OCT_Dataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
    
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        self.crop_size = crop_size
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
                
    def __getitem__(self, index):
        image = self.rgb_loader(self.imgs[index])
        gt = cv2.imread(self.labels[index])
    
        gt_img  = np.array(gt)
#        gt_img[gt_img == 80] = 1
#        gt_img[gt_img == 160] = 2
#        gt_img[gt_img == 255] = 3
#        gt_img = Image.fromarray(gt_img)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
#            gt_img = self.gt_transform(gt)
            gt_img[gt_img == 80] = 1
            gt_img[gt_img == 160] = 2
            gt_img[gt_img == 255] = 3
            gt_img = cv2.resize(gt_img,(512, 512))
            gt_img = gt_img[:,:,1]
            gt_img = torch.from_numpy(gt_img)
#            print(torch.unique(gt_img))
        
        sample = {'img': image, 'label': gt_img}
        return sample

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

            
    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def __len__(self):
        return len(self.imgs)


