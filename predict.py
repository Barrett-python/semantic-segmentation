import pandas as pd
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cfg
from DaTransNet import DaTransNet
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as ff
import albumentations as A
import torchvision.transforms as transforms



class LabelProcessor:
    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):  # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        name = label
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label, 'name':name}
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        data = ff.resize(data, crop_size)
        label = ff.resize(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        label = np.array(label)  # 以免不是np格式的数据
        img = np.array(img)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)
        return img, label



label_processor = LabelProcessor(cfg.class_dict_path)
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=1, shuffle=False, num_workers=8)

net = DaTransNet().to(device)

net.load_state_dict(t.load("/mnt/DATA-1/DATA-2/Feilong/sematic_segmentation/weight/304.pth"))
net.eval()

pd_label_color = pd.read_csv('GOALS/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = "/mnt/DATA-1/DATA-2/Feilong/sematic_segmentation/example/Layer_Segmentations/"

for i, sample in enumerate(test_data):
	valImg = sample['img'].to(device)
	valLabel = sample['label'].long().to(device)
	name = sample['name'][0].split('/')[-1]
	print(name)
	out_0,out_1 = net(valImg)
	out = F.upsample(out_0 + out_1 , size=[800, 1100], mode='bilinear', align_corners=True)
	out = F.log_softmax(out, dim=1)
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	pre1.save(dir + str(name))
	print('Done')

os.system('zip -r example.zip example/')


