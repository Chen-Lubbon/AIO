import os
import torch as t
from torch.functional import split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import random

class LoadData(Dataset):
    def __init__(self, root=None, random_sequence=False, total_num=None, random_seed=None):
        self.root = root
        self.random_sequence = random_sequence
        self.total_num = total_num
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            
        required_folders = ['landscape', 'degraded', 'mask', 'transmission', 'atmosphere']
        
        if all(folder in os.listdir(self.root) for folder in required_folders):
            self.root_input = os.path.join(self.root, 'degraded')
            self.root_label = os.path.join(self.root, 'landscape')
            self.root_transmission = os.path.join(self.root, 'transmission')
            self.root_mask = os.path.join(self.root, 'mask')
            self.root_atmosphere = os.path.join(self.root, 'atmosphere')
            
            input_imgs = self.__rankdata__(self.root_input)
            label_imgs = self.__rankdata__(self.root_label)
            transmission_imgs = self.__rankdata__(self.root_transmission)
            mask_imgs = self.__rankdata__(self.root_mask)
            atmosphere_datas = self.__rankdata__(self.root_atmosphere)
            
            indices = list(range(len(input_imgs)))
            if self.random_sequence:
                combined = list(zip(indices, input_imgs, label_imgs, transmission_imgs, mask_imgs, atmosphere_datas))
                random.shuffle(combined)
                indices, input_imgs, label_imgs, transmission_imgs, mask_imgs, atmosphere_datas = zip(*combined)


        if self.total_num is not None:
            input_imgs = input_imgs[:self.total_num]
            label_imgs = label_imgs[:self.total_num]
            transmission_imgs = transmission_imgs[:self.total_num]
            mask_imgs = mask_imgs[:self.total_num]
            atmosphere_datas = atmosphere_datas[:self.total_num]
            self.indices = indices[:self.total_num] if self.random_sequence else None
        else:
            self.indices = indices if self.random_sequence else None


        imgs_num = len(input_imgs)
        print(f'Load data中数据集加载完成: {imgs_num} 张图像')
        self.h, self.w = cv2.imread(os.path.join(self.root_input, input_imgs[0])).shape[:2]

        self.transforms = T.Compose([
            # T.ToPILImage(),
            # T.RandomRotation(10),
            # T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor()
        ])    
        

        self.input_imgs = input_imgs
        self.transmission_imgs = transmission_imgs
        self.mask_imgs = mask_imgs
        self.atmosphere_datas = atmosphere_datas
        self.label_imgs = label_imgs

    def __rankdata__(self, path):
        images = os.listdir(path)
        images = [img for img in images if img.endswith('.jpg') or img.endswith('. png') or img.endswith('.txt')]
        images = sorted(images, key=lambda x: int(x.split(".")[-2].split("/")[-1])) 
        return images
        
    def __getitem__(self, index):
        input_img_path = os.path.join(self.root_input, self.input_imgs[index])
        transmission_img_path = os.path.join(self.root_transmission, self.transmission_imgs[index])
        mask_img_path = os.path.join(self.root_mask, self.mask_imgs[index])
        atmosphere_data_path = os.path.join(self.root_atmosphere, self.atmosphere_datas[index])
        label_img_path = os.path.join(self.root_label, self.label_imgs[index])
        
        input_image = cv2.imdecode(np.fromfile(input_img_path, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)  
        transmission_map = cv2.imdecode(np.fromfile(transmission_img_path, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
        mask_map = cv2.imdecode(np.fromfile(mask_img_path, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
        # atmosphere_illu = np.loadtxt(atmosphere_data_path)
        atmosphere_illu = np.loadtxt(atmosphere_data_path, dtype=float, usecols=range(3), delimiter=' ')
        label_image = cv2.imdecode(np.fromfile(label_img_path, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)  
        
        input_image = input_image.astype("float32") / 255
        transmission_map = transmission_map.astype("float32") / 255
        mask_map = mask_map.astype("float32") / 255
        label_image = label_image.astype("float32") / 255
        
        label_image = cv2.resize(label_image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        
        input_image = self.transforms(input_image)
        transmission_map = self.transforms(transmission_map)
        atmosphere_illu = t.tensor(atmosphere_illu, dtype=t.float32)
        mask_map = self.transforms(mask_map)
        label_image = self.transforms(label_image)

        return input_image, transmission_map, mask_map, atmosphere_illu, label_image

    def __len__(self):
        return len(self.input_imgs)
    
    def __get_indices__(self):
        return self.indices



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_path = "data"

    train_diff = LoadData(data_path, train=True)
    # print(train_diff[0][1])
    print(train_diff[4][0].shape)
    # plt.imshow( train_diff[0][1].permute(1,2,0) )
    # plt.show()
