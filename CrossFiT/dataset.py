import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from PIL import Image
import torch
import numpy as np 
import openpyxl as pxl
import pandas as pd
from regist.crop import RandomResizedCrop_img, RandomCrop_img
from regist.affine import RandomAffine_img
from regist.flip import RandomHorizontallyFlip_img, RandomVerticallyFlip_img



class drtid(data.Dataset):
    def __init__(self, train=False, test=False):
        self.train = train
        self.test = test
        self.path = './DRTiD/'
        self.file = 'Original Images/'

        if train:
            e_file = 'Ground Truths/DR_grade/a. DR_grade_Training.csv' 
        else:
            e_file = 'Ground Truths/DR_grade/b. DR_grade_Testing.csv'   
        self.imgs = []

        csv_file = pd.read_csv(self.path + e_file)
        self.dict_label = {}
        for index, row in csv_file.iterrows():
            rank = row['Grade']
            img1_name = row['Macula']
            img2_name = row['Optic disc']
            # id = row['ID']
            self.imgs.append([self.path+self.file+img1_name+'.jpg',self.path+self.file+img2_name+'.jpg',rank])

        self.rgb_norm_global = T.Normalize(
            mean = [0.372487, 0.217266, 0.119367],
            std = [0.281526, 0.179457, 0.109162])

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'saturation': 0.4,
            'hue': 0.1,
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from
            'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
        }
        if train:
            self.transform_1 = T.Compose([
                T.Resize((640,640)),
                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                    saturation=data_aug['saturation'],
                    hue=data_aug['hue']
                ),
            ])

            # self.rhf = RandomHorizontallyFlip_img(p=0.5)
            # self.rvf = RandomVerticallyFlip_img(p=0.5)

            self.transform_2 = RandomResizedCrop_img(
                    size=(512),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                )

            self.transform_3 = RandomAffine_img(
                    degrees=data_aug['degrees'],
                    translate=data_aug['translate']
                )
            self.transform_4 = T.Compose([
                T.RandomGrayscale(0.2),
                T.ToTensor(),
                self.rgb_norm_global
            ])
        else:
            self.transform = T.Compose([
                T.Resize((512,512)),
                T.ToTensor(),
                self.rgb_norm_global
            ])

        print(len(self.imgs))
        
    def __getitem__(self, index):
        img1, img2, label = self.imgs[index]

        data1 = Image.open(img1).convert('RGB')
        data2 = Image.open(img2).convert('RGB')

        if self.train:
            data1 = self.transform_1(data1)
            data2 = self.transform_1(data2)

            # data1, data2 = self.rhf(data1,data2)
            # data1, data2 = self.rvf(data1,data2)

            data1, data2 = self.transform_2(data1,data2)
            data1, data2 = self.transform_3(data1,data2)

            data1 = self.transform_4(data1)
            data2 = self.transform_4(data2)
        else:
            data1 = self.transform(data1)
            data2 = self.transform(data2)

        id = img1.split('/')[-1]
        return data1, data2, label, id

    def __len__(self):
        return len(self.imgs)


class deepdrid_clf(data.Dataset):
    def __init__(self, train=False, val=False, test=False, epoch=0):
        self.train = train
        self.val = val
        self.test = test
        self.path = './DeepDRiD/'

        if train:
            self.file = 'Regular_DeepDRiD/regular_train/'
            e_file = 'DR_label/DR_label/regular-fundus-training.csv'
        elif val:
            self.file = 'Regular_DeepDRiD/regular_valid/'
            e_file = 'DR_label/DR_label/regular-fundus-validation.csv'      
        elif test:
            self.file = 'Regular_DeepDRiD/regular_test/'   
        
        img_list = [[[] for _ in range(2)] for _ in range(5)]

        if test:
            for i in range(0,len(os.listdir(self.path+self.file)),2):
                image_1_id = sorted(os.listdir(self.path+self.file))[i]
                image_2_id = sorted(os.listdir(self.path+self.file))[i+1]
                self.imgs.append([self.path + self.file + image_1_id, self.path + self.file + image_2_id, -1, image_1_id[:-1]])

        else:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image_id']
                lr = image_id[-2]
                if lr == 'l':
                    rank = int(row['left_eye_DR_Level'])
                    img_list[rank][0].append(image_id)
                if lr == 'r':
                    rank = int(row['right_eye_DR_Level'])
                    img_list[rank][1].append(image_id)
            if val or train:
                self.imgs=[]
                for i in range(5):
                    for j in range(2):
                        print(i,j,len(img_list[i][j])//2,'pairs')
                        for index in range(0,len(img_list[i][j]),2):
                            image_1_id = img_list[i][j][index]
                            image_2_id = img_list[i][j][index+1]
                            self.imgs.append([self.path+self.file+image_1_id+'.jpg', self.path+self.file+image_2_id+'.jpg', i, image_1_id[:-1]])
                      
        self.rgb_norm_global = T.Normalize(
            mean = [0.380463, 0.234838, 0.142012],
            std = [0.326214, 0.215156, 0.157832]
        )

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'saturation': 0.4,
            'hue': 0.1,
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from
            'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
        }
        if train:
            self.transform_1 = T.Compose([
                T.Resize((640,640)),
                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                    saturation=data_aug['saturation'],
                    hue=data_aug['hue']
                ),
            ])
            self.transform_2 = RandomResizedCrop_img(
                    size=(384),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                )
            self.transform_3 = RandomAffine_img(
                    degrees=data_aug['degrees'],
                    translate=data_aug['translate']
                )
            self.transform_4 = T.Compose([
                T.RandomGrayscale(0.2),
                T.ToTensor(),
                self.rgb_norm_global
            ])
        elif val or test:
            self.transform = T.Compose([
                T.Resize((384,384)),
                T.ToTensor(),
                self.rgb_norm_global
            ])

        print(len(self.imgs)) 


    def __getitem__(self, index):
            
        img1, img2, label, id = self.imgs[index]

        data1 = Image.open(img1).convert('RGB')
        data2 = Image.open(img2).convert('RGB')

        if self.train:
            data1 = self.transform_1(data1)
            data2 = self.transform_1(data2)

            data1, data2 = self.transform_2(data1,data2)
            data1, data2 = self.transform_3(data1,data2)

            data1 = self.transform_4(data1)
            data2 = self.transform_4(data2)
        else:
            data1 = self.transform(data1)
            data2 = self.transform(data2)

        return data1, data2, label, id

    def __len__(self):
        return len(self.imgs)
    
