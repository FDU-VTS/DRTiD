import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from PIL import Image
import torch
import numpy as np 
import openpyxl as pxl
import pandas as pd
from torch.nn import functional as F
from regist.crop import RandomResizedCrop
from regist.affine import RandomAffine 
from regist.flip import RandomHorizontallyFlip, RandomVerticallyFlip



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

        op_path = 'Ground Truths/Optic_Macula_Localization/op_ma_localization.csv' 
        op_file = pd.read_csv(self.path+op_path)
        self.dict_op = {}
        for index, row in op_file.iterrows():
            image_name = row['image'].split('.')[0]
            self.dict_op[image_name] = [row['op_x'],row['op_y']]
            
            
        csv_file = pd.read_csv(self.path + e_file)
        self.dict_label = {}
        for index, row in csv_file.iterrows():
            rank = row['Grade']
            img1_name = row['Macula']
            img2_name = row['Optic disc']
            img1_op = self.dict_op[img1_name]
            img2_op = self.dict_op[img2_name]
            self.imgs.append([self.path+self.file+img1_name+'.jpg',self.path+self.file+img2_name+'.jpg',rank,img1_op,img2_op])

        self.rgb_norm_global = T.Normalize(
            mean = [0.372487, 0.217266, 0.119367],
            std = [0.281526, 0.179457, 0.109162]
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

            # self.rhf = RandomHorizontallyFlip(p=0.5)
            # self.rvf = RandomVerticallyFlip(p=0.5)

            self.transform_2 = RandomResizedCrop(
                    size=(512),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                )
            self.transform_3 = RandomAffine(
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
        img1, img2, label, img1_op, img2_op = self.imgs[index]

        data1 = Image.open(img1).convert('RGB')
        data2 = Image.open(img2).convert('RGB')
        
        if self.train:
            data1 = self.transform_1(data1)
            data2 = self.transform_1(data2)

            # data1, data2, img1_op, img2_op = self.rhf(data1,data2,img1_op,img2_op)
            # data1, data2, img1_op, img2_op = self.rvf(data1,data2,img1_op,img2_op)

            data1_result, data2_result = self.transform_2(data1,data2,img1_op,img2_op)
            data1, img1_op_new = data1_result[0], data1_result[1]
            data2, img2_op_new = data2_result[0], data2_result[1]

            data1, data2, img1_op_new, img2_op_new = self.transform_3(data1,data2,img1_op_new,img2_op_new)

            data1 = self.transform_4(data1)
            data2 = self.transform_4(data2)

        else:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
            img1_op_new = img1_op
            img2_op_new = img2_op

        img1_opx, img1_opy = img1_op_new[0], img1_op_new[1]
        img2_opx, img2_opy = img2_op_new[0], img2_op_new[1]

        x_trans = img2_opx - img1_opx
        y_trans = img2_opy - img1_opy

        theta = torch.tensor([
            [1,0,x_trans*2],
            [0,1,y_trans*2] 
            ], dtype=torch.float)
        img1_grid = F.affine_grid(theta.unsqueeze(0), data2.unsqueeze(0).shape) # 1,512,512,2
        img1_grid = img1_grid.squeeze(0) # 512,512,2

        id = img1.split('/')[-1]

        return data1, data2, label, img1_grid, id

    def __len__(self):
        return len(self.imgs)


class deepdrid_clf(data.Dataset):
    def __init__(self, train=False, val=False, test=False, tta_val=False, tta_test=False):
        self.train = train
        self.val = val
        self.test = test
        self.tta_val = tta_val
        self.tta_test = tta_test
        self.path = './DeepDRiD/'

        if train:
            self.file = 'Regular_DeepDRiD/regular_train/'
            e_file = 'DR_label/DR_label/regular-fundus-training.csv'
        if val or tta_val:
            self.file = 'Regular_DeepDRiD/regular_valid/'
            e_file = 'DR_label/DR_label/regular-fundus-validation.csv'      
        if test or tta_test:
            self.file = 'Regular_DeepDRiD/regular_test/'  
            e_file = 'DR_label/DR_label/Challenge1_labels_test.csv' 
        
        op_path = 'deepdrid_op.csv' 
        op_file = pd.read_csv(self.path+op_path)
        self.dict_op = {}
        self.dict_if_op = {}
        self.dict_key = {}
        for index, row in op_file.iterrows():
            image_name = row['image'].split('.')[0]
            self.dict_op[image_name] = [row['op_x'],row['op_y']]
            self.dict_if_op[image_name] = row['if_op']
            self.dict_key[image_name] = [row['key_x'],row['key_y']]


        img_list = [[[] for _ in range(2)] for _ in range(5)]
        self.imgs = []
        if test or tta_test:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows(): 
                image_id = row['image_id']
                rank = row['DR_Levels']
                self.dict_label[image_id]=rank

            for i in range(0,len(os.listdir(self.path+self.file)),2):
                image_1 = sorted(os.listdir(self.path+self.file))[i]
                image_2 = sorted(os.listdir(self.path+self.file))[i+1]
                image_1_id = image_1.split('.')[0]
                image_2_id = image_2.split('.')[0]
                label1 = self.dict_label[image_1_id]
                label2 = self.dict_label[image_2_id]

                img1_op = self.dict_op[image_1_id]
                img2_op = self.dict_op[image_2_id]

                if_op1 = self.dict_if_op[image_1_id]
                if_op2 = self.dict_if_op[image_2_id]
                img1_key = self.dict_key[image_1_id]
                img2_key = self.dict_key[image_2_id]
                if if_op1 * if_op2 == 1:
                    self.imgs.append([self.path+self.file+image_1_id+'.jpg', self.path+self.file+image_2_id+'.jpg', label1, image_1_id[:-1],img1_op,img2_op])
                else: # no op, use key point
                    self.imgs.append([self.path+self.file+image_1_id+'.jpg', self.path+self.file+image_2_id+'.jpg', label1, image_1_id[:-1],img1_key,img2_key])
                
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
            if val or train or tta_val:
                self.imgs=[]
                for i in range(5):
                    for j in range(2):
                        print(i,j,len(img_list[i][j])//2,'pairs')
                        for index in range(0,len(img_list[i][j]),2):
                            image_1_id = img_list[i][j][index]
                            image_2_id = img_list[i][j][index+1]
                            img1_op = self.dict_op[image_1_id]
                            img2_op = self.dict_op[image_2_id]

                            if_op1 = self.dict_if_op[image_1_id]
                            if_op2 = self.dict_if_op[image_2_id]
                            img1_key = self.dict_key[image_1_id]
                            img2_key = self.dict_key[image_2_id]
                            if if_op1 * if_op2 == 1:
                                self.imgs.append([self.path+self.file+image_1_id+'.jpg', self.path+self.file+image_2_id+'.jpg', i, image_1_id[:-1],img1_op,img2_op])
                            else: # no op, use key point
                                self.imgs.append([self.path+self.file+image_1_id+'.jpg', self.path+self.file+image_2_id+'.jpg', i, image_1_id[:-1],img1_key,img2_key])

        self.rgb_norm_global = T.Normalize(
            mean = [0.380463, 0.234838, 0.142012],
            std = [0.326214, 0.215156, 0.157832]
        )

        if train:
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
        elif tta_val or tta_test:
            data_aug = {
                'brightness': 0.4,  # how much to jitter brightness
                'contrast': 0.4,  # How much to jitter contrast
                'saturation': 0.4,
                'hue': 0.1,
                'scale': (0.8, 1.2),  # range of size of the origin size cropped
                'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
                'degrees': (0, 0),  # range of degrees to select from
                'translate': (0, 0)  # tuple of maximum absolute fraction for horizontal and vertical translations
            }            
        if train or tta_val or tta_test:
            self.transform_1 = T.Compose([
                T.Resize((640,640)),
                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                    saturation=data_aug['saturation'],
                    hue=data_aug['hue']
                ),
            ])
            # self.rhf = RandomHorizontallyFlip(p=0.5)
            # self.rvf = RandomVerticallyFlip(p=0.5)
            self.transform_2 = RandomResizedCrop(
                    size=(384),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                )
            self.transform_3 = RandomAffine(
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
            
        img1, img2, label, id, img1_op, img2_op = self.imgs[index]

        data1 = Image.open(img1).convert('RGB')
        data2 = Image.open(img2).convert('RGB')

        if self.train or self.tta_val or self.tta_test:
            data1 = self.transform_1(data1)
            data2 = self.transform_1(data2)

            # data1, data2, img1_op, img2_op = self.rhf(data1,data2,img1_op,img2_op)
            # data1, data2, img1_op, img2_op = self.rvf(data1,data2,img1_op,img2_op)

            data1_result, data2_result = self.transform_2(data1,data2,img1_op,img2_op)
            data1, img1_op_new = data1_result[0], data1_result[1]
            data2, img2_op_new = data2_result[0], data2_result[1]

            data1, data2, img1_op_new, img2_op_new = self.transform_3(data1,data2,img1_op_new,img2_op_new)

            data1 = self.transform_4(data1)
            data2 = self.transform_4(data2)

        else:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
            img1_op_new = img1_op
            img2_op_new = img2_op

        img1_opx, img1_opy = img1_op_new[0], img1_op_new[1]
        img2_opx, img2_opy = img2_op_new[0], img2_op_new[1]

        x_trans = img2_opx - img1_opx
        y_trans = img2_opy - img1_opy

        theta = torch.tensor([
            [1,0,x_trans*2],
            [0,1,y_trans*2] 
            ], dtype=torch.float)
        img1_grid = F.affine_grid(theta.unsqueeze(0), data2.unsqueeze(0).shape) # 1,512,512,2
        img1_grid = img1_grid.squeeze(0) # 512,512,2

        return data1, data2, label, img1_grid, id

    def __len__(self):
        return len(self.imgs)
    

