import os
import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
import cv2
import numpy as np
import pandas as pd
import torchvision as tv
import time
import Augmentor as aug
from . import image_augmentation as ia


# light_spectrum = {
#     "0":"RGB 1",
#     "1":"UVA 365nm",
#     "2":"Violet 405nm",
#     "3":"Indigo 430nm",
#     "4":"Blue 450nm",
#     "5":"Blue 470nm",
#     "6":"Azure 490nm",
#     "7":"Cyan 515nm",
#     "8":"Green 540nm",
#     "9":"Yellow 570nm",
#     "10":"Amber 590nm",
#     "11":"Red 630nm",
#     "12":"Red 645nm",
#     "13":"Red 660nm",
#     "14":"Red 690nm",
#     "15":"NTR 780nm",
#     "16":"NTR 850nm",
#     "17":"NTR 880nm",
#     "18":"NTR 940nm",
#     "19":"NTR 970nm"
# }

def load_seed_data_v1(image_dir,label_dir,maxlen = 10000):
    '''
    create a sample for each (x,y) pair
    '''
    seed_images = [] # [n, l, h, w]
    seed_shapes = [] # [n, 15]
    seed_light_spectrums = [] # [n, l, 2]
    seed_process_days = [] # [n,]
    seed_labels = [] # [n, 3]
    each_sample_types = []
    
    seed_types = [v for v in os.listdir(image_dir) if v.startswith('CY')]
    for seed_type in seed_types: # 5 types
        seed_type_dir = os.path.join(image_dir,seed_type)
        process_times = os.listdir(seed_type_dir) #
        for process_time in process_times:
            xlsx_path = os.path.join(os.path.join(label_dir,seed_type),seed_type+"_%st.xlsx"%(process_time))
            df = pd.read_excel(xlsx_path,sheet_name=[0,1,2,4])
            light_index = 0
            for i,v in enumerate(df[1].values):
                light_index += 1
                if 'Mean' in v:
                    break
            process_time_dir = os.path.join(seed_type_dir,process_time)
            for i in range(50):
                npy_path = os.path.join(process_time_dir,"%d.npy"%i)
                if os.path.exists(npy_path):
                    seed_image = np.load(npy_path)
                else:
                    seed_image = []
                    for j in range(20):
                        light_spectrum_dir = os.path.join(process_time_dir,str(j))
                        if os.path.exists(light_spectrum_dir):
                            image_path = os.path.join(light_spectrum_dir,str(i+1)+".png")
                            if j == 0:
                                image = cv2.imread(image_path,1)
                            else:
                                image = cv2.imread(image_path,0)
                            image = ia.resize_pad_image(image)
                            seed_image.append(image)
                    seed_image = np.concatenate([seed_image[0],np.stack(seed_image[1:],axis=-1)],axis=-1)
                
                seed_images.append(seed_image)
                seed_shapes.append(np.array(df[0].values[i,1:-1],dtype=np.int32))
                seed_light_spectrum = df[1].values[light_index+i*24:light_index+i*24+19,1:6]
                seed_light_spectrums.append(np.array(seed_light_spectrum,np.float32))
                seed_process_days.append(int(process_time))
                seed_label = df[4].values[i,[1,2,-1]]
                seed_label[0] = 1 if seed_label[0] == '是' else 0
                seed_labels.append(seed_label)
                each_sample_types.append(seed_type)
                if len(seed_images) == maxlen:
                    return np.array(seed_images), \
                            np.array(seed_shapes),\
                            np.array(seed_light_spectrums), \
                            np.array(seed_process_days),\
                            np.array(seed_labels)
    return np.array(seed_images), \
            np.array(seed_shapes),\
            np.array(seed_light_spectrums), \
            np.array(seed_process_days),\
            np.array(seed_labels),\
            np.array(each_sample_types)

def load_seed_data_v2(image_dir,label_dir,maxlen = 10000):
    '''
    create a sample for each (x,y) pair
    '''
    seed_images = [] # [n, l, h, w]
    seed_shapes = [] # [n, 15]
    seed_light_spectrums = [] # [n, l, 2]
    seed_process_days = [] # [n,]
    seed_labels = [] # [n, 3]
    each_sample_types = []
    
    # seed_types = [v for v in os.listdir(image_dir) if v.startswith('CY')]
    seed_types = ['CY-7','CY-7-1','CY-80','CY-80-1','CY-203','CY-203-1','CY-229','CY-229-1','CY-256','CY-256-1']
    for seed_type in seed_types: # 5 types
        seed_type_dir = os.path.join(image_dir,seed_type)
        process_times = os.listdir(seed_type_dir) #
        for process_time in process_times:
            xlsx_tail = "_%st.xlsx"%(process_time) if process_time != '0' else '.xlsx' 
            xlsx_type = seed_type if len(seed_type.split('-'))==2 else seed_type[:-2]
            si = 0 if len(seed_type.split('-'))==2 else 50
            xlsx_path = os.path.join(label_dir,xlsx_type+xlsx_tail)
            df = pd.read_excel(xlsx_path,sheet_name=[0,1,2,4])
            light_index = 0
            for i,v in enumerate(df[1].values):
                light_index += 1
                if 'Mean' in v:
                    break
            process_time_dir = os.path.join(seed_type_dir,process_time)
            for i in range(50):
                seed_image = []
                for j in range(20):
                    light_spectrum_dir = os.path.join(process_time_dir,str(j))
                    if os.path.exists(light_spectrum_dir):
                        image_path = os.path.join(light_spectrum_dir,str(i+1)+".png")
                        if j == 0:
                            image = cv2.imread(image_path,1)
                        else:
                            image = cv2.imread(image_path,0)
                        image = ia.resize_pad_image(image)
                        seed_image.append(image)
                seed_image = np.concatenate([seed_image[0],np.stack(seed_image[1:],axis=-1)],axis=-1)
                seed_images.append(seed_image)
                # seed_shapes.append(np.array(df[0].values[si+i,1:-1],dtype=np.int32))
                seed_shapes.append(np.random.rand(15,))
                # seed_light_spectrum = df[1].values[light_index+i*24:light_index+i*24+19,1:6]
                # seed_light_spectrums.append(np.array(seed_light_spectrum,np.float32))
                seed_light_spectrums.append(np.random.rand(19,2))
                seed_process_days.append(int(process_time))
                seed_label = df[4].values[si+i,[1,2,-1]]
                seed_label[0] = 1 if seed_label[0] == '是' else 0
                seed_labels.append(seed_label)
                each_sample_types.append(seed_type)
                if len(seed_images) == maxlen:
                    return np.array(seed_images), \
                            np.array(seed_shapes),\
                            np.array(seed_light_spectrums), \
                            np.array(seed_process_days),\
                            np.array(seed_labels)
    return np.array(seed_images), \
            np.array(seed_shapes),\
            np.array(seed_light_spectrums), \
            np.array(seed_process_days),\
            np.array(seed_labels),\
            np.array(each_sample_types)

def build_dataset(image_dir,label_dir,presave_path='/second_disk/SeedClassification/data_22c_v1',test_num=120,wants=['train','eval'],data_type='v1'):
    if os.path.exists(presave_path):
        print('loading data form ndarray...')
        if 'train' in wants:
            train_seed_images = np.load(os.path.join(presave_path,'train/seed_images.npy'),allow_pickle=True)
            train_seed_shapes = np.load(os.path.join(presave_path,'train/seed_shapes.npy'),allow_pickle=True)
            train_seed_light_spectrums = np.load(os.path.join(presave_path,'train/seed_light_spectrums.npy'),allow_pickle=True)
            train_seed_process_days = np.load(os.path.join(presave_path,'train/seed_process_days.npy'),allow_pickle=True)
            train_seed_labels = np.load(os.path.join(presave_path,'train/seed_labels.npy'),allow_pickle=True)
            train_seed_types = np.load(os.path.join(presave_path,'train/seed_types.npy'),allow_pickle=True)
            train_set = SeedDataset(
                train_seed_images, 
                train_seed_shapes, 
                train_seed_light_spectrums, 
                train_seed_process_days, 
                train_seed_labels,
                train_seed_types,
                mode='train'
            )
        else:
            train_set = None
        if 'eval' in wants:
            test_seed_images = np.load(os.path.join(presave_path,'test/seed_images.npy'),allow_pickle=True)
            test_seed_shapes = np.load(os.path.join(presave_path,'test/seed_shapes.npy'),allow_pickle=True)
            test_seed_light_spectrums = np.load(os.path.join(presave_path,'test/seed_light_spectrums.npy'),allow_pickle=True)
            test_seed_process_days = np.load(os.path.join(presave_path,'test/seed_process_days.npy'),allow_pickle=True)
            test_seed_labels = np.load(os.path.join(presave_path,'test/seed_labels.npy'),allow_pickle=True)
            test_seed_types = np.load(os.path.join(presave_path,'test/seed_types.npy'),allow_pickle=True)
            test_set = SeedDataset(
                test_seed_images, 
                test_seed_shapes, 
                test_seed_light_spectrums, 
                test_seed_process_days, 
                test_seed_labels,
                test_seed_types,
                mode='train'
            )
        else:
            test_set = None
    else:
        if data_type == 'v1':
            seed_images,seed_shapes,seed_light_spectrums,seed_process_days,seed_labels,seed_types = \
                load_seed_data_v1(image_dir, label_dir)
        elif data_type == 'v2':
            seed_images,seed_shapes,seed_light_spectrums,seed_process_days,seed_labels,seed_types = \
                load_seed_data_v2(image_dir, label_dir)
        else:
            raise NotImplementedError()
        os.mkdir(presave_path)
        train_path = os.path.join(presave_path,'train')
        test_path = os.path.join(presave_path,'test')
        os.mkdir(train_path)
        os.mkdir(test_path)
        np.save(os.path.join(presave_path,'seed_images.npy'),seed_images)
        np.save(os.path.join(presave_path,'seed_shapes.npy'),seed_shapes)
        np.save(os.path.join(presave_path,'seed_light_spectrums.npy'),seed_light_spectrums)
        np.save(os.path.join(presave_path,'seed_process_days.npy'),seed_process_days)
        np.save(os.path.join(presave_path,'seed_labels.npy'),seed_labels)
        np.save(os.path.join(presave_path,'seed_types.npy'),seed_types)
    
        label = seed_labels[:,0]
        t_index = np.where(label==1)[0]
        f_index = np.where(label==0)[0]
        t_index = t_index[[i * (t_index.shape[0] // (test_num//2)) for i in range(test_num//2)]]
        f_index = f_index[[i * (f_index.shape[0] // (test_num//2)) for i in range(test_num//2)]]
        test_index = list(t_index) + list(f_index)
        test_index.sort()
        train_index = list(set(list(range(label.shape[0]))) - set(test_index))

        train_seed_images = seed_images[train_index]
        train_seed_shapes = seed_shapes[train_index]
        train_seed_light_spectrums = seed_light_spectrums[train_index]
        train_seed_process_days = seed_process_days[train_index]
        train_seed_labels = seed_labels[train_index]
        train_seed_types = seed_types[train_index]
        np.save(os.path.join(train_path,'seed_images.npy'),train_seed_images)
        np.save(os.path.join(train_path,'seed_shapes.npy'),train_seed_shapes)
        np.save(os.path.join(train_path,'seed_light_spectrums.npy'),train_seed_light_spectrums)
        np.save(os.path.join(train_path,'seed_process_days.npy'),train_seed_process_days)
        np.save(os.path.join(train_path,'seed_labels.npy'),train_seed_labels)
        np.save(os.path.join(train_path,'seed_types.npy'),train_seed_types)

        test_seed_images = seed_images[test_index]
        test_seed_shapes = seed_shapes[test_index]
        test_seed_light_spectrums = seed_light_spectrums[test_index]
        test_seed_process_days = seed_process_days[test_index]
        test_seed_labels = seed_labels[test_index]
        test_seed_types = seed_types[test_index]
        np.save(os.path.join(test_path,'seed_images.npy'),test_seed_images)
        np.save(os.path.join(test_path,'seed_shapes.npy'),test_seed_shapes)
        np.save(os.path.join(test_path,'seed_light_spectrums.npy'),test_seed_light_spectrums)
        np.save(os.path.join(test_path,'seed_process_days.npy'),test_seed_process_days)
        np.save(os.path.join(test_path,'seed_labels.npy'),test_seed_labels)
        np.save(os.path.join(test_path,'seed_types.npy'),test_seed_types)
    
    if 'train' in wants:
        train_set = SeedDataset(
            train_seed_images, 
            train_seed_shapes, 
            train_seed_light_spectrums, 
            train_seed_process_days, 
            train_seed_labels,
            train_seed_types,
            mode='train'
        )
    else:
        train_set = None
    if 'eval' in wants:
        test_set = SeedDataset(
            test_seed_images, 
            test_seed_shapes, 
            test_seed_light_spectrums, 
            test_seed_process_days, 
            test_seed_labels,
            test_seed_types,
            mode='eval'
        )
    else:
        test_set = None
    return train_set,test_set


class SeedDataset(Dataset):
    def __init__(self,seed_images,seed_shapes,seed_light_spectrums,seed_process_days,seed_labels,seed_types,mode='train'):
        self.seed_images = seed_images
        self.seed_shapes = seed_shapes
        self.seed_light_spectrums = seed_light_spectrums
        self.seed_process_days = seed_process_days
        self.seed_labels = seed_labels
        self.seed_types = seed_types
        self.mode = mode
        
    
    def __getitem__(self, index):
        seed_image = self.seed_images[index]
        if self.mode == 'train':
            if np.random.rand(1) > 0.7:
                seed_image = np.ascontiguousarray(seed_image[:,::-1,...])
            if np.random.rand(1) > 0.7:
                seed_image = ia.random_rotation(seed_image,45,[127.5,127.5,127.5,127.5])
        
        seed_image = torch.from_numpy(seed_image)
        seed_image = (seed_image - 127.5) / 127.5
        sample = {
            'seed_image':seed_image,
            'seed_shape':torch.from_numpy(self.seed_shapes[index]),
            'seed_light':torch.from_numpy(self.seed_light_spectrums[index]),
            'seed_process_day':torch.tensor(self.seed_process_days[index]),
            'seed_label':torch.tensor(self.seed_labels[index][0]),
            # 'seed_type':torch.tensor(self.seed_types[index]),
        }
        return sample
    def __len__(self):
        return len(self.seed_images)
            
        