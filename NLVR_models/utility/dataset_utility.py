import os
import glob
import numpy as np
from scipy import misc
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import torchvision.transforms.functional as TF
from scipy.ndimage.interpolation import rotate
        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size,data_prop=1, transform=None, shuffle=False,rotate=False,vertical_flip=False, vertical_roll =False,
                horizontal_flip = False, horizontal_roll= False, max_rotate_angle = 0,matrix=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*", "*.npz")) \
                            if dataset_type in f]
        self.file_names = random.sample(self.file_names,int(data_prop*len(self.file_names)))
        self.img_size = img_size
#         self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), allow_pickle=True)
        self.shuffle = shuffle
        self.switch = [3,4,5,0,1,2,6,7]
        self.rotate = rotate
        self.vertical_flip = vertical_flip
        self.vertical_roll = vertical_roll
        self.horizontal_flip = horizontal_flip
        self.horizontal_roll = horizontal_roll
        self.max_rotate_angle = max_rotate_angle
        self.matrix = matrix
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = data["structure"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]
        meta_matrix = data["meta_matrix"] 

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = list(range(8))
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            switch_2_rows = np.random.rand()            
            if switch_2_rows < 0.5:                
                context = context[self.switch, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        if self.vertical_flip:
            vertical_flip_rand = np.random.rand()
#             vertical_flip_transform = T.RandomVerticalFlip(p=1)
        if self.horizontal_flip:
            horizontal_flip_rand = np.random.rand()
#             horizontal_flip_transform = T.RandomHorizontalFlip(p=1)
        if self.vertical_roll:
            vertical_roll_rand = np.random.rand()
            v_shift = random.randint(0, 80)
            v_roll_transform =  lambda x: np.roll(x, shift=v_shift, axis=0)
            
        if self.horizontal_roll:
            horizontal_roll_rand = np.random.rand()
            h_shift = random.randint(0, 80)
            h_roll_transform =  lambda x: np.roll(x, shift=h_shift, axis=1)
            
        if self.rotate:
            rotate_rand = np.random.rand()
            angle = random.randint(0, self.max_rotate_angle)
            rotate_transform = lambda x: rotate(x,angle)
        
        resize_image = []
        
        
        for idx in range(0, 16):
            
              
            img = image[idx,:,:]
#             print(img.shape)
            if self.vertical_roll:
                if vertical_roll_rand >0.5:
                    img = v_roll_transform(img)
#                     print(img.shape)
            if self.horizontal_roll:
                if horizontal_roll_rand >0.5:
                    img = h_roll_transform(img)
#                     print(img.shape)
            if self.vertical_flip:
                if vertical_flip_rand >0.5:
#                     print("v_flip")
                    img = np.flipud(img)
            if self.horizontal_flip:
                if horizontal_flip_rand >0.5:
                    img = np.fliplr(img)
            if self.rotate:
                if rotate_rand >0.5:
                    img = rotate_transform(img)
           
            img = np.array(Image.fromarray(img).resize((self.img_size, self.img_size)))
#             misc.imresize(, (self.img_size, self.img_size)) 
#             print(img.shape)
            resize_image.append(img)
        resize_image = np.stack(resize_image)

        

        
        
        
        # image = resize(image, (16, 128, 128))
        # meta_matrix = data["mata_matrix"]

#         embedding = torch.zeros((6, 300), dtype=torch.float)
#         indicator = torch.zeros(1, dtype=torch.float)
#         element_idx = 0
#         for element in structure:
#             if element != '/':
#                 embedding[element_idx, :] = torch.tensor(self.embeddings.item().get(element), dtype=torch.float)
#                 element_idx += 1
#         if element_idx == 6:
#             indicator[0] = 1.
#         # if meta_target.dtype == np.int8:
        #     meta_target = meta_target.astype(np.uint8)
        # if meta_structure.dtype == np.int8:
        #     meta_structure = meta_structure.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            # meta_matrix = self.transform(meta_matrix)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
#             meta_structure = self.transform(meta_structure)
            # meta_target = torch.tensor(meta_target, dtype=torch.long)
        if self.matrix:
            mat_inner = np.zeros((4,5))
            mat_outer = np.zeros((4,5))
            for x in meta_matrix[0:4,:]:
                for i in range(4):
                    for j in range(5):
                        if x[i] == 1 and x[4+j] == 1:
                            mat_inner[i][j] += 1
            for x in meta_matrix[4:,:]:
                for i in range(4):
                    for j in range(5):
                        if x[i] == 1 and x[4+j] == 1:
                            mat_outer[i][j] += 1
            rule_attribute_pair_matrix = mat_inner+mat_outer 
            return resize_image, target, meta_target, rule_attribute_pair_matrix
        
        return resize_image, target, meta_target
        
