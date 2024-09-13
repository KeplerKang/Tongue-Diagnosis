import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])

def keep_image_size_open(path,label=False,size=(512,512)):
    #把图片大小固定为256*256,这个你可以自己改,把图片读进来
    img=Image.open(path)
    temp=512
    #(0，0，0）表示掩码这张rgb图像是全黑的
    if label == False:
        mask=Image.new("RGB",(temp,temp),(0,0,0))
    else:mask=Image.new("1",(temp,temp),0)
    #将原图粘上来：(0,0）表示粘到左上角
    mask.paste(img,(0,0))
    #现在原图已经粘到mask上面了，此时对它进行resize，缩放到你想要的，传入的那个size
    # mask=mask.resize(size)
    return mask

class MyDataset(Dataset):
    def __init__(self, path,start=0,end=215):
        self.path = path
        self.name = glob(os.path.join(self.path, 'img/')+'*.png')[start:end]

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        img_path = self.name[index]  # xx.png
        label_path = img_path.replace('img', 'label')
        img=keep_image_size_open(img_path,label=False)
        label=keep_image_size_open(label_path,label=True)
        return transform(img), transform(label).squeeze(0).long()

if __name__ == '__main__':
    data = MyDataset('train_dataset',1,3)
    print(len(data),data[0][1].shape)
    # print(type(data[0][1]))
    # print(set(data[0][0].numpy().flatten().tolist()))
