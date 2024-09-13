# this is a run program
import os
import logging
import numpy as np
import torch
from collections import OrderedDict
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from PIL import Image
from glob import glob

from dataload import *
from evaluate import *
from dense_fuse_net_o4 import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = r'train_dataset'
weight_path = r'params/DFN1.pth'
num_classes = 2  # 背景也为一类
data_loader = DataLoader(MyDataset(train_data_path, 0, 180), batch_size=1)

net = DenseFuseNet().to(device)
net.load_state_dict(torch.load('params/DFN1.pth'))
# net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
loss_fun = nn.CrossEntropyLoss()


def get_loss(out_img, seg_img):  # densefusenet用
    train_loss = []
    for j in range(len(out_image)):  # 对四个输出分别求loss,权重从大图到小图依次为0.4,0.3,0.2,0.1
        pool = nn.MaxPool2d(kernel_size=(2 ** j, 2 ** j))
        train_loss.append(loss_fun(out_img[j], pool(seg_img)))
    avg_train_loss = train_loss[0] * 0.4 + train_loss[1] * 0.3 + train_loss[2] * 0.2 + train_loss[3] * 0.1
    return avg_train_loss


opt = optim.Adam(net.parameters(), lr=0.0002)
scheduler = lr_scheduler.MultiStepLR(opt, milestones=[120, 250], gamma=0.5)
logging.basicConfig(filename='log/DFN1.log', filemode='a', format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)
epoch = 1
max = 0
max_epoch = 0

while epoch <= 300:
    loss = []
    torch.cuda.empty_cache()
    for i, (image, label) in enumerate(tqdm.tqdm(data_loader)):
        image, label = image.to(device), label.to(device)
        out_image = net(image)
        # print(out_image[0].cpu().detach().shape)
        train_loss = loss_fun(out_image[0], label)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        loss.append(train_loss.cpu().detach().numpy())
        logging.info(f"epoch={epoch}, lr={opt.param_groups[0]['lr']}, loss={train_loss}")  # np.mean(loss)

    torch.save(net.state_dict(), weight_path)
    print(f'Successfully save weights in {weight_path} in epoch:{epoch}!')
    torch.cuda.empty_cache()

    if epoch % 300 == 0:
        miou, recall, precision = get_evaluate(net=net, device=device, train_data_path=train_data_path)
        print(f'miou:{miou}, recall:{recall}, precision:{precision}')
        if miou >= max:
            max = miou
            max_epoch = epoch

    scheduler.step()
    epoch += 1
