import os
import tqdm
import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image
from sklearn.metrics import confusion_matrix
from glob import glob
from dense_fuse_net_o4 import *
from dataload import *
from torch.utils.data import DataLoader
# 输入必须为灰度图
# labels为像素值的类别

def get_miou_recall_precision(label_image, pred_image, labels):
    label = label_image.reshape(-1)
    pred = pred_image.reshape(-1)
    out = confusion_matrix(label, pred, labels=labels)
    r, l = out.shape
    iou_temp = 0
    recall = []
    precision = []
    for i in range(r):
        TP = out[i][i]
        temp = np.concatenate((out[0:i, :], out[i + 1:, :]), axis=0)
        sum_one = np.sum(temp, axis=0)
        FP = sum_one[i]
        temp2 = np.concatenate((out[:, 0:i], out[:, i + 1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        TN = temp2.reshape(-1).sum() - FN
        iou_temp += (TP / (TP + FP + FN)) if (TP + FP + FN)!=0 else 0
        recall.append((TP / (TP + FN)) if (TP + FN)!=0 else 0)
        precision.append((TP / (TP + FP)) if (TP + FP)!=0 else 0)
    MIOU = iou_temp / len(labels)
    return MIOU, recall, precision

def get_evaluate(net, train_data_path, save_path='', mode=False, device=torch.device('cpu')):
    Mious=[]
    recalls=[]
    precisions=[]
    data_loader = DataLoader(MyDataset(train_data_path, start=180), batch_size=1)
    for i, (img,label) in enumerate(tqdm.tqdm(data_loader)):
        img_data=img.to(device)
        # img_data=torch.unsqueeze(img_data,dim=0)
        net.eval()
        out_img=net(img_data)
        out=torch.argmax(out_img[0][0],dim=0)
        if device==torch.device('cuda'):
            pred=out.cpu().detach().numpy()
        else:pred=out.detach().numpy()
        l, p = np.array(label).astype(int), np.array(pred).astype(int)
        MIOU, recall, precision = get_miou_recall_precision(l, p, [0, 1])
        Mious.append(MIOU)
        recalls.append(recall)
        precisions.append(precision)
        if mode:
            output = Image.fromarray(np.uint8(pred),mode='P')
            output.save(os.path.join(save_path,f'result_{i}.png'))
    avg_miou=np.average(np.array(Mious))
    avg_recall=np.average(np.array(recalls),axis=0).tolist()
    avg_precision=np.average(np.array(precisions),axis=0).tolist()
    if mode:
        print(f'Sucessfully predict {len(Mious)} images and save all in file:{save_path}!')
    else:print(f'Sucessfully predict {len(Mious)} images!')
    return avg_miou, avg_recall, avg_precision

if __name__ == '__main__':
    net = DenseFuseNet().cuda()
    # net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.load_state_dict(torch.load('params/gcnunet_all1.pth'))
    m,r,p = get_evaluate(net, r'data/test',r'test_result', mode=True)
    print(m,r,p)