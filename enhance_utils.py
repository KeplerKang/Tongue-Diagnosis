from PIL import Image
from random import *

#resize大小原为256
def keep_image_size_open(path, size=(512,512)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
def keep_image_size_open_rgb(path, size=(512,512)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

# 随机裁剪旋转镜像
def image_both_enhancement_open(image_path,segment_path, size=(512,512)):
    ori_img = Image.open(image_path)
    seg_img = Image.open(segment_path)
    if ori_img.size!=seg_img.size:
        print('ori and seg img size error!')
    # 裁剪
    if random()>0.8:
        w, h = ori_img.size[0], ori_img.size[1]
        left, up = randint(0, w // 3), randint(0, h // 3)
        right, below = left + (w // 3)*2, up + (h // 3)*2
        ori_img = ori_img.crop((left, up, right, below))
        seg_img = seg_img.crop((left, up, right, below))
    # 旋转
    angle=randint(0,3)*10
    ori_img = ori_img.rotate(angle)
    seg_img = seg_img.rotate(angle)
    # 镜像
    if randint(0,1)==1:
        ori_img = ori_img.transpose(Image.FLIP_LEFT_RIGHT)
        seg_img = seg_img.transpose(Image.FLIP_LEFT_RIGHT)

    ori_mask = Image.new('RGB', (ori_img.size[0],ori_img.size[1]))
    ori_mask.paste(ori_img, (0, 0))
    ori_mask = ori_mask.resize(size)
    seg_mask = Image.new('P', (ori_img.size[0],ori_img.size[1]))
    seg_mask.paste(seg_img, (0, 0))
    seg_mask = seg_mask.resize(size)
    return ori_mask,seg_mask