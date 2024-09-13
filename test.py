import torch
import tqdm
from evaluate import *
from dense_fuse_net_o4 import *
import re

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = DenseFuseNet().to(device)
net.load_state_dict(torch.load('params/DFN1.pth'))
train_data_path = r'train_dataset'


def test(net, path, device=torch.device('cpu')):
    torch.cuda.empty_cache()
    names = glob(path + '/img/*.png')
    for name in tqdm.tqdm(names):
        num = re.findall(r'\d+', name)[0]

        img = Image.open(name)
        w, h = img.size
        mask = Image.new("RGB", (512, 512), (0, 0, 0))
        mask.paste(img, (0, 0))
        tran1 = transforms.ToTensor()
        mask = tran1(mask)
        mask = torch.unsqueeze(mask, dim=0)
        mask = mask.to(device)

        out = net(mask)
        result = torch.argmax(out[0][0], dim=0).to(torch.float32).detach()
        if device == torch.device('cuda'):
            result = result.cpu()
        tran2 = transforms.ToPILImage()
        result = tran2(result)
        result = result.crop((0, 0, w, h))
        result.save(f'test_dataset_A/test_label/{num}.png')


def val(net, val_data_path, device=torch.device('cpu')):
    miou, recall, precision = get_evaluate(net, val_data_path, device=device)
    print(f'miou:{miou}, recall:{recall}, precision:{precision}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    test(net, 'test_dataset_A', device=device)
    # val(net, train_data_path, device=device)
    # Sucessfully predict 35 images!
    # miou: 0.7765996248130544, recall: [0.9778737614195026, 0.7903640664856102], precision: [0.9818377051986946, 0.7229707536344403]
