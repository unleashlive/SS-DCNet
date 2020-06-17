import sys, os
import scipy.io as sio
import torch
import numpy as np

from pathlib import Path
from torchvision import transforms
from PIL import Image

from .Network.SSDCNet import SSDCNet_classify
from .load_data_V2 import get_pad


verbose = False
cuda = False


def vprint(*data):
    if verbose:
        print(*data)


FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)
FILE_DIR = os.path.abspath(FILE_DIR)
vprint(FILE_DIR)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
        if not os.path.isabs(img_path):
            img_path = os.path.join(FILE_DIR, img_path)
        if not os.path.exists(img_path):
            print("File not found",file=sys.stderr)
            exit()
    else:
        print("Please enter file name",file=sys.stderr)
        exit()
    mod_path = 'best_epoch.pth'
    max_num = 7
    step = 0.5
    label_indice = np.arange(step,max_num+step,step)
    add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45])
    label_indice = np.concatenate( (add,label_indice) )
    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice)+1
    div_times = 2
    psize, pstride = 64, 64
    if cuda:
        net = SSDCNet_classify(class_num,label_indice,div_times=div_times,\
            frontend_name='VGG16',block_num=5,\
            IF_pre_bn=False,IF_freeze_bn=False,load_weights=True,\
            psize=psize,pstride = pstride,parse_method ='maxp').cuda()
    else:
        net = SSDCNet_classify(class_num,label_indice,div_times=div_times,\
            frontend_name='VGG16',block_num=5,\
            IF_pre_bn=False,IF_freeze_bn=False,load_weights=True,\
            psize=psize,pstride = pstride,parse_method ='maxp').cpu()
    if not os.path.isabs(mod_path):
        mod_path = os.path.join(FILE_DIR, mod_path)
    if os.path.exists(mod_path):
        if cuda:
            all_state_dict = torch.load(mod_path)
        else:
            all_state_dict = torch.load(mod_path, map_location='cpu')
        net.load_state_dict(all_state_dict['net_state_dict'])
        tmp_epoch_num = all_state_dict['tmp_epoch_num']
        with torch.no_grad():
            net.eval()
            rgb_dir = os.path.join(FILE_DIR, 'rgbstate.mat')
            mat = sio.loadmat(rgb_dir)
            rgb = mat['rgbMean'].reshape(1, 1, 3)
            image = Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)
            image = image[None,:,:,:]
            image = get_pad(image,DIV=64)
            image = image - torch.Tensor(rgb).view(3,1,1)
            if cuda:
                image = image.cuda()
            image = image.type(torch.float32)
            features = net(image)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div'+str(net.div_times)]
            del merge_res
            pre =  (outputs).sum()
            print('%d' % (pre))


