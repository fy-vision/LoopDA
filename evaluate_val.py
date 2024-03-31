import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as torch_data
import os
from PIL import Image
from torch.autograd import Variable

from util.metrics import runningScore
from model.model import SharedEncoder
from util.utils import load_models
from util.loader.DarkZurichValLoader import DarkZurichValLoader
from util.loader.NightDrivingValLoader import NightDrivingValLoader

num_classes = 19
DATASET_NAME = 'DarkZurich'
DATA_PATH_VAL = './data/DarkZurich'
DATA_LIST_PATH_VAL_IMG  = './util/loader/dark_zurich_list/zurich_val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/dark_zurich_list/label_zurich.txt'
#DATA_PATH_VAL = './data/NighttimeDrivingTest/'
#DATA_LIST_PATH_VAL_IMG  = './util/loader/dark_zurich_list/nd_val.txt'
#DATA_LIST_PATH_VAL_LBL  = './util/loader/dark_zurich_list/nd_val_label.txt'
WEIGHT_DIR = './weights'
DEFAULT_GPU = 0
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='LoopDA for domain adaptive nighttime semantic segmentation')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, help='the name of the dataset.')
parser.add_argument('--data_path_val', type=str, default=DATA_PATH_VAL, help='the path to target val dataset.')
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class InvalidDatasetError(Exception):
    pass


args = parser.parse_args()

print ('gpu:', ','.join(args.gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

if args.dataset_name == 'NightDriving':
    val_set   = NightDrivingValLoader(args.data_path_val, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[540, 960], mean=IMG_MEAN)
elif args.dataset_name == 'DarkZurich':
    val_set   = DarkZurichValLoader(args.data_path_val, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[540, 960], mean=IMG_MEAN)
else:
    raise InvalidDatasetError(f'Invalid Dataset Name: {args.dataset_name}')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

upsample_1080 = nn.Upsample(size=[1080, 1920], mode='bilinear', align_corners=True)
upsample_540 = nn.Upsample(size=[540, 960], mode='bilinear', align_corners=True)

model_dict = {}

enc_shared = SharedEncoder(bn_clr=True).cuda()
model_dict['student'] = enc_shared


load_models(model_dict, args.weight_dir)

enc_shared.eval()


cty_running_metrics = runningScore(num_classes)
print('evaluating models ...')
for i_val, (images_val, labels_val) in enumerate(val_loader):
    print(i_val)
    images_val = Variable(images_val.cuda(), requires_grad=False)
    with torch.no_grad():
        _, _, pred, _ = enc_shared(images_val[:, [2, 1, 0], :, :])
    pred = upsample_540(pred)
    pred = pred.data.max(1)[1].cpu().numpy()
    gt = labels_val.data.cpu().numpy()
    cty_running_metrics.update(gt, pred)
cty_score, cty_class_iou = cty_running_metrics.get_scores()

for k, v in cty_score.items():
    print(k, v)
