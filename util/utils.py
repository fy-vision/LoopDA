'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]


def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def poly_lr_scheduler_warm(base_lr, iter, warmup = 250, max_iter=80000, power=1.0):
    if iter<=warmup:
        return base_lr * (iter / warmup)
    else:
	    return base_lr * ((1 - float(iter-warmup) / max_iter) ** (power))



def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_warm(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler_warm(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def save_models(model_dict, prefix='./'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key+'.pth'))

def load_models(model_dict, prefix='./'):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(os.path.join(prefix, key+'.pth')))

def label_one_hot(labels_batch, num_classes=19):
    labels = labels_batch.clone()
    labels[labels == 255] = num_classes
    label_one_hot = torch.nn.functional.one_hot(labels, num_classes + 1).float().cuda()
    label_one_hot = label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :]
    return label_one_hot


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_mean = feat_mean.repeat(1, 1, size[2], size[3])
    return feat_mean


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        ten = tensor.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, self.mean, self.std):
            t.mul_(s).add_(m)
        return ten.permute(3, 0, 1, 2)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        ten = tensor.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, self.mean, self.std):
            t.sub_(m).div_(s)
        return ten.permute(3, 0, 1, 2)

def process_label(label, class_numbers=19):
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, class_numbers + 1, w, h).cuda()
    id = torch.where(label < class_numbers, label, torch.Tensor([class_numbers]).cuda())
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1


def momentum_pseudo_label(pred, thres_prior, beta=0.005):
    B, C, H, W = pred.size()
    predicted_label = np.zeros((B, 512, 496))
    predicted_prob = np.zeros((B, 512, 496))
    upsample_512 = nn.Upsample(size=[512, 496], mode='bilinear', align_corners=True)

    for i in range(B):
        #print('pred[i].size()', pred[i].size())
        output = F.softmax(pred[i].reshape(-1,C,H,W), dim=1)
        #print('outshape:', output.size())
        output = upsample_512(output).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        #print('prob is:', prob)
        predicted_label[i] = label.copy()
        predicted_prob[i] = prob.copy()

    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[int(np.floor(len(x) * 0.5))])
    #print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    #print(thres)
    for i in range(19):
        if thres[i]==0.0:
            thres[i] = thres_prior[i]
    thres_prior = [x * (1-beta) for x in thres_prior]
    thres = [x * beta for x in thres]
    thres_momentum = [sum(x) for x in zip(thres_prior, thres)]

    label_pseudo_out = []
    for index in range(B):
        label_pred = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label_pred[(prob < thres_momentum[i]) * (label_pred == i)] = 255
        label_pseudo = np.asarray(label_pred, dtype=np.uint8)
        label_pseudo = torch.from_numpy(label_pseudo).reshape(-1,512,496)
        #print('label_pseudo_size():', label_pseudo.size())
        #print('label_pseudo:', label_pseudo)
        label_pseudo_out.append(label_pseudo)
    return torch.cat(label_pseudo_out).long().cuda(), thres_momentum


def ias_thresh(conf_dict, alpha, w=None, gamma=1.0, num_class = 19):
    if w is None:
        w = np.ones(num_class)
    # threshold
    cls_thresh = np.ones(num_class,dtype = np.float32)
    for idx_cls in np.arange(0, num_class):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh


def adaptive_instance_normalization(content_feat, style_feat):
    if content_feat.size()[:2] != style_feat.size()[:2]:
        return content_feat
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_old(style_feat)
    content_mean, content_std = calc_mean_std_old(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std_old(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    feat_var = feat.view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    return feat_mean, feat_std

def semantic_aware_recoloring(content, style, label_content, label_style):
    content_feat = content.clone()
    style_feat = style.clone()
    if content_feat.size()[:2] != style_feat.size()[:2]:
        return content_feat
    label_content_onehot = label_one_hot(label_content)
    label_style_onehot = label_one_hot(label_style)
    for idx in range(content_feat.size()[0]):
        cls_set = set(torch.unique(label_content[idx]).tolist()).intersection(set(torch.unique(label_style[idx]).tolist()))
        #print('label_content:', list(torch.unique(label_content[idx])))
        #print('label_style:', list(torch.unique(label_style[idx])))
        #print('cls_set:', cls_set)
        isEmpty = (len(cls_set) == 0)
        if isEmpty:
            continue
        else:
            #mean_dict_style = {}
            #var_dict_style = {}
            #mean_dict_content = {}
            #var_dict_content = {}
            for cls in cls_set:
                #print('cls is:', cls)
                mask_style = label_style_onehot[idx,cls,:,:].unsqueeze(0)
                masked_style_feat = style_feat[idx] * mask_style
                size = masked_style_feat.size()
                masked_style_mean, _ = calc_mean_std(masked_style_feat) #C*1*1
                masked_style_mean = masked_style_mean *(size[-1] * size[-2]) / mask_style.sum()
                style_feat_tmp = masked_style_feat + masked_style_mean.expand(size) * (1-mask_style) #C*H*W
                _, masked_style_std = calc_mean_std(masked_style_feat) #C*1*1
                masked_style_std = masked_style_std *(size[-1] * size[-2])**0.5 / (mask_style.sum())**0.5
                #mask_style_mean = (masked_style_feat).sum() / mask_style.sum()
                #mask_style_std = torch.sqrt(torch.square((masked_style_feat + mask_style_mean * (1-mask_style)) - mask_style_mean).sum() / (mask_style.sum() -1))
                #mean_dict_style[cls] = mask_style_mean
                #var_dict_style[cls] = mask_style_std
                
                mask_content = label_content_onehot[idx,cls,:,:].unsqueeze(0)
                masked_content_feat = content_feat[idx] * mask_content
                masked_content_mean, _ = calc_mean_std(masked_content_feat) #C*1*1
                masked_content_mean = masked_content_mean*(size[-1] * size[-2]) / mask_content.sum()
                content_feat_tmp = masked_content_feat + masked_content_mean.expand(size) * (1-mask_content) #C*H*W
                _, masked_content_std = calc_mean_std(masked_content_feat) #C*1*1
                masked_content_std = masked_content_std *(size[-1] * size[-2])**0.5 / (mask_content.sum())**0.5
                #mask_content_mean = (masked_content_feat).sum() / mask_content.sum()
                #mask_content_std = torch.sqrt(torch.square((masked_content_feat + mask_content_mean * (1 - mask_content)) - mask_content_mean).sum() / (mask_content.sum() -1))
                #mean_dict_content[cls] = mask_content_mean
                #var_dict_content[cls] = mask_content_std
                
                normalized_content_feat = (content_feat[idx] - masked_content_mean.expand(
                    size)) / masked_content_std.expand(size)
                normalized_content_feat = normalized_content_feat * masked_style_std.expand(size) + masked_style_mean.expand(size)
                content_feat[idx] = mask_content*normalized_content_feat + (1-mask_content)*content_feat[idx]
    return content_feat
                
                

                
