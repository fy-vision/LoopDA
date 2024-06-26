import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

valid_colors = [[128,  64, 128],
                [244,  35, 232],
                [ 70,  70,  70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170,  30],
                [220, 220,   0],
                [107, 142,  35],
                [152, 251, 152],
                [ 70, 130, 180],
                [220,  20,  60],
                [255,   0,   0],
                [  0,   0, 142],
                [  0,   0,  70],
                [  0,  60, 100],
                [  0,  80, 100],
                [  0,   0, 230],
                [119,  11,  32]]
label_colours = dict(zip(range(19), valid_colors))

class DarkZurichLoader_Pseudo(data.Dataset):
	def __init__(self, root, img_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None, return_name = False):
		self.n_classes = 19
		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.transform = transform
		self.return_name = return_name
		#self.use_pseudo = use_pseudo
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
		#self.lbl_ids = [i_id.strip() for i_id in open(lbl_list_path)]
		#if self.use_pseudo:
			#self.pseudo_lbl_ids = [i_id.strip() for i_id in open(img_list_path)]

		if not max_iters==None:
		   self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		   #self.lbl_ids = self.lbl_ids * int(np.ceil(float(max_iters) / len(self.lbl_ids)))
		   #if self.use_pseudo:
			   #self.pseudo_lbl_ids = self.pseudo_lbl_ids * int(np.ceil(float(max_iters) / len(self.pseudo_lbl_ids)))

		self.files = []
		self.id_to_trainid = {7: 0, 8 : 1, 11: 2, 12: 3, 13: 4 , 17: 5,
				     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
				     26: 13,27:14, 28:15, 31:16, 32: 17, 33: 18}
		#self.set = set
		# for split in ["train", "trainval", "val"]:
		'''
		for img_name, lbl_name in zip(self.img_ids, self.lbl_ids):
			img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, img_name))
			lbl_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbl_name))
			if self.use_pseudo:
				pseudo_name = img_name.split('/')[-1]
				pseudo_lbl_file = osp.join(self.root, "pseudo_train/%s" % (pseudo_name))
				self.files.append({
					"img": img_file,
					"label": lbl_file,
					"pseudo_label": pseudo_lbl_file,
					"name": img_name
				})
			else:
				self.files.append({
					"img": img_file,
					"label": lbl_file,
					"name": img_name
				})
		'''
		for pair in self.img_ids:
			night, day = pair.split(",")
			img_night = osp.join(self.root, "Dark_Zurich_train_anon/rgb_anon/%s" % (night) + "_rgb_anon.png")
			img_day = osp.join(self.root, "Dark_Zurich_train_anon/rgb_anon/%s" % (day) + "_rgb_anon.png")
			night_pseudo = night.split("/")[-1]
			day_pseudo = day.split("/")[-1]
			pseudo_lbl_file_night = osp.join(self.root, "Dark_Zurich_train_anon/rgb_anon/pseudo_train_night_grad_soft_warm_30/%s" % (night_pseudo) + "_rgb_anon.png")
			pseudo_lbl_file_day = osp.join(self.root, "Dark_Zurich_train_anon/rgb_anon/pseudo_train_day_grad_soft_warm_30/%s" % (day_pseudo) + "_rgb_anon.png")
			self.files.append({
				"img_night": img_night,
				"img_day": img_day,
				"pseudo_label_night": pseudo_lbl_file_night,
				"pseudo_label_day": pseudo_lbl_file_day,
				"name_night": night + "_rgb_anon.png",
				"name_day": day + "_rgb_anon.png"
			})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image_n = Image.open(datafiles["img_night"]).convert('RGB')
		image_d = Image.open(datafiles["img_day"]).convert('RGB')

		pseudo_label_n = Image.open(datafiles["pseudo_label_night"])
		pseudo_label_d = Image.open(datafiles["pseudo_label_day"])
		name_n = datafiles["name_night"]
		name_d = datafiles["name_day"]

		# resize
		if self.crop_size != None:
			image_n = image_n.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
			image_d = image_d.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
			pseudo_label_n = pseudo_label_n.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
			pseudo_label_d = pseudo_label_d.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)

		# transform
		if self.transform != None:
			image_n, image_d, pseudo_label_n, pseudo_label_d = self.transform(image_n, image_d, pseudo_label_n, pseudo_label_d)

		image_n = np.asarray(image_n, np.float32)
		size = image_n.shape
		image_n = image_n[:, :, ::-1]  # change to BGR
		image_n -= self.mean
		image_n = image_n.transpose((2, 0, 1)) / 128.0

		image_d = np.asarray(image_d, np.float32)
		size = image_d.shape
		image_d = image_d[:, :, ::-1]  # change to BGR
		image_d -= self.mean
		image_d = image_d.transpose((2, 0, 1)) / 128.0


		pseudo_label_n = np.asarray(pseudo_label_n, np.compat.long)
		pseudo_label_d = np.asarray(pseudo_label_d, np.compat.long)

		# re-assign labels to match the format of Cityscapes
		pseudo_label_n_copy = 255 * np.ones(pseudo_label_n.shape, dtype=np.compat.long)
		for v in range(self.n_classes):
			pseudo_label_n_copy[pseudo_label_n == v] = v

		pseudo_label_d_copy = 255 * np.ones(pseudo_label_d.shape, dtype=np.compat.long)
		for v in range(self.n_classes):
			pseudo_label_d_copy[pseudo_label_d == v] = v

		if not self.return_name:
			return image_n.copy(), image_d.copy(), pseudo_label_n_copy.copy(), pseudo_label_d_copy.copy()
		else:
			return image_n.copy(), image_d.copy(), pseudo_label_n_copy.copy(), pseudo_label_d_copy.copy(), name_n, name_d

	def decode_segmap(self, img):
		map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
		for idx in range(img.shape[0]):
			temp = img[idx, :, :]
			r = temp.copy()
			g = temp.copy()
			b = temp.copy()
			for l in range(0, self.n_classes):
				r[temp == l] = label_colours[l][0]
				g[temp == l] = label_colours[l][1]
				b[temp == l] = label_colours[l][2]
	
			rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
			rgb[:, :, 0] = r / 255.0
			rgb[:, :, 1] = g / 255.0
			rgb[:, :, 2] = b / 255.0
			map[idx, :, :, :] = rgb
		return map

if __name__ == '__main__':
	dst = DarkZurichLoader_Pseudo("./data")
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs1, imgs2, labels1, labels2 = data
		if i == 0:
			imgs1 = torchvision.utils.make_grid(imgs1).numpy()
			imgs1 = np.transpose(imgs1, (1, 2, 0))
			imgs1 = imgs1[:, :, ::-1]
			plt.imshow(imgs1)
			plt.show()
