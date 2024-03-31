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

class DarkZurichTestLoader(data.Dataset):
	def __init__(self, root, img_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), return_name = False):
		self.n_classes = 19
		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.return_name = return_name
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]

		if not max_iters==None:
		   self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

		self.files = []
		self.id_to_trainid = {7: 0, 8 : 1, 11: 2, 12: 3, 13: 4 , 17: 5,
					 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
					 26: 13,27:14, 28:15, 31:16, 32: 17, 33: 18}
		self.trainid_to_id = {0: 7, 1 : 8, 2: 11, 3: 12, 4: 13 , 5: 17,
					 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25,
					 13: 26, 14: 27, 15:28, 16:31, 17: 32, 18: 33}
		#self.set = set
		for night in self.img_ids:
			img_night = osp.join(self.root, "Dark_Zurich_test_anon/rgb_anon/%s" % (night))
			#print("img_night_path",img_night)
			#print('self.root:',self.root)
			self.files.append({
				"img_night": img_night,
				"name_night": night, #"name_night": night + "_rgb_anon.png",
			})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]
		image_n = Image.open(datafiles["img_night"]).convert('RGB')
		name_night = datafiles["name_night"]

		# resize
		if self.crop_size != None:
			image_n = image_n.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)

		image_n = np.asarray(image_n, np.float32)
		size = image_n.shape
		image_n = image_n[:, :, ::-1]  # change to BGR
		image_n -= self.mean
		image_n = image_n.transpose((2, 0, 1)) / 128.0


		if not self.return_name:
			return image_n.copy()
		else:
			return image_n.copy(), name_night

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
	def convert_back_to_id(self, pred):
		coverted_arr = np.zeros(pred.shape)
		for train_id in self.trainid_to_id.keys():
			coverted_arr[pred==train_id] = self.trainid_to_id[train_id]
		return coverted_arr
			

if __name__ == '__main__':
	dst = DarkZurichTestLoader("./data/DarkZurich")
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs = data
		if i == 0:
			imgs = torchvision.utils.make_grid(imgs).numpy()
			imgs = np.transpose(imgs, (1, 2, 0))
			imgs = imgs[:, :, ::-1]
			plt.imshow(imgs)
			plt.show()
