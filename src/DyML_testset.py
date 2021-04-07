import os
import torch
import json
import numpy as np
from collections import defaultdict
import cv2
import csv


class DVR_product_Test():
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        # self.path = "s3://zhuyuke/datasets/IProduct/dvr_format/{}.pkl".format(dataset)
        self.dataset_path = os.path.join(dataset, 'mini-bmk_all_in_one')
        self.metas = []
        self.load_config_file()


    def load_config_file(self):
        with open(os.path.join(self.dataset_path, 'label.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        self.datas = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        # self.img_for_class = {'coarse': defaultdict(list),
        #                       'middle': defaultdict(list),
        #                       'fine': defaultdict(list), }
        for i, d in enumerate(self.datas):
            if len(d) == 4:
                # fname, coarse, middle, fine = d
                self.metas.append(d)
            else:
                break


    def __getitem__(self, idx):

        meta = self.metas[idx]
        img = cv2.imread(os.path.join(self.dataset_path, 'imgs', meta[0]))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return {"data": img.astype("uint8"), "label": meta[3], "mid_g_label": meta[2], "low_g_label": meta[1]}

    def __len__(self):
        return len(self.metas)


class DVR_animal_Test():
    def __init__(self, dataset, level, transform=None, mode="query"):
        assert level in ["fine", "middle", "coarse"]
        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        self.mode = mode
        # self.path = "s3://oneshot-datasets/datasets/animal_dataset/DVR_animal_bmk_%s.pkl" % dataset

        self.dataset_path = os.path.join(dataset, 'bmk_' + level)
        self.metas = []
        self.load_config_file()

    def load_config_file(self):
        with open(os.path.join(self.dataset_path, self.mode + '.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        self.datas = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        for i, d in enumerate(self.datas):
            if len(d) == 2:
                # fname, coarse, middle, fine = d
                self.metas.append(d)
            else:
                break

    def __getitem__(self, idx):

        meta = self.metas[idx]
        img = cv2.imread(os.path.join(self.dataset_path, self.mode, meta[0]))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return {"data": img.astype("uint8"), "label": meta[1]}

    def __len__(self):
        return len(self.metas)

def test_Dyml_dataset():
    # ds = DyMLDataset('../data/DyML-animal')
    ds = DVR_animal_Test('/data/dyml_animal', 'coarse')
    # ds = DVR_product_Test('/data/dyml_product')
    # ds = DyMLDataset('/data/dyml_vehicle')
    print(len(ds))
    sample = ds[10]
    imgs = sample['data']
    # cat_imgs = []
    # for i in range(0, len(imgs) // ds.K):
    #     tmp = imgs[(i * ds.K):(i + 1) * ds.K]
    #     tmp = [cv2.resize(itm, (224, 224)) for itm in tmp]
    #     tmp1 = np.concatenate(tmp, axis=1)
    #     cat_imgs.append(tmp1)
    # cat_img = np.concatenate(cat_imgs, axis=0)
    cv2.imwrite('test_p.png', imgs[..., ::-1])


if __name__ == "__main__":
    test_Dyml_dataset()

