import os
import torch
import json
import numpy as np
from collections import defaultdict
import cv2
import csv


class DyMLDataset:
    def __init__(self, dataset_path, K=4, transform=None):
        """
        Args:
            dataset_path (str): path to the DyML dataset
            K (int, optional): [description]. Defaults to 4.
            transform ([type], optional): [description]. Defaults to None.
        """
        self.dataset_path = os.path.join(dataset_path, 'train')
        self.K = K
        self.transform = transform
        self.load_config_file()
        # self.load()

    def load_config_file(self):
        with open(os.path.join(self.dataset_path, 'label.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        self.datas = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        self.img_for_class = {'coarse': defaultdict(list),
                              'middle': defaultdict(list),
                              'fine': defaultdict(list), }
        for i, d in enumerate(self.datas):
            if len(d) == 4:
                fname, coarse, middle, fine = d
            else:
                break
            self.img_for_class['coarse'][coarse].append(i)
            self.img_for_class['middle'][middle].append(i)
            self.img_for_class['fine'][fine].append(i)

        for k in ['coarse', 'middle', 'fine']:
            self.img_for_class[k] = list(self.img_for_class[k].values())


        

    def __getitem__(self, index):
        # from each level sample K images
        samples = []
        for k in ['fine', 'middle', 'coarse']:
            metas = self.img_for_class[k][index % len(self.img_for_class[k])]
            samples.extend([
                metas[x]
                for x in np.random.choice(
                    np.arange(len(metas)), self.K, replace=(len(metas) < self.K)
                )
            ])

        # for id_map, label2id in zip([self.fine_id_map, self.mid_id_map, self.coarse_id_map], [self.fine_label2id, self.mid_label2id, self.coarse_label2id]):
        #     metas = id_map[label2id[index % len(id_map)]]
        #     samples.extend([
        #         metas[x]
        #         for x in np.random.choice(
        #             np.arange(len(metas)), self.K, replace=(len(metas) < self.K)
        #         )
        #     ])
        datas = [self.datas[idx] for idx in samples]
        # 数据的rgb通道反了，读取的时候变一下[..., ::-1]
        imgs = [cv2.imread(os.path.join(self.dataset_path, 'imgs', data[0])) for data in datas]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
            imgs = np.stack(imgs)
        fine_labels = [data[3] for data in datas]
        mid_labels = [data[2] for data in datas]
        coarse_labels = [data[1] for data in datas]
        return {
            'data': imgs,
            'fine_g_label': np.array(fine_labels, dtype='int32'),
            'middle_g_label': np.array(mid_labels, dtype='int32'),
            'coarse_g_label': np.array(coarse_labels, dtype='int32'),
        }
    

    def __len__(self):
        return len(self.datas) // self.K



def test_Dyml_dataset():
    # ds = DyMLDataset('../data/DyML-animal')
    # ds = DyMLDataset('/data/dyml_animal')
    ds = DyMLDataset('/data/dyml_product')
    # ds = DyMLDataset('/data/dyml_vehicle')
    print(len(ds))
    sample = ds[1]
    imgs = sample['data']
    cat_imgs = []
    for i in range(0, len(imgs) // ds.K):
        tmp = imgs[(i * ds.K):(i + 1) * ds.K]
        tmp = [cv2.resize(itm, (224, 224)) for itm in tmp]
        tmp1 = np.concatenate(tmp, axis=1)
        cat_imgs.append(tmp1)
    cat_img = np.concatenate(cat_imgs, axis=0)
    cv2.imwrite('test_cat.png', cat_img)


if __name__ == "__main__":
    test_Dyml_dataset()




