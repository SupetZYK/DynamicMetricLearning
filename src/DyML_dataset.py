import os
import torch
import json
import numpy as np
from collections import defaultdict
import cv2


class DyMLDataset:
    def __init__(self, dataset_path, K=4, transform=None):
        """
        Args:
            dataset_path (str): path to the DyML dataset
            K (int, optional): [description]. Defaults to 4.
            transform ([type], optional): [description]. Defaults to None.
        """
        self.dataset_path = dataset_path
        self.K = K
        self.transform = transform
        self.load()

    def load(self):
        # re-arange data
        self.datas = json.load(open(os.path.join(self.dataset_path, 'protocols', 'train.json')))
        self.fine_id_map = defaultdict(list)
        self.mid_id_map = defaultdict(list)
        self.coarse_id_map = defaultdict(list)

        for idx, data in enumerate(self.datas):
            # import ipdb;ipdb.set_trace()
            fine_id = data['fine_g_label']
            mid_id = data['middle_g_label']
            coarse_id = data['coarse_g_label']
            self.fine_id_map[fine_id].append(idx)
            self.mid_id_map[mid_id].append(idx)
            self.coarse_id_map[coarse_id].append(idx)

        # generate bi-directional map between id and label
        self.fine_id2label = dict()
        self.mid_id2label = dict()
        self.coarse_id2label = dict()
        self.fine_label2id = dict()
        self.mid_label2id = dict()
        self.coarse_label2id = dict()

        for idx, fid in enumerate(self.fine_id_map):
            self.fine_id2label[fid] = idx
            self.fine_label2id[idx] = fid
        
        for idx, mid in enumerate(self.mid_id_map):
            self.mid_id2label[mid] = idx
            self.mid_label2id[idx] = mid
        
        for idx, cid in enumerate(self.coarse_id_map):
            self.coarse_id2label[cid] = idx
            self.coarse_label2id[idx] = cid
        

    def __getitem__(self, index):
        # from each level sample K images
        samples = []
        for id_map, label2id in zip([self.fine_id_map, self.mid_id_map, self.coarse_id_map], [self.fine_label2id, self.mid_label2id, self.coarse_label2id]):
            metas = id_map[label2id[index % len(id_map)]]
            samples.extend([
                metas[x]
                for x in np.random.choice(
                    np.arange(len(metas)), self.K, replace=(len(metas) < self.K)
                )
            ])
        datas = [self.datas[idx] for idx in samples]
        imgs = [cv2.imread(os.path.join(self.dataset_path, 'imgs', data['path'])) for data in datas]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
            imgs = np.stack(imgs)
        fine_labels = [self.fine_id2label[data['fine_g_label']] for data in datas]
        mid_labels = [self.mid_id2label[data['middle_g_label']] for data in datas]
        coarse_labels = [self.coarse_id2label[data['coarse_g_label']] for data in datas]
        return {
            'data': imgs,
            'fine_g_label': np.array(fine_labels, dtype='int32'),
            'middle_g_label': np.array(fine_labels, dtype='int32'),
            'coarse_g_label': np.array(fine_labels, dtype='int32'),
        }
    

    def __len__(self):
        return len(self.datas) // self.K



def test_Dyml_dataset():
    ds = DyMLDataset('../data/DyML-animal')
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



