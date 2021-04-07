import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
from megskull.utils.meta import cached_property
from collections import defaultdict
from IPython import embed

from video3.datasets.base import BaseDataset
from ._config import ONESHOTMATCHING
import os

class TrainDataloader(BaseDataset):
    def __init__(self, datasets, transform=None):
        """
        :param datasets:
        :type datasets: list
        :param transform: data augment transformation, for example, resize, crop and so on
        :type transform (callable, optional): Optional transform to be applied on a sample. (default: ``None``)
        """
        assert len(datasets) > 0
        self.datasets = datasets
        self.range = 50
        self.pos_ratio = 0.5
        self.transform = transform if transform else lambda x: x
        self.fetcher = nori.Fetcher()

    @cached_property
    def metas(self):
        metas = []
        for x in self.datasets:
            datas = io.load(ONESHOTMATCHING[x])
            metas += [datas[k] for k in datas.keys()]
            print("---------------------", len(metas))
        return metas

    def __getitem__(self, index):
        meta = self.metas[index]
        # get gallery
        id_g = np.random.choice(range(len(meta)))
        g_nori_id = meta[id_g]['nori_id']
        g_bbox = meta[id_g].get('bbox', None)
        # get query
        probablity = np.random.random()
        if probablity < self.pos_ratio:
            # get pos sample
            label = 1
            min_range = max(0, id_g - self.range)
            max_range = min(len(meta), id_g + self.range)
            idx = np.random.choice(range(min_range, max_range))
            q_nori_id = meta[idx]['nori_id']
            q_bbox = meta[idx].get('bbox', None)
        else:
            # get neg sample
            label = 0
            q_nori_id, q_bbox = self.get_neg_sample(index)

        g_img = imdecode(self.fetcher.get(g_nori_id))[..., :3]
        if g_bbox is not None:
            x, y, w, h = g_bbox[:]
            g_img = g_img[y:y + h, x:x + w, :]
        q_img = imdecode(self.fetcher.get(q_nori_id))[..., :3]
        if q_bbox is not None:
            x, y, w, h = q_bbox[:]
            q_img = q_img[y:y + h, x:x + w, :]
        if self.transform is not None:
            g_img = self.transform(image=g_img)['image']
            q_img = self.transform(image=q_img)['image']
        return {'gallery': g_img.astype('uint8'), 'query': q_img.astype('uint8'), 'label': label}

    def get_neg_sample(self, index):
        index = np.random.choice(list(range(index)) + list(range(index + 1, len(self.metas))))
        meta = self.metas[index]
        frames = len(meta)
        idx = np.random.choice(range(frames))
        nori_id = meta[idx]['nori_id']
        bbox = meta[idx].get('bbox', None)
        return nori_id, bbox

    def __len__(self):
        return len(self.metas)

class TrainDataloaderDatadiv(BaseDataset):
    def __init__(self, datasets, K, transform=None):
        """
        :param datasets:
        :type datasets: list
        :param transform: data augment transformation, for example, resize, crop and so on
        :type transform (callable, optional): Optional transform to be applied on a sample. (default: ``None``)
        """
        assert len(datasets) > 0
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.K = K if isinstance(K, list) else [K] * len(datasets)
        self.range = 50
        self.pos_ratio = 0.5
        self.transform = transform if transform else lambda x: x
        self.fetcher = nori.Fetcher()

    @cached_property
    def metas(self):
        metas = []
        for x in self.datasets:
            datas = io.load(ONESHOTMATCHING[x])
            datas = [datas[k] for k in datas.keys()]
            metas.append(datas)
        return metas

    def __getitem__(self, idx):
        gallerys = []
        querys = []
        labels = []
        for d_id in range(len(self.datasets)):
            gs, qs, ls = self.get_sample(d_id, idx)
            gallerys.extend(gs)
            querys.extend(qs)
            labels.extend(ls)
        gallerys = np.stack(gallerys)
        querys = np.stack(querys)
        labels = np.stack(labels)
        print(idx, gallerys.shape, querys.shape, labels.shape)
        return {'gallery': gallerys.astype('uint8'), 'query': querys.astype('uint8'), 'label': labels.astype('int32')}

    def get_sample(self, dataset_id, global_sample_id):
        sample_num = self.K[dataset_id]
        assert sample_num > 0, 'sample_num need to be positive.'
        gallerys = []
        querys = []
        labels = []
        metas = self.metas[dataset_id]
        global_sample_id = global_sample_id % len(metas)
        for _ in range(sample_num):
            # get gallery
            meta = metas[global_sample_id]
            id_g = np.random.choice(range(len(meta)))
            g_nori_id = meta[id_g]['nori_id']
            g_bbox = meta[id_g].get('bbox', None)

            # get query
            probablity = np.random.random()
            if probablity < self.pos_ratio:
                # get pos sample
                label = 1
                min_range = max(0, id_g - self.range)
                max_range = min(len(meta), id_g + self.range)
                idx = np.random.choice(range(min_range, max_range))
                q_nori_id = meta[idx]['nori_id']
                q_bbox = meta[idx].get('bbox', None)
            else:
                # get neg sample from different meta
                label = 0
                q_nori_id, q_bbox = self.get_neg_sample(metas, global_sample_id)

            g_img = imdecode(self.fetcher.get(g_nori_id))[..., :3]
            q_img = imdecode(self.fetcher.get(q_nori_id))[..., :3]
            if g_bbox is not None:
                x, y, w, h = g_bbox[:]
                g_img = g_img[y:y + h, x:x + w, :]
            if q_bbox is not None:
                x, y, w, h = q_bbox[:]
                q_img = q_img[y:y + h, x:x + w, :]
            if self.transform is not None:
                g_img = self.transform(image=g_img)['image']
                q_img = self.transform(image=q_img)['image']
            gallerys.append(g_img)
            querys.append(q_img)
            labels.append(label)
        return gallerys, querys, labels

    def get_neg_sample(self, metas, index):
        index = np.random.choice(list(range(index)) + list(range(index + 1, len(metas))))
        meta = metas[index]
        frames = len(meta)
        idx = np.random.choice(range(frames))
        nori_id = meta[idx]['nori_id']
        bbox = meta[idx].get('bbox', None)
        return nori_id, bbox

    def __len__(self):
        return max([len(x) for x in self.metas])


class TestDataloader(BaseDataset):
    def __init__(self, dataset=None, transform=None):
        """
        :param dataset: test dataset's name
        :type dataset: str
        :param transform: data augment transformation, for example, resize, crop and so on
        :type transform (callable, optional): Optional transform to be applied on a sample. (default: ``None``)
        """
        self.dataset = dataset
        self.transform = transform
        self.fetcher = nori.Fetcher()

    @cached_property
    def metas(self):
        metas = io.load(ONESHOTMATCHING[self.dataset])
        if self.dataset == 'lasot_test_xifen':
            return metas['query'] + metas['gallery']
        return metas

    def get_template(self):
        # assert self.dataset in ['taxi', 'dache']
        querys = self.metas['query']
        templates_ids = self.metas['templates']
        templates_ = []
        for x in templates_ids:
            img = imdecode(self.fetcher.get(querys[x]['nori_id']))[..., :3]
            if self.transform is not None:
                img = self.transform(image=img)['image']
            templates_.append(img[np.newaxis, :, :, :])
        return np.concatenate(templates_, axis=0)

    def get_gt_matrix(self):
        assert self.dataset == 'lasot_test_xifen'
        # following 2772 is the number of querys
        query_ids = [x['video_id'] for x in self.metas[:2772]]
        gallery_ids = [x['video_id'] for x in self.metas[2772:]]
        gt = np.zeros((len(query_ids), len(gallery_ids)), np.int32)
        for i in range(len(query_ids)):
            for j in range(len(gallery_ids)):
                gt[i, j] = int(query_ids[i] == gallery_ids[j])
        return gt

    def __getitem__(self, index):
        if self.dataset in ['got10k_test', 'lasot_test', 'got10k_test_neg30', 'high', 'middle']:
            meta = self.metas[index]
            if self.dataset in ['high', 'middle']:
                g_nori_id = meta['gallery']
                g_bbox = None
                q_nori_id = meta['query']
                q_bbox = None
            else:
                g_nori_id, g_bbox = meta['gallery']
                q_nori_id, q_bbox = meta['query']

            label = meta['label']
            g_img = imdecode(self.fetcher.get(g_nori_id))[..., :3]
            if g_bbox is not None:
                x, y, w, h = g_bbox[:]
                g_img = g_img[y:y + h, x:x + w, :]
            q_img = imdecode(self.fetcher.get(q_nori_id))[..., :3]
            if q_bbox is not None:
                x, y, w, h = q_bbox[:]
                q_img = q_img[y:y + h, x:x + w, :]

        elif self.dataset in ['taxi', 'dache'] or self.dataset.startswith('low_g'):
            meta = self.metas['gallery'][index]
            g_img = imdecode(self.fetcher.get(meta['nori_id']))[..., :3]
            q_img = None
            label = meta['is_%s' % self.dataset]

        elif self.dataset == 'lasot_test_xifen':
            meta = self.metas[index]
            g_img = imdecode(self.fetcher.get(meta['nori_id']))[..., :3]
            q_img = None
            label = None

        if self.transform is not None:
            g_img = self.transform(image=g_img)['image']
            if q_img is not None:
                q_img = self.transform(image=q_img)['image']

        if label is None:
            return {'gallery': g_img}
        elif q_img is None:
            return {'gallery': g_img, 'label': label}
        else:
            return {'gallery': g_img, 'query': q_img, 'label': label}

    def __len__(self):
        if self.dataset in ['taxi', 'dache'] or self.dataset.startswith('low_g'):
            return len(self.metas['gallery'])
        return len(self.metas)


#class DVR_TestDataloader(BaseDataset):
#    def __init__(self, dataset=None, transform=None):
#        """
#        :param dataset: test dataset's name
#        :type dataset: str
#        :param transform: data augment transformation, for example, resize, crop and so on
#        :type transform (callable, optional): Optional transform to be applied on a sample. (default: ``None``)
#        """
#        assert dataset in [ 'middle_g', 'fine_g'] or dataset.startswith('coarse_g')
#        assert ONESHOTMATCHING[dataset]['select_n_query'] <= ONESHOTMATCHING[dataset]['all_query_num']
#        self.dataset = dataset
#        self.transform = transform
#        self.fetcher = nori.Fetcher()
#        self.templates_ids=[]
#        self.choosed_nums = ONESHOTMATCHING[self.dataset]['select_n_query']
#        metas = self.metas()
#        if 'middle' in self.dataset or 'coarse' in self.dataset:
#            self.querys = metas['query']   # 对于粗粒度的话，querylabel就只有一个.
#            if isinstance(self.querys, dict):
#                self.query_labels = list(self.querys.keys())
#                #self.querys = [self.querys[k] for k in self.query_labels]   # 这个是具体每一个id的所有的querys_set
#                #self.query_set_num = len(self.querys)
#            self.gallerys = metas['gallery']
#
#    def metas(self):
#        metas = io.load(ONESHOTMATCHING[self.dataset]['path'])
#        return metas
#
#    def get_query(self, idx, middle_id=None):
#        #assert self.dataset.startswith('coarse_g')
#        if middle_id is not None:
#            querys = self.querys[middle_id][idx]
#        else:
#            querys = self.querys[idx]
#
#        choosed_ids = np.random.choice(range(len(querys)), self.choosed_nums, replace=len(querys) < self.choosed_nums)
#        query_imgs = []
#        for i in choosed_ids:
#            item = querys[i]
#            img = imdecode(self.fetcher.get(item['nori_id']))[..., :3]
#            if self.transform is not None:
#                img = self.transform(image=img)['image']
#            query_imgs.append(img[np.newaxis, :, :, :])
#        return np.concatenate(query_imgs, axis=0).astype(np.uint8), self.query_labels[idx]
#
#    def __getitem__(self, index):
#        #if self.dataset.startswith('coarse_g'):
#        #    meta = self.metas['gallery'][index]
#        #    g_img = imdecode(self.fetcher.get(meta['nori_id']))[..., :3]
#        #    q_img = None
#        #    label = meta['is_%s' % self.dataset]
#
#        #elif self.dataset in ['fine_g']:
#        #    meta = self.metas[index]
#        #    g_img = imdecode(self.fetcher.get(meta['gallery']))[..., :3]
#        #    q_img = imdecode(self.fetcher.get(meta['query']))[..., :3]
#        #    label = meta['label']
#
#        #elif self.dataset=='middle_g':
#        if self.dataset=='middle_g':
#            meta=self.gallerys[index]
#            middle_id=meta['query_middle_id']
#            g_img = imdecode(self.fetcher.get(meta['gallery']))[..., :3]
#            g_img = self.transform(image=g_img)['image']
#            return {'gallery': g_img, 'label': middle_id}         # label是middle_id
#
#        #if self.transform is not None:
#        #    g_img = self.transform(image=g_img)['image']
#        #    if q_img is not None:
#        #        q_img = self.transform(image=q_img)['image']
#
#        #if label is None:
#        #    return {'gallery': g_img}
#        #elif q_img is None:
#        #    return {'gallery': g_img, 'label': label}
#        #else:
#        #    return {'gallery': g_img, 'query': q_img, 'label': label}
#
#    def __len__(self):
#        #if self.dataset in ['taxi', 'dache','middle_g'] or self.dataset.startswith('coarse_g'):
#        #    return len(self.metas['gallery'])
#        #return len(self.metas)
#        return len(self.gallerys)

class DVR_TestDataloader(BaseDataset):
    def __init__(self, dataset=None, transform=None):
        """
        :param dataset: test dataset's name
        :type dataset: str
        :param transform: data augment transformation, for example, resize, crop and so on
        :type transform (callable, optional): Optional transform to be applied on a sample. (default: ``None``)
        """
        assert dataset in [ 'middle_g', 'fine_g'] or dataset.startswith('coarse_g')
        assert ONESHOTMATCHING[dataset]['select_n_query']<=ONESHOTMATCHING[dataset]['all_query_num']
        self.dataset = dataset
        self.transform = transform
        self.fetcher = nori.Fetcher()
        self.templates_ids=[]
        self.middle_query_pool={}


    @cached_property
    def metas(self):
        metas = io.load(ONESHOTMATCHING[self.dataset]['path'])
        return metas


    def get_query(self):
        assert self.dataset.startswith('coarse_g')
        querys = np.array(self.metas['query'])
        assert len(querys[0])>=ONESHOTMATCHING[self.dataset]['select_n_query']
        if len(querys[0])==ONESHOTMATCHING[self.dataset]['select_n_query']:
            pass
        else:
            querys_ids=np.random.randint(0,len(querys[0]),ONESHOTMATCHING[self.dataset]['select_n_query'])
            querys=querys[:,querys_ids]
        all_set_query = []
        for one_set in querys:
            querys_ = []
            for one_query in one_set:
                # print(one_query)
                # exit()
                # nori_id=one_query['nori_id'][0]
                img = imdecode(self.fetcher.get(one_query['nori_id'][0]))[..., :3]
                if self.transform is not None:
                    img = self.transform(image=img)['image']
                querys_.append(img[np.newaxis, :, :, :])
            all_set_query.append(np.concatenate(querys_, axis=0))
        return all_set_query

    def __getitem__(self, index):

        if self.dataset.startswith('coarse_g'):
            meta = self.metas['gallery'][index]
            g_img = imdecode(self.fetcher.get(meta['nori_id']))[..., :3]
            q_img = None
            label = meta['is_%s' % self.dataset]


        elif self.dataset in ['fine_g']:
            meta = self.metas[index]
            g_img = imdecode(self.fetcher.get(meta['gallery']))[..., :3]
            q_img = imdecode(self.fetcher.get(meta['query']))[..., :3]
            label = meta['label']
        elif self.dataset=='middle_g':
            meta=self.metas['gallery'][index]
            middle_id=meta['query_middle_id']  # 中粒度label

            label = meta['label']
            query_all=np.array(self.metas['query'][middle_id])  # 有50个, 每个里面有4个.
            g_img = imdecode(self.fetcher.get(meta['gallery']))[..., :3]
            g_img = self.transform(image=g_img)['image']
            all_set_query=[]

            #从多query中按照indx选query，将随机选出的idx按照middle id存下来，减少重复计算
            # if ONESHOTMATCHING[self.dataset]['all_query_num']==ONESHOTMATCHING[self.dataset]['select_n_query']:
            #     query_selected=query_all
            # else:
            if middle_id in self.middle_query_pool.keys():
                query_id=self.middle_query_pool[middle_id]['idx']
            else:
                self.middle_query_pool[middle_id]={}
                query_id=np.random.randint(0,len(query_all[0]),ONESHOTMATCHING[self.dataset]['select_n_query'])   # 看选择多少个query
                self.middle_query_pool[middle_id]['idx']=query_id
            query_selected=query_all[:,query_id]

            #从多query pool中读取query img，将选出的query img按照middle id存下来，减少重复计算
            if 'imgs' in list(self.middle_query_pool[middle_id].keys()):
                all_set_query=self.middle_query_pool[middle_id]['imgs']
            else:
                for one_set in query_selected:
                    querys = []
                    for one_query in one_set:
                        img = imdecode(self.fetcher.get(one_query['nori_id']))[..., :3]
                        if self.transform is not None:
                            img = self.transform(image=img)['image']
                        querys.append(img[np.newaxis, :, :, :])
                    all_set_query.append(np.concatenate(querys, axis=0))
                all_set_query=np.stack(all_set_query)
                self.middle_query_pool[middle_id]['imgs']=all_set_query

            return {'gallery': g_img, 'query': all_set_query, 'label': label, 'middle_id': middle_id}

        if self.transform is not None:
            g_img = self.transform(image=g_img)['image']
            if q_img is not None:
                q_img = self.transform(image=q_img)['image']

        if label is None:
            return {'gallery': g_img}
        elif q_img is None:
            return {'gallery': g_img, 'label': label}
        else:
            return {'gallery': g_img, 'query': q_img, 'label': label}

    def __len__(self):
        if self.dataset in ['taxi', 'dache','middle_g'] or self.dataset.startswith('coarse_g'):
            return len(self.metas['gallery'])
        return len(self.metas)


class DVR_animal(BaseDataset):
    def __init__(self, dataset, K=4, transform=None):
        self.dataset = dataset
        self.K = K
        self.fetcher = None
        self.transform = transform
        self.nr_class = len(self.metas[0])
        self.nr_class_mid = len(self.metas[1])
        self.nr_class_low = len(self.metas[2])

    @cached_property
    def metas(self):
        data = io.load(self.dataset)
        self.total_imgs = len(data)
        fine = defaultdict(list)
        mid = defaultdict(list)
        coarse = defaultdict(list)
        for item in data:
            fine[item["fine_g_label"]].append(item["nori_id"])
            mid[item["middle_g_label"]].append(item["nori_id"])
            coarse[item["coarse_g_label"]].append(item["nori_id"])

        fine = [fine[k] for k in sorted(fine.keys())]
        mid = [mid[k] for k in sorted(mid.keys())]
        coarse = [coarse[k] for k in sorted(coarse.keys())]
        return [fine, mid, coarse]

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        datas = []
        labels = []
        for i in range(3):
            metas = self.metas[i]
            meta = metas[idx % len(metas)]
            meta = [meta[x] for x in np.random.choice(range(len(meta)), self.K, replace=len(meta) < self.K)]
            datas += meta
            labels.append(([idx % len(metas)] * self.K))

        def get_img(nori_id):
            return imdecode(self.fetcher.get(nori_id))[..., :3]

        imgs = [get_img(x) for x in datas]
        if self.transform is not None:
            imgs = [self.transform(image=img)["image"] for img in imgs]
        imgs = np.stack(imgs)

        return {"data": imgs.astype("uint8"), "label": np.array(labels[0], dtype="int32"),
                "mid_g_label": np.array(labels[1], dtype="int32"),
                "low_g_label": np.array(labels[2], dtype="int32")}

    def __len__(self):
        return self.total_imgs // self.K
        # return 36 * 500


class DVR_product(BaseDataset):
    def __init__(self, dataset, K=4, transform=None):
        self.dataset = dataset
        self.K = K
        self.fetcher = None
        self.transform = transform
        self.nr_class = len(self.metas[0])
        self.nr_class_mid = len(self.metas[1])
        self.nr_class_low = len(self.metas[2])

    @cached_property
    def metas(self):
        data = io.load(self.dataset)
        self.total_imgs = len(data)
        fine = defaultdict(list)
        mid = defaultdict(list)
        coarse = defaultdict(list)
        for item in data:
            fine[item["fine_g_label"]].append(item["nori_id"])
            mid[item["middle_g_label"]].append(item["nori_id"])
            coarse[item["coarse_g_label"]].append(item["nori_id"])

        fine = [fine[k] for k in sorted(fine.keys())]
        mid = [mid[k] for k in sorted(mid.keys())]
        coarse = [coarse[k] for k in sorted(coarse.keys())]
        return [fine, mid, coarse]

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        datas = []
        labels = []
        for i in range(3):
            metas = self.metas[i]
            meta = metas[idx % len(metas)]
            meta = [meta[x] for x in np.random.choice(range(len(meta)), self.K, replace=len(meta) < self.K)]
            datas += meta
            labels.append(([idx % len(metas)] * self.K))

        def get_img(nori_id):
            return imdecode(self.fetcher.get(nori_id))[..., :3]

        imgs = [get_img(x) for x in datas]
        if self.transform is not None:
            imgs = [self.transform(image=img)["image"] for img in imgs]
        imgs = np.stack(imgs)

        return {"data": imgs.astype("uint8"), "label": np.array(labels[0], dtype="int32"),
                "mid_g_label": np.array(labels[1], dtype="int32"),
                "low_g_label": np.array(labels[2], dtype="int32")}

    def __len__(self):
        #return self.nr_class * 15   # 也可以按每个数据来供的方式，不然有些数据可能一直用不到。因为每个数据有其自己的三个label，再从同类的里面选K-1个就可以了.
        return self.total_imgs // self.K // 3  # 也可以按每个数据来供的方式，不然有些数据可能一直用不到。因为每个数据有其自己的三个label，再从同类的里面选K-1个就可以了.
    # 这样供数据的方式也有不好的地方，不能保证每一个数据都能被取到.


class DVR_all_in_one_Dataset(BaseDataset):
    def __init__(self, dataset_names=['DVR_animal_bmk_fine', 'DVR_animal_bmk_middle', 'DVR_animal_bmk_coarse'], dataset_root="s3://oneshot-datasets/datasets/animal_dataset", mode='query', transform=None):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names
        self.dataset_root = dataset_root
        self.fetcher = None
        self.mode = mode
        self.transform = transform

    @cached_property
    def metas(self):
        metas = []
        for ds_name in self.dataset_names:
            data= io.load(os.path.join(self.dataset_root, '{}.pkl'.format(ds_name)))
            datas = data[self.mode]
            for item in datas:
                # label = item["label"]
                imgs = item["img"] # is a list
                for img in imgs:
                    metas.append({"nori_id": img["nori_id"], "fine_g_label": img['fine_id'], "middle_g_label": img["middle_id"], "coarse_g_label": img["coarse_id"]})
        return metas

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        meta = self.metas[idx]
        nori_id = meta["nori_id"]
        label = meta["fine_g_label"]
        mid_label = meta["middle_g_label"]
        low_label = meta["coarse_g_label"]
        img = imdecode(self.fetcher.get(nori_id))[..., :3]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return {"data": img.astype("uint8"), "label": label, "mid_g_label": mid_label, "low_g_label": low_label}

    def __len__(self):
        return len(self.metas)

class DVR_product_Test(BaseDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        self.path = "s3://zhuyuke/datasets/IProduct/dvr_format/{}.pkl".format(dataset)

    @cached_property
    def metas(self):
        metas = io.load(self.path)
        return metas

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        meta = self.metas[idx]
        nori_id = meta["nori_id"]
        label = meta["fine_g_label"]
        mid_label = meta["middle_g_label"]
        low_label = meta["coarse_g_label"]
        img = imdecode(self.fetcher.get(nori_id))[..., :3]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return {"data": img.astype("uint8"), "label": label, "mid_g_label": mid_label, "low_g_label": low_label}

    def __len__(self):
        return len(self.metas)


class DVR_animal_Test(BaseDataset):
    def __init__(self, dataset, transform=None, mode="query"):
        assert dataset in ["fine", "middle", "coarse"]
        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        self.mode = mode
        self.path = "s3://oneshot-datasets/datasets/animal_dataset/DVR_animal_bmk_%s.pkl" % dataset

    @cached_property
    def metas(self):
        data = io.load(self.path)
        datas = data[self.mode]
        metas = []
        for item in datas:
            label = item["label"]
            imgs = item["img"] # is a list
            for img in imgs:
                metas.append({"nori_id": img["nori_id"], "label": label})
        return metas

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        meta = self.metas[idx]
        nori_id = meta["nori_id"]
        label = meta["label"]
        img = imdecode(self.fetcher.get(nori_id))[..., :3]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return {"data": img.astype("uint8"), "label": label}

    def __len__(self):
        return len(self.metas)


class DVR_vehicle_3pk(BaseDataset):
    def __init__(self, dataset, K=4, transform=None):
        self.dataset = dataset
        self.K = K
        self.fetcher = None
        self.transform = transform
        self.nr_class = len(self.metas[0])
        self.nr_class_mid = len(self.metas[1])
        self.nr_class_low = len(self.metas[2])

    @cached_property
    def metas(self):
        data = io.load(self.dataset)
        fine = defaultdict(list)
        mid = defaultdict(list)
        coarse = defaultdict(list)
        for item in data:
            fine[item["fine_g_label"]].append(item["nori_id"])
            mid[item["middle_g_label"]].append(item["nori_id"])
            coarse[item["coarse_g_label"]].append(item["nori_id"])

        fine = [fine[k] for k in sorted(fine.keys())]
        mid = [mid[k] for k in sorted(mid.keys())]
        coarse = [coarse[k] for k in sorted(coarse.keys())]
        return [fine, mid, coarse]

    def __getitem__(self, idx):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()

        datas = []
        labels = []
        for i in range(3):
            metas = self.metas[i]
            meta = metas[idx % len(metas)]
            meta = [meta[x] for x in np.random.choice(range(len(meta)), self.K, replace=len(meta) < self.K)]
            datas += meta
            labels.append(([idx % len(metas)] * self.K))

        def get_img(nori_id):
            return imdecode(self.fetcher.get(nori_id))[..., :3]

        imgs = [get_img(x) for x in datas]
        if self.transform is not None:
            imgs = [self.transform(image=img)["image"] for img in imgs]
        imgs = np.stack(imgs)

        return {"data": imgs.astype("uint8"), "label": np.array(labels[0], dtype="int32"),
                "mid_g_label": np.array(labels[1], dtype="int32"),
                "low_g_label": np.array(labels[2], dtype="int32")}

    def __len__(self):
        return self.nr_class


class VehicleDVRDataset(BaseDataset):
    def __init__(self, dataset, choosen_id=[i for i in range(6)], K=4, K_thresh=4, transform=None):
        r"""
        :param string datasets: dataset names for vehicle_reid
        :param int nr_class: number of human ids
        :param K: number of images per person. (default: ``4``)
        :type K: int, optional
        :param transform: optional transform to be applied on a sample. (default: ``None``)
        :type transform: callable, optional
        """

        self.dataset = dataset
        self.choosen_id = choosen_id
        self.K = K
        self.K_thresh = K_thresh
        self.transform = transform
        self.nr_class = len(self.metas)
        mid_label_space, low_label_space = set(), set()
        # self.mid_g_label = dict()
        # self.low_g_label = dict()
        for tmp in self.metas:
            mid_label_space.add(tmp[0]['middle_g_label'])
            low_label_space.add(tmp[0]['coarse_g_label'])

        self.mid_label_space, self.low_label_space = dict(), dict()
        for idx, label in enumerate(mid_label_space):
            self.mid_label_space[label] = idx
        for idx, label in enumerate(low_label_space):
            self.low_label_space[label] = idx

        self.nr_class_mid = len(self.mid_label_space)
        self.nr_class_low = len(self.low_label_space)
        self.mid_g_label, self.low_g_label = self.get_W_label()

        self.fetcher = None
        print("collecting %d identities" % self.nr_class)

    @cached_property
    def metas(self):
        data = io.load(self.dataset)
        self.total_imgs = len(data)
        items = defaultdict(list)
        for tmp in data:
            # print(tmp)
            if self.choosen_id is None or tmp['coarse_g_label'] in self.choosen_id:   # choose the specified ids within assigned low granularity range
                items[tmp['fine_g_label']].append(tmp)

        keys = sorted(items.keys())
        metas = [items[tmp] for tmp in keys if len(items[tmp]) >= self.K_thresh]

        return metas

    def get_W_label(self):
        mid_g_label = {idx: self.mid_label_space[tmp[0]['middle_g_label']] for idx, tmp in enumerate(self.metas)}
        low_g_label = {idx: self.low_label_space[tmp[0]['coarse_g_label']] for idx, tmp in enumerate(self.metas)}
        return mid_g_label, low_g_label

    def __getitem__(self, index):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()
        meta = self.metas[index % len(self.metas)]
        meta = [
            meta[x]
            for x in np.random.choice(
                np.arange(len(meta)), self.K, replace=(len(meta) < self.K)
            )
        ]

        def get_img(info):
            img = self.fetcher.get(info['nori_id'])
            img = imdecode(img)
            return img

        def get_mid_label(info):
            anotated_mid_label = info['middle_g_label']
            anotated_low_label = info['coarse_g_label']
            return self.mid_label_space[anotated_mid_label], self.low_label_space[anotated_low_label]

        mid_g_label, low_g_label = get_mid_label(meta[0])

        imgs = [get_img(x) for x in meta]
        if self.transform is not None:
            imgs = [self.transform(image=img)["image"] for img in imgs]
        imgs = np.stack(imgs)
        return {"data": imgs.astype("uint8"), "label": np.array([index % len(self.metas)], dtype="int32"),
                "mid_g_label": np.array([mid_g_label], dtype="int32"),
                "low_g_label": np.array([low_g_label], dtype="int32")}

    def __len__(self):
        # return self.total_imgs // self.K // 3
        return self.total_imgs // self.K
        # return self.nr_class
