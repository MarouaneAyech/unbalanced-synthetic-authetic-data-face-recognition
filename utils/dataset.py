import collections
import copy
import numbers
import os
import pickle
import queue as Queue
import threading
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from utils.augmentation import aug_rand_4_16



class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


# class MXFaceDataset(Dataset):
#     def __init__(self, root_dir, local_rank):
#         super(MXFaceDataset, self).__init__()
#         self.transform = transforms.Compose(
#             [transforms.ToPILImage(),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#              ])
#         self.root_dir = root_dir
#         self.local_rank = local_rank
#         path_imgrec = os.path.join(root_dir, 'train.rec')
#         path_imgidx = os.path.join(root_dir, 'train.idx')
#         self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#         s = self.imgrec.read_idx(0)
#         header, _ = mx.recordio.unpack(s)
#         if header.flag > 0:
#             self.header0 = (int(header.label[0]), int(header.label[1]))
#             self.imgidx = np.array(range(1, int(header.label[0])))
#         else:
#             self.imgidx = np.array(list(self.imgrec.keys))
class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank

        # Remplace mx.recordio par lecture directe des dossiers
        self.imgidx = []
        self.labels = []
        lb = 0
        list_dir = sorted(os.listdir(root_dir))
        for folder in list_dir:
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.imgidx.append(os.path.join(folder_path, img_file))
                        self.labels.append(lb)
                lb += 1

        self.imgidx = np.array(self.imgidx)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        img_path = self.imgidx[index]
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        # Remplace mx.image.imdecode par cv2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(img)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank, root2=None, synth_ids=10000, auth_ids=10000, shuffle=False, randaug=False):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_aug = transforms.Compose(aug_rand_4_16)
        self.randaug = randaug
        self.shuffle = shuffle
        self.root_dir = root_dir
        self.root_dir2 = root2
        self.local_rank = local_rank
        self.imgidx, self.labels, self.num_ids, self.is_synth, self.fld_name = self.scan(root_dir, root2, synth_ids=synth_ids, auth_ids=auth_ids)

    def scan(self, root_syn, root_auth, synth_ids, auth_ids):
        imgidex = []
        labels = []
        is_synth = []
        fld_name = []
        lb = -1
        indexes = None



        if root_syn is not None:
            list_dir = os.listdir(root_syn)

            if self.shuffle:
                random.shuffle(list_dir)
            else:
                list_dir.sort()


            for l in list_dir[:synth_ids]:
                images = os.listdir(os.path.join(root_syn, l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l, img))
                    labels.append(lb)
                    fld_name.append(l)
                    is_synth.append(True)


        syn = lb
        if root_auth is not None:
            list_dir2 = os.listdir(root_auth)

            if self.shuffle:
                random.shuffle(list_dir2)
            else:
                list_dir2.sort()

            authentics = list_dir2[:auth_ids]
            for l in authentics:
                images = os.listdir(os.path.join(root_auth, l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l, img))
                    labels.append(lb)
                    fld_name.append(l)
                    is_synth.append(False)

        return imgidex, labels, [lb, lb - syn], is_synth, fld_name

    def readImage(self, path, issyn):
        rt = self.root_dir if issyn else self.root_dir2
        return cv2.imread(os.path.join(rt, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        fld_name = self.fld_name[index]
        lnt = len(fld_name)
        if lnt < 7:
            pad = "_" * (7 - lnt)
            fld_name += pad
        fld_name = torch.tensor([ord(c) for c in fld_name], dtype=torch.int32)
        img = self.readImage(path, self.is_synth[index])
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = self.transform(sample) if not self.randaug else self.transform_aug(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class TestDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank, root2=None, auth_ids=1000):
        super(TestDatasetFolder, self).__init__()
        self.root_dir = root_dir
        self.root_dir2 = root2
        self.local_rank = local_rank
        self.imgidx, self.labels, self.num_ids, self.is_synth = self.scan(root_dir, root2, auth_ids=auth_ids)

    def scan(self, root_syn, root_auth, auth_ids):
        imgidex = []
        labels = []
        is_synth = []
        lb = -1
        list_dir = os.listdir(root_syn)
        list_dir.sort()
        for l in list_dir:
            images = os.listdir(os.path.join(root_syn, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(True)
        list_dir2 = os.listdir(root_auth)
        list_dir2.sort()
        syn = lb

        for l in list_dir2:
            images = os.listdir(os.path.join(root_auth, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(False)

        return imgidex, labels, [lb, lb - syn], is_synth

    def readImage(self, path, issyn):
        if issyn:
            return cv2.imread(os.path.join(self.root_dir, path))
        return cv2.imread(os.path.join(self.root_dir2, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.readImage(path, self.is_synth[index])
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #sample = self.transform_aug(sample)
        sample = torch.from_numpy(np.transpose(sample, axes=(2, 0, 1)))
        sample = ((sample / 255) - 0.5) / 0.5

        return index, sample, label

    def __len__(self):
        return len(self.imgidx)




