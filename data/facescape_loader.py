from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os

# import h5py
# import subprocess
# import shlex
# import json
import random


def _get_data_files(list_filename: str):
    """Returns a list of lists with [id, gender_tag age]"""
    with open(list_filename) as f:
        return [line.rstrip().split(" ") for line in f]


def load_npy_data(dataset_path: str, mmap=None):
    if mmap == "r":
        dataset_numpy = np.load(dataset_path, mmap_mode=mmap, allow_pickle=True)
    else:
        dataset_numpy = np.load(dataset_path, allow_pickle=True)

    return dataset_numpy


def load_npy_data_feat(dataset_path: str, mmap=None):
    if mmap == "r":
        dataset_numpy = np.load(dataset_path, mmap_mode=mmap, allow_pickle=True)
    else:
        dataset_numpy = np.load(dataset_path, allow_pickle=True)

    points = dataset_numpy.item().get("points", None)
    texture = dataset_numpy.item().get("textures", None)

    if points != None and texture != None:
        return np.concatenate(
            (points.squeeze(0).cpu().numpy(), texture.squeeze(0).cpu().numpy()), axis=-1
        )
    else:
        return None


class FaceScape(data.Dataset):
    def __init__(
        self,
        num_points: int,
        transforms,
        base_directory,
        annotations="all_annotations1024_feat.npy",
        split=0.85,
        seed=123,
        train=True,
        contrastive=False,
    ):
        super().__init__()
        self.transforms = transforms
        self.train = train
        # to make sure the same data is sampled for each run
        random.seed(seed)

        # contrastive training for privacy
        self.contrastive = contrastive
        base_name = base_directory

        self.annotation_file = os.path.join(base_name, annotations)

        os.path.exists(self.annotation_file) == True
        # dict of annotations
        print("Loading from: ", self.annotation_file)
        self.annotation = np.load(self.annotation_file, allow_pickle=True).item()

        self.train_names = random.sample(
            list(self.annotation.keys()), int(len(self.annotation) * (split))
        )
        self.test_names = list(
            filter(lambda x: x not in self.train_names, list(self.annotation.keys()))
        )

        if annotations.find("feat") != -1:
            # loads point cloud data
            self.train_data = []
            ignore_train = []
            for i, name in enumerate(self.train_names):
                npy_data = load_npy_data_feat(os.path.join(base_name, name))
                if isinstance(npy_data, np.ndarray):
                    self.train_data.append(npy_data)
                else:
                    ignore_train.append(name)

            self.train_data = np.array(self.train_data)

            self.test_data = []
            ignore_test = []
            for i, name in enumerate(self.test_names):
                npy_data = load_npy_data_feat(os.path.join(base_name, name))
                if isinstance(npy_data, np.ndarray):
                    self.test_data.append(npy_data)
                else:
                    ignore_test.append(name)

            self.test_data = np.array(self.test_data)
        else:
            # loads point cloud data
            self.train_data = np.array(
                [
                    load_npy_data(os.path.join(base_name, name))
                    for name in self.train_names
                ]
            )
            self.test_data = np.array(
                [
                    load_npy_data(os.path.join(base_name, name))
                    for name in self.test_names
                ]
            )

            ignore_test, ignore_train = [], []

        assert len(self.train_names) == int(len(self.annotation.keys()) * (split))
        assert len(self.test_names) == len(self.annotation.keys()) - len(
            self.train_names
        )

        # labels: ignore name of expression
        self.test_labels = []
        for key in self.test_names:
            if key not in ignore_test:
                label_list = self.annotation[key][:-1]
                new_label_list = []
                for i, label in enumerate(label_list):
                    if i == 2:  # gender
                        label = int(label)
                        assert label in [0, 1]

                    else:  # else [0..]
                        label = int(label) - 1

                    assert label >= 0
                    new_label_list.append(label)

                self.test_labels.append(new_label_list)

        self.test_labels = np.array(self.test_labels)

        self.train_labels = []
        for key in self.train_names:
            if key not in ignore_train:
                label_list = self.annotation[key][:-1]
                new_label_list = []
                for i, label in enumerate(label_list):
                    if i == 2:  # gender
                        label = int(label)
                        assert label in [0, 1]
                    else:  # else [0..]
                        label = int(label) - 1

                    assert label >= 0
                    new_label_list.append(label)

                self.train_labels.append(new_label_list)

        self.train_labels = np.array(self.train_labels)
        self.set_num_points(num_points)
        self.num_points_ = num_points

        print("Number of Train Samples:", len(self.train_labels))
        print("Number of Test Samples:", len(self.test_labels))

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        if self.train:
            current_points = self.train_data[idx, pt_idxs].copy()
            label = torch.from_numpy(self.train_labels[idx]).type(torch.LongTensor)

            if self.contrastive:  # binary only support (gender)
                while True:
                    pos_idx = np.random.choice(len(self.train_labels))
                    neg_idx = np.random.choice(len(self.train_labels))
                    pos_label = self.train_labels[pos_idx][2]
                    neg_label = self.train_labels[neg_idx][2]
                    curr_label = self.train_labels[idx][2]
                    if (pos_label == curr_label) and (neg_label != curr_label):
                        break

                positive_label = self.train_labels[pos_idx]
                negative_label = self.train_labels[neg_idx]

                p_points = self.train_data[pos_idx, pt_idxs].copy()
                n_points = self.train_data[neg_idx, pt_idxs].copy()

        else:
            np.random.shuffle(pt_idxs)

            current_points = self.test_data[idx, pt_idxs].copy()
            label = torch.from_numpy(self.test_labels[idx]).type(torch.LongTensor)

            if self.contrastive:  # binary only support (gender)
                while True:
                    pos_idx = np.random.choice(len(self.test_labels))
                    neg_idx = np.random.choice(len(self.test_labels))
                    pos_label = self.test_labels[pos_idx][2]
                    neg_label = self.test_labels[neg_idx][2]
                    curr_label = self.test_labels[idx][2]
                    if (pos_label == curr_label) and (neg_label != curr_label):
                        break

                positive_label = self.test_labels[pos_idx]
                negative_label = self.test_labels[neg_idx]

                p_points = self.test_data[pos_idx, pt_idxs].copy()
                n_points = self.test_data[neg_idx, pt_idxs].copy()

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        if self.contrastive:
            p_label = torch.from_numpy(positive_label).type(torch.LongTensor)
            n_label = torch.from_numpy(negative_label).type(torch.LongTensor)
            if self.transforms is not None:
                p_points = self.transforms(p_points)
                n_points = self.transforms(n_points)
            return current_points, label, (p_points, p_label, n_points, n_label)

        return current_points, label, torch.tensor([])

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def set_num_points(self, pts):
        self.num_points = (
            min(self.train_data.shape[1], pts)
            if self.train
            else min(self.test_data.shape[1], pts)
        )
        return self.num_points


if __name__ == "__main__":
    from torchvision import transforms
    import sys
    import os

    BASE_NAME = "../../../../../Faces_Data/"

    try:
        import src.pctransforms as d_utils
    except:
        PACKAGE_PARENT = ".."
        SCRIPT_DIR = os.path.dirname(
            os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
        )
        sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
        import src.pctransforms as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.OnUnitCube(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = FaceScape(16, train=True, base_directory=BASE_NAME, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    #     print(dset.train_labels)
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
