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
import h5py
import subprocess
import shlex
import json


ALLOWED_CLASSES = ["person", "plant", "bed", "sofa"]


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNetCls(data.Dataset):
    def __init__(
        self,
        num_points,
        transforms,
        train,
        base_directory,
        download=True,
        cinfo=None,
        folder="modelnet40_ply_hdf5_2048",
        url="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        include_shapes=False,
        contrastive=False,
    ):
        super().__init__()
        BASE_DIR = base_directory
        self.transforms = transforms

        self.folder = folder
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = url
        self.contrastive = contrastive

        CATFILE40 = BASE_DIR + "modelnet40_ply_hdf5_2048/shape_names.txt"
        assert os.path.exists(BASE_DIR)

        self.all_categories = [line.rstrip("\n") for line in open(CATFILE40)]
        self.allowed_labels = []
        for i, label in enumerate(self.all_categories):
            if label in ALLOWED_CLASSES:
                self.allowed_labels.append(i)

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl -k {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train = train
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))

            # filter points and labels
            points, labels = zip(
                *filter(lambda x: x[1] in self.allowed_labels, zip(points, labels))
            )

            # 2, 24, 26, 30 --> 0, 1, 2, 3
            # 2: bed, 24: person, 26: plant, 30: sofa
            # make another copy of the labels for binary classification objective
            # this is done since NLL loss gradients get messed up on implicit changes

            self.living, self.non_living = [1, 2], [0, 3]
            labels = list(labels)
            replacement_dict = {2: 0, 24: 1, 26: 2, 30: 3}
            for i, label in enumerate(labels):
                label[0] = replacement_dict[label[0]]

                if label[0] == 3:
                    l = np.array([3, 0])
                elif label[0] == 2:
                    l = np.array([2, 1])
                else:
                    l = np.array([label[0], label[0]])

                labels[i] = l

            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        if np.ndim(self.labels) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)

        self.set_num_points(num_points)

        if cinfo is not None:
            self.classes, self.class_to_idx = cinfo
        else:
            self.classes, self.class_to_idx = (None, None)

        self.shapes = []
        self.include_shapes = include_shapes
        if self.include_shapes:
            N = len(self.files)
            if self.train:
                T = "train"
            else:
                T = "test"

            for n in range(N):
                jname = os.path.join(self.data_dir, f"ply_data_{T}_{n}_id2file.json")
                with open(jname, "r") as f:
                    shapes = json.load(f)
                    self.shapes += shapes

        if self.train:
            print("Number of Train Samples:", len(self.labels))
        else:
            print("Number of Test Samples:", len(self.labels))

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.contrastive:
            while True:
                pos_idx = np.random.choice(len(self.labels))
                neg_idx = np.random.choice(len(self.labels))
                pos_label = self.labels[pos_idx][0]
                neg_label = self.labels[neg_idx][0]

                if (pos_label in self.living) and (neg_label in self.non_living):
                    if self.labels[idx][0] in self.living:
                        break

                elif (pos_label in self.non_living) and (neg_label in self.living):
                    if self.labels[idx][0] in self.non_living:
                        break

            p_points = self.points[pos_idx, pt_idxs].copy()
            n_points = self.points[neg_idx, pt_idxs].copy()

            p_label = torch.from_numpy(self.labels[pos_idx]).type(torch.LongTensor)
            n_label = torch.from_numpy(self.labels[neg_idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)
            if self.contrastive:
                p_points, n_points = (
                    self.transforms(p_points),
                    self.transforms(n_points),
                )

        if self.include_shapes:
            shape = self.shapes[idx]
            if self.contrastive:
                return (
                    current_points,
                    label,
                    shape,
                    (p_points, p_label, n_points, n_label),
                )

            return current_points, label, shape

        if self.contrastive:
            return current_points, label, (p_points, p_label, n_points, n_label)

        return current_points, label, torch.tensor([])

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass


if __name__ == "__main__":
    from torchvision import transforms
    import sys

    BASE_DIR = "../../../../../"

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
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNetCls(
        16, train=True, transforms=transforms, contrastive=True, base_directory=BASE_DIR
    )
    #     print(dset[0][0])
    #     print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)

    for data in dloader:
        print("loading")
        break
