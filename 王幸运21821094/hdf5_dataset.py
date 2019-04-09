
import torch.nn as nn
import h5py
import torch.utils.data
import numpy as np
import torch.optim as optim
import traceback
import numpy as np
import os
import re
import copy
import random

# Dataset is a h5 file or a directory of renders/render patches
# Each render is dataset (compressed float16) with it's own name
# It comprises of concatenated  (sub)renders with different spp counts
# For training, render patches must be the same size
# Each h5 files contains following attributes:
# - array of spp_count - represents spp count of sub renders
#                        0 represents ground truth. We can have
#                        multiple subrender with the same spp,
#                        but unique seed
# - array of layers_size - represents the number of channels for each
#                          subrender.
#                          So the total num of channels = sum(layers_size)

# Currently, we have 11 channels in the following order
# 3x RGB Color (mean=0, std=1)
# 1x Depth (mean=40, std=150)
# 3x RGB Texture (mean=.45, std=.45)
# 3x Normals (mean=0, std=.45))
# 1x Shadow (mean=.05, std=.37))
#


def get_paths(path):
    if os.path.isfile(path):
        return [path]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_info(path):
    f = h5py.File(get_paths(path)[0], 'r')
    m = int(np.max(f.attrs['spp_count']))

    def r(n):
        if n == 1:
            return [n]
        else:
            return r(n//2) + [n]

    return r(m)


class hdf5_dataset(torch.utils.data.Dataset):
    def __init__(self, path, needed_spps, num_group=1, group=0, permute_spp=False,
dataset_filter=0 ):
        super(hdf5_dataset, self).__init__()

        assert dataset_filter != 0
        self.dataset_filter = dataset_filter

        self.pathes = get_paths(path)

        self.permute_spp = permute_spp

        self.renders = []
        self.renders_file_index = []

        self.spp_map_per_file = []
        self.saccum_per_file = []


        for idx, path in enumerate(self.pathes):
            try:
                hdf5_file = h5py.File(path,'r')
            except Exception as e:
                print(path)
                raise e
            renders = list(hdf5_file)

            spp_map_per_file = {}

            if 'spp_count' in hdf5_file.attrs:
                spp_counts = (hdf5_file.attrs['spp_count'])
                layers_size = hdf5_file.attrs['layers_size']

            else:
                assert False

            layers_size = np.insert(layers_size, 0, 0)
            layers_size_accum = np.cumsum(layers_size)

            for id, spp in enumerate(spp_counts):
                if spp not in spp_map_per_file:
                    spp_map_per_file[spp] = [id]
                else:
                    spp_map_per_file[spp].append(id)


            self.spp_map_per_file.append(spp_map_per_file)
            self.saccum_per_file.append(layers_size_accum)

            def good(x):

                bad_list = ["Japanese", "75302_Contemporary_Bathroom", "88727_STEAMPUNK_TARANTAS", "bmw27",
                            "classroom_classro", "pabellon_barcelona", "Revenge"]

                for bad in bad_list:

                    if x.find(bad) != -1:
                        return False

                return True

            renders = [x for x in renders if good(x)]


            if self.dataset_filter is not None:

                prog = re.compile(self.dataset_filter)

                renders = [x for x in renders if prog.search(x) is not None]

            self.renders.extend(renders)
            self.renders_file_index.extend([idx for _ in range(len(renders))])


        self.needed_spp = needed_spps


    def __len__(self):
        return len(self.renders )


    def __getitem__(self, index):
        filename = self.renders[index]
        h5_index = self.renders_file_index[index]

        hdf5_file = h5py.File(self.pathes[h5_index], 'r')
        data = hdf5_file[filename]

        spp_map = copy.deepcopy(self.spp_map_per_file[h5_index])
        saccum = self.saccum_per_file[h5_index]

        indexes = []

        for spp in self.needed_spp:

            if self.permute_spp:
                indexes.append(spp_map[spp].pop(random.randrange(len(spp_map[spp]))))
            else:
                indexes.append(spp_map[spp].pop())

        data = [data[saccum[x]:saccum[x+1], :, :].astype(np.float32) for x in indexes]


        return [0, index] + data

    def get_name(self, index):
        return self.renders[index]

    def get_by_name(self, name):
        print(self.renders)
        if name in self.renders:
            index = self.renders.index(name)

            return self[index]
        else:
            raise Exception("{} not found in the dataset".format(name))
