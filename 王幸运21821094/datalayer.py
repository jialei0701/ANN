import full_buffer
import random
from functools import reduce

class DataLayer:
    def __init__(self, all_fast_buffers2, all_gt_buffers2, permutator_group2, normalizers2, output_target_buffers):
        self.all_fast_buffers2 = all_fast_buffers2
        self.all_gt_buffers2 = all_gt_buffers2
        self.permutator_group2 = permutator_group2
        self.normalizers2 = None
        self.output_target_buffers = output_target_buffers

        self.init_normalizers(normalizers2)
        fast_perms = DataLayer.get_all_perms(self.all_fast_buffers2)
        gt_perms = DataLayer.get_all_perms(self.all_gt_buffers2)
        self.set_permutator_group(fast_perms.union(gt_perms), self.permutator_group2 )

    @staticmethod
    def get_all_perms(buffers):
        perms = [buffer.name_perm for buffer in buffers]
        return set(filter(lambda x: x is not None, perms))


    def set_normalizers(self, buffers, normalizers):
        for buffer in buffers:
            # print(buffer.normalizer_name)
            buffer.set_normalizers(normalizers)

    def set_permutator_group(self, perms, permutator_group):
        for perm in perms:
            perm.set_permutator_group(permutator_group)

    def init_normalizers(self, normalizers2):
        self.normalizers2 = normalizers2
        self.set_normalizers(self.all_fast_buffers2, self.normalizers2)
        self.set_normalizers(self.all_gt_buffers2, self.normalizers2)
        otbs = [x for x in self.output_target_buffers.values()]

        otbs = reduce((lambda x, y: x + (y if isinstance(y, list) else [y])), otbs, list())
        self.set_normalizers(otbs, self.normalizers2)

    def normalizers_print(self):
        nor = "\n".join(x + ": " + str(v) for x, v in self.normalizers2.items() if v.is_init)
        print(nor)