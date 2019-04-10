from perm import Perm
import gglue

def find_same_buffer_index2(buffer_list, buffers_candidates):

    if not isinstance(buffers_candidates, list):
        buffers_candidates = [buffers_candidates]

    for buffer in buffers_candidates:
        for idx, buffer_b in enumerate(buffer_list):
                if buffer_b.is_same(buffer):
                    return idx, buffer

    assert False

def get_real_buffers(buffers):
    return filter(lambda x: gglue.get_buffer(x).random_constant_range is None, buffers)

class Buffer:
    def __init__(self, name, normalizer_name=None, hidden=False, random_constant_range=None):
        self.name_real = None
        self.name_perm = None
        if isinstance(name, Perm):
            self.name_perm = name
        else:
            self.name_real = name
        self.normalizer_name = normalizer_name
        self.normalizers3 = None
        self.hidden = hidden
        self.random_constant_range = random_constant_range



    def reverse_np(self, x):
        assert self.normalizers3 is not None
        if self.normalizer_name is not None:
            return self.normalizers3[self.normalizer_name].reverse_np(x)
        else:
            return x

    def forward_torch(self, idx, img):
        if self.normalizer_name is not None:
            self.normalizers3[self.normalizer_name].forward_torch(idx, img)

    def accum_pixels(self, img):
        self.normalizers3[self.normalizer_name].accum_pixels(img)

    def apply_filter(self, img):
        if self.normalizer_name is not None:
            return self.normalizers3[self.normalizer_name].apply_filter(img)
        else:
            return img

    def get_real_buffer_name(self, perm_inst):
        if self.name_real is not None:
            return self.name_real
        else:
            return self.name_perm.get_real_buffer_name(perm_inst)

    def set_normalizers(self, normalizers3):
        self.normalizers3 = normalizers3

    def get_all_buffer_names(self):
        if self.name_real is not None:
            return [self.name_real]
        else:
            return self.name_perm.get_all_buffer_names()

    def is_hidden(self):
        return self.hidden


    def is_same(self, that):
        equal = True

        equal = equal and self.normalizers3 == that.normalizers3
        equal = equal and self.normalizer_name == that.normalizer_name

        if self.name_real:
            equal = equal and self.name_real == that.name_real
        else:
            equal = equal and self.name_perm.is_same(that.name_perm)


        return equal


    def __str__(self):
        if self.name_real is not None:
            name_value = self.name_real
        else:
            name_value = str(self.name_perm)

        return "Buffer(name={}, normalizer_name={})".format(name_value, self.normalizer_name)
