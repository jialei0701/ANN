import numpy as np
import math
import scipy.signal
import sys
import traceback

class BufferNormalizer:
    def __init__(self, calc_normalize=False, log_plus_one=False, custom_normalize=None,
                 grad_x=False, grad_y=False, avg='Not Init', std='Not Init', guassian=False):
        #self.name = name



        # custom_normalize = [0, 1]
        # grad_x = False
        # grad_y = False
        # log_plus_one = False
        # calc_normalize = False

        self.log_plus_one = log_plus_one

        assert self.log_plus_one == False
        assert grad_x == False
        assert grad_y == False
        assert guassian== False


        self.grad_x = grad_x
        self.grad_y = grad_y

        self.sum  = 0.0
        self.sum_sq  = 0.0
        self.avg = avg
        self.std = std
        self.num_pixels = 0
        #self.calc_normalize = calc_normalize
        self.normalize = not (custom_normalize is not None)
        self.is_init = not self.normalize

        self.guassian = guassian

        if custom_normalize is not None:
            self.avg = custom_normalize[0]
            self.std = custom_normalize[1]

    def __str__(self):
        out = {}
        if self.normalize:
            out['normalize'] = True
            out['std'] = self.std
            out['avg'] = self.avg

        if self.log_plus_one:

            out['log_plus_one'] = True

        return str(out)

    def reverse_np(self, x):

        if self.normalize:
            x = x*self.std+self.avg
        if self.log_plus_one:
            x = np.exp(x) - 1
        return x


    def reverse_torch(self, x):

        if self.normalize:
            x = x*self.std+self.avg
        if self.log_plus_one:
            assert False
            x = np.exp(x) - 1
        return x

    def forward_torch(self, idx, img):


        if self.log_plus_one:
            img[:, idx, :, :].clamp_(-.5, 1e5)
            img[:, idx, :, :].log1p_()

        if self.normalize:
            img[:, idx, :, :].add_(-self.avg)
            img[:, idx, :, :].mul_(1 / self.std)


    def forward_torch_single(self, img):


        if self.log_plus_one:
            assert False
            img = img.clamp(-.5, 1e5)
            img = img.log1p()

        if self.normalize:
            img = img.add(-self.avg)
            img = img.mul(1 / self.std)

        return img


    def forward_numpy(self, img):
        if self.log_plus_one:
            assert False
            img = img.clip(-.5, 1e5)
            img = np.log1p(img)

        if self.normalize:
            print(self.avg, self.std)

            img = (img - self.avg) /  self.std

        return img

    def apply_filter(self, img):

        if self.grad_x:
            img = scipy.signal.convolve2d(img, np.array([[-1, 1]], dtype=np.float32), mode='same', boundary='symm')
        if self.grad_y:
            img = scipy.signal.convolve2d(img, np.array([[-1], [1]], dtype=np.float32), mode='same', boundary='symm')

        if self.guassian:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma=2, truncate=4)
        return img


    def accum_pixels(self, img):
        assert not self.is_init

        img = self.apply_filter(img)

        if self.log_plus_one:
            assert False
            img = np.clip(img, -.5, 1e5)
            img = np.log1p(img)

        self.num_pixels += img.size

        if self.normalize:
            sm = np.sum(img)
            assert np.isfinite(sm)
            self.sum +=sm
            smsq = (img * img).sum()
            assert np.isfinite(smsq)
            self.sum_sq += smsq

    def finilize_norm(self):
        if self.normalize:
            assert not self.is_init
            assert self.num_pixels != 0
            self.avg = self.sum / self.num_pixels

            std2 = self.sum_sq / self.num_pixels - self.avg*self.avg
            if not std2 > 1e-5:
                print(self.sum_sq, self.num_pixels, self.avg)
                print(std2)
                traceback.print_exc()
                sys.exit(-1)
            self.std = math.sqrt(std2)

            self.is_init = True

            # self.avg = 0.0
            # self.std = 1.0
