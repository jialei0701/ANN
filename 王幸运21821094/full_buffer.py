import numpy as np
import gglue
import random
import itertools
import math
import config

if config.config.small_gpu:
    img_size = (64, 64)
    #img_size = (128, 128)
else:
    img_size = (384, 384)
    img_size = (512, 512)

safe_border = (22, 22)
border = 0

