import socket


class Config:
    def __init__(self):
        self.small_gpu = False
        self.num_workers = 9
        self.remote = False

config = Config()