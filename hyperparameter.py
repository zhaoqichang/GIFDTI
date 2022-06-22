# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import torch
from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.Learning_rate = 5e-5
        # self.Learning_rate = 4e-5
        self.Epoch = 500
        self.Batch_size = 32
        self.validation_split = 0.2
        self.weight_decay = 5e-5