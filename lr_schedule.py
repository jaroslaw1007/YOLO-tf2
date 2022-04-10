import numpy as np

from config import *

class YOLOLRSchedule(object):
    def __init__ (self, optimizer, steps_per_epoch):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch

    def step(self, global_step):
        epoch = global_step // self.steps_per_epoch
        if epoch < 1:
            lr = 1e-3 + 0.009 * (global_steps / self.steps_per_epoch)
        elif epoch < 75:
            lr = 1e-2
        elif epoch < 105:
            lr = 1e-3
        else:
            lr = 1e-4

        self.optimizer.learning_rate.assign(lr/LINEAR_SCALE)

class MutlistepLR(object):
    def __init__(self, optimizer, steps_per_epoch):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch

    def step(self, global_step):
        epoch = global_step // self.steps_per_epoch
        if epoch < 75:
            lr = 1e-3
        elif epoch < 105:
            lr = 1e-4
        else:
            lr = 1e-5

        self.optimizer.learning_rate.assign(lr)

class CosineAnnealingLR(object):
    def __init__(self, optimizer, steps_per_epoch, warmup=False):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.warmup = warmup

        self.total_epochs = EPOCHS - 1
        self.lr_max = LEARNING_RATE
        self.lr_min = LEARNING_RATE_MIN
        self.learning_rate = self.lr_max
        if self.warmup:
            self.warmup_epochs = EPOCHS_WARMUP
            self.total_epochs -= self.warmup_epochs
            self.lr_warmup = LEARNING_RATE_WARMUP
            self.learning_rate = self.lr_warmup

    def step(self, global_step):
         epoch = global_step // self.steps_per_epoch
         if self.warmup:
             if epoch < self.warmup_epochs:
                 lr = self.lr_warmup + (self.lr_max - self.lr_warmup) * epoch / self.warmup_epochs
             else:
                 epoch -= self.warmup_epochs
                 lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(epoch/self.total_epochs*np.pi))
         else:
             lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(epoch/self.total_epochs*np.pi))

         self.learning_rate = lr
         self.optimizer.learning_rate.assign(self.learning_rate)
