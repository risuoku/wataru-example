from .base import *
import os

STORAGE_DIR = '/dockervolume'
VENDOR_STORAGE_DIR = os.path.join(STORAGE_DIR, 'vendor')

os.environ['CHAINER_DATASET_ROOT'] = os.path.join(VENDOR_STORAGE_DIR, 'chainer')
