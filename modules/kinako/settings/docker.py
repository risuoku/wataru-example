import os

STORAGE_DIR = '/dockervolume'

os.environ['CHAINER_DATASET_ROOT'] = '{}/{}'.format(STORAGE_DIR, 'chainer')
