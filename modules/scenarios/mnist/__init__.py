from .provider import Provider
from .trainer import Trainer

import wataru.workflows.base.scenario as basescenario

import itertools


_train_parameters = {
    'unit': [100, 500, 1000, 2000],
    'batchsize': [10, 50, 100, 500],
}


class TrainParameters:
    def __init__(self, param):
        self._keys = list(param.keys())
        self._product = itertools.product(*[param[k] for k in self._keys])

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self._product)
        labeled_x = list(zip(self._keys, list(x)))
        label = '__'.join(['{}_{}'.format(k, v) for k, v in labeled_x])
        return label, dict(labeled_x)


class Scenario(basescenario.Scenario):
    def __init__(self):
        self._provider = None
        self._train_parameters = TrainParameters(_train_parameters)

    def build(self):
        self._provider = Provider().build()
        self._trainer = Trainer(*(self._provider.to_chainer()), parameters = self._train_parameters).build()
        return self
