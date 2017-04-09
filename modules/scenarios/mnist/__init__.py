from .provider import Provider
from .trainer import Trainer

import wataru.workflows.base.scenario as basescenario


class Scenario(basescenario.Scenario):
    def __init__(self):
        self._provider = None

    def build(self):
        self._provider = Provider().build()
        self._trainer = Trainer(*(self._provider.to_chainer())).build()
        return self
