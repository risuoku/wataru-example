import chainer
import wataru.workflows.base.provider as baseprovider


class Provider(baseprovider.Provider):
    def __init__(self):
        self._value = None
    
    def build(self):
        self._value = chainer.datasets.get_mnist()
        return self
    
    def to_chainer(self):
        return self._value
