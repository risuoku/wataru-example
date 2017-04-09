import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import wataru.workflows.base.trainer as basetrainer
import wataru.workflows.utils as utils

BATCH_SIZE = 100
EPOCH = 2


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Trainer(basetrainer.Trainer):
    def __init__(self, train_data, test_data):
        self._train_data = train_data
        self._test_data = test_data
        self._value = None
    
    def build(self):
        model = L.Classifier(MLP(1000, 10))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        gpu_id = utils.get_available_device_id()
        if gpu_id is None:
            gpu_id = -1
        if gpu_id >= 0:
            chainer.cuda.get_device(gpu_id).use()
            model.to_gpu() 

        train_iter = chainer.iterators.SerialIterator(self._train_data, BATCH_SIZE)
        test_iter = chainer.iterators.SerialIterator(self._test_data, BATCH_SIZE,
                                                 repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (EPOCH, 'epoch'))

        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

        self._trainer = trainer
        return self
