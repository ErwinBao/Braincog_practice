from functools import partial

import torch
import torch.nn.functional as F
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data
from braincog.datasets.datasets import get_mnist_data
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d

from torchviz import make_dot


@register_model     # register model in timm
class Lenet(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'mnist'
        if not is_dvs_data(self.dataset):
            init_channel = 1
        else:
            init_channel = 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, 8, kernel_size=(5, 5), padding=(2, 2), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(8, 16, kernel_size=(5, 5), padding=(2, 2), node=self.node),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            BaseLinearModule(784, 256, node=self.node),
            BaseLinearModule(256, 84, node=self.node),
            nn.Linear(84, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


# if __name__ == '__main__':
#     # train_loader, test_loader, _, _ = get_mnist_data(batch_size=1, step=8)
#     # it = iter(train_loader)
#     # # inputs, labels = it.next()
#     # inputs, labels = next(it)
#     # print(inputs.shape, labels.shape)
#     # print(type(inputs))
#     # print(inputs)
#     # spike_rate_vis(inputs[0, :, 0])
#
#     # model = Lenet(layer_by_layer=True, datasets='mnist').cuda()
#     # # print(model)
#     #
#     # outputs = model(inputs.cuda())
#     # print(outputs)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     inputs = torch.randn((1,1,28,28)).to(device)
#     print(inputs)
#     model = Lenet(layer_by_layer=False, datasets='mnist').to(device)
#     yat = model(inputs)
#     chart = make_dot(yat, params=dict(model.named_parameters()))
#     chart.view()
