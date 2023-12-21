from functools import partial
import torch
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data
from braincog.datasets.datasets import get_mnist_data
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d
from torch import optim
from tqdm import tqdm

from lenet import Lenet
from utils import reset_net


if __name__ == '__main__':
    train_loader, test_loader, _, _ = get_mnist_data(batch_size=1, step=8)
    # it = iter(train_loader)
    # inputs, labels = it.next()
    # inputs, labels = next(it)
    # print(inputs.shape, labels.shape)
    # print(type(inputs))
    # print(inputs)
    # spike_rate_vis(inputs[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lenet(layer_by_layer=True, datasets='mnist').to(device)
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # 输入数据
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            reset_net(model)

            running_loss += loss.item()

            if i % 100 == 99:
                print(
                    '[{}/{}] loss: {:.3f}'.format(epoch + 1, 5, running_loss / 100))
                running_loss = 0.0

        # 保存参数文件
        torch.save(model.state_dict(), 'checkpoints/model_{}.pth'.format(epoch + 1))
        print('model_{}.pth saved'.format(epoch + 1))

    print('Finished Training')

    # outputs = model(inputs.cuda())
    # print(outputs)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
