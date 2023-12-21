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

from lenet import Lenet
from utils import reset_net


if __name__ == '__main__':
    train_loader, test_loader, _, _ = get_mnist_data(batch_size=1, step=8)

    model = Lenet()
    model.load_state_dict(torch.load('checkpoints/model_2.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_num = 0.
    all_num = 0.
    for i, data in enumerate(test_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        reset_net(model)

        # 取最大值为预测结果
        _, predicted = torch.max(outputs, 1)

        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
            # 预测值与标签值进行比较
            if predicted_num == label_num:
                correct_num += 1.
            all_num += 1.

    correct_rate = correct_num / all_num
    print('correct rate is {:.3f}%'.format(correct_rate * 100))

