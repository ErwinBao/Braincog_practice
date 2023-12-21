import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_net(net: nn.Module):
    """
    在计算下一个输入前，需要将网络中所有节点重置
    :param net: 任何属于 ``nn.Module`` 子类的网络
    :return: None
    """
    for m in net.modules():
        if hasattr(m, 'n_reset'):
            m.n_reset()


