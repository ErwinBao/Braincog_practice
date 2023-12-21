# This repository is a practice based on Braincog



### In this repo, we implemented Lenet SNN and Spikformer using Braincog, and verified them on the MNIST and CIFAR-10  respectively.



## Requirements

```
timm==0.9.12

pytorch==1.12.1

braincog==0.2.7.19
```



## Lenet



We used LIF neuron to model this network. The threshold and the membrane potential time constant tau of each neuron are set to 0.5 and 2 respectively.



### Training on MNIST

```
cd Lenet
python train.py
```



### Testing MNIST Test Data

```
cd Lenet
# You need to modify the path in test.py
python test.py
```



### Result

After only two epochs of training, the model accuracy reached **97.4%**





## Spikformer



### Reference

**Paper:** Spikformer: When Spiking Neural Network Meets Transformer, [ICLR 2023](https://openreview.net/forum?id=frE4fUwz_h)

**Code:** [ZK-Zhou/spikformer: ICLR 2023, Spikformer: When Spiking Neural Network Meets Transformer (github.com)](https://github.com/ZK-Zhou/spikformer?tab=readme-ov-file)



### What we did

In this project,  we replaced the original Spikingjelly platform with the Braincog platform. We modified model.py, and made small adjustments to other files.

We also used LIF neuron. The threshold and the membrane potential time constant tau of each neuron are set to 0.5 and 2 respectively.



### Training on CIFAR10

```
cd spikformer_cifar10
# You need to modify the path in cifar10.yml
python train.py
```



### Result 

![top1](https://github.com/ErwinBao/Braincog_practice/tree/master/png/top1.png)

The best result is **82.45%**, at epoch 262.



## Additionally

utils.py in each directory contains a function called reset_net. It's used to reset all neurons in a network after each calculation. 



### 分工

包睿：Spikformer代码    

王子雄：Lenet代码    

吴歌：代码调试
