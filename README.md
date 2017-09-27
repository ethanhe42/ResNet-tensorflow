# Fast ResNet for CIFAR
This ResNet spends 3 hours to train on a modern CPU. It can reach 63% on CIFAR100 coarse 20 classes task.   
This residual neural network is different from the [original paper](https://github.com/KaimingHe/deep-residual-networks) in there ways:
- Only have 13 layers, which the original paper didn't studied.  
- No ReLU before subsampling convolutional layer, which improve accuracy by 3%  
- BatchNorm is done before addition, which improve accuracy a little bit.  

This ResNet-13 can't beat ResNet-18/34/150/1000 layers residual nets in the long run, however, more efficient with non-sufficient training time, interestingly.  
Details are shown [here](report/mp2_Yihui%20He.pdf). Archtecture shown at the bottom.  

### results on CIFAR-10/CIFAR-100
Traning 3 hours on CPU:  

Acc. | CIFAR-10 | CIFAR-100 
--- | --- | ---
[Alexnet](https://www.tensorflow.org/versions/r0.8/tutorials/deep_cnn/index.html) | 82% | -
[Mimic Learning](resource/do-deep-nets-really-need-to-be-deep.pdf) | - | 50%
[2-layer NN with PCA and Kmeans](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans) | 78% | 56%
ResNet-13 (this repo) | **84%**  | **63%** 

### How to run  
`python redo.py /path/to/CIFAR100python/`  

### Features  
- [x] Output layers contain 20 labels
- [x] Using tensorflow
- [x] number of filters
- [x] iterations
- [x] learning rate
- [x] batch size
- [x] regularization strength
- [x] number of layers
- [x] optimization methods
  - [Mimic Learning](resource/do-deep-nets-really-need-to-be-deep.pdf)  
- [x] drop out
- [x] initialization
  - [LSUV init](resource/ALL%20YOU%20NEED%20IS%20A%20GOOD%20INIT.pdf)
  - Kaiming He's initialization
- [x] hidden neurons
- [x] filter size of convolution layers
- [x] filter size of pooling layers  

### architecture  
![arch](report/arch.png)


