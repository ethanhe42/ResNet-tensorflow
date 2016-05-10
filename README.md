# Fast Residual neural network  
This RNN spends 3 hours to train on a modern CPU. It can reach 63% on CIFAR100 coarse 20 classes task.   
This residual neural network is different from the [original paper](https://github.com/KaimingHe/deep-residual-networks) in there ways:
- Only have 13 layers, which the original paper didn't studied.  
- No ReLU before subsampling convolutional layer, which improve accuracy by 3%  
- Batchnorm is done before addition, which improve accuracy a little bit.  

This residual net can't beat 18/34/150/1000 layers residual nets in the long run, however, more efficient with non-sufficient training time, interestingly.  
Details are shown [here](report/mp2_Yihui%20He.pdf). Archtecture shown at the bottom.  

### results on cifar100  
Traning 3 hours on CPU:  
- Single layer network with PCA whitening and Kmeans which is 75% accurate on CIFAR10, reaches   
    - Train accuracy:  0.613040816327
    - Validation accuracy:  0.562
    - Test accuracy:  0.559
  
- 13 layers ResNet(this repo) **63%**  
  
- [Mimic Learning](resource/do-deep-nets-really-need-to-be-deep.pdf)  50% (with bad teacher model)  

### results on cifar10  
Traning 3 hours on CPU:  
- [Alexnet](https://www.tensorflow.org/versions/r0.8/tutorials/deep_cnn/index.html) reaches 82%  
- 13 layers Residual network(this repo) reaches 84%  
- [Single layer neural network with PCA and Kmeans](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans) reaches 78%(after I fixed a minor bug.)  

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


