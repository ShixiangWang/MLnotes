# MLnotes

Machine learning and deep learning notes in R or Python

## 安装

### Python 安装 keras

```sh
conda create -n learn
conda activate learn
# 先安装 tensorflow 更好，不然后面装的包都太新了
conda install tensorflow-gpu
conda install numpy scipy matplotlib jupyter yaml h5py
pip install pydot-ng opencv-python
#conda install tensorflow-gpu
## If use GPU, download and install CUDA and CUDNN,
## Anaconda 好像封装了 GPU 的依赖
## 没有 GPU for MacOS 
conda install keras
```

### R 安装 keras


```r
install.packages("keras")

library(keras)
install_keras()
```

### Python 安装 pytorch

```sh
conda install -c pytorch pytorch
conda install -c pytorch torchvision
# gpu install
# <https://github.com/williamFalcon/pytorch-gpu-install>
#conda install -c pytorch cuda100
```