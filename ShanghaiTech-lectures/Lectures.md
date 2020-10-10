# ShanghaiTech CS286

超参数控制：

- 模型复杂度
- 模型复杂度类型
- 优化算法
- 模型类型

## 传统机器学习


### 回归

#### 线性回归

$$
\hat{y} = h_{\theta}(x) = \theta_0 + \theta_1x_1 + ... +  \theta_nx_n
$$

向量表示为参数向量 $\theta$ 与$X$ 的点积。

用均方误差（root mean square error, RMSE）表示拟合优度。

实践中直接使用 MSE（mean square error）构建损失函数。
$$
J(\theta) = MSE
$$

#### 梯度下降（Gradient descent）

- 随机化参数
- 使用梯度下降改变参数（偏微分），优化损失函数



问题：

- 局部最优
- 平台停滞



其他类型：

- 批量梯度下降
- 随机梯度下降（SGD）
- 小批量梯度下降



#### 多项式回归

对变量加幂次建模。



#### 学习曲线

随着训练集增大，训练和验证集损失变化。



#### 正则化线性模型

引入惩罚项。

- 岭回归：让参数变小
- Lasso 回归：让参数变得稀疏（参数减少）



#### 逻辑回归

$$
\hat{p} = h_{\theta}(x)\\

\hat{y} = 1 \space if \space \hat{p} > 0.5
$$



#### Softmax 回归

用于多分类。



### 贝叶斯统计

#### 条件概率

- 联合概率
- 条件概率
- 独立事件

#### 贝叶斯定理

贝叶斯公式

![image-20200915192833364](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200915192833.png)

主观判断和客观事件以概率进行联系。

最大似然法（ML）
$$
P(D|\alpha, M)
$$
给定模型 $M$ 和参数 $\alpha$，给定数据 D 的似然度。

由此反向找到对应的参数。

优点：

- 一致性好
- 高效
- 参数转换不变异

缺点：

- 数据非常少时结果差

最大化后验概率（MAP）
$$
P(\alpha|D, M)
$$
参数先验概率 
$$
P(\alpha|M)
$$

MCMC（马尔可夫链蒙特卡洛方法） 推断后验概率使得贝叶斯计算变得可行。

#### 图模型

- 概率图模型

- 条件独立

##### 贝叶斯网络

also called belief networks

- 概率推断 - 计算每一种原因/解释的后验概率

### 支持向量机

#### 线性 SVM 分类

寻找一个距离不同类都比较远的直线/平面。

类之间距离最近的点是支持向量，在点之间寻找一个决策边界。

- 硬边界分类

  - 问题：
    - 数据必须线性
    - 对离群值很敏感

- 软边界分类：允许错误

  - 需要定义阈值平衡错误和泛化能力

#### 非线性 SVM

- landmark

#### 核 SVM

通过映射函数将数据映射到高纬度空间，以便于可分。

#### 缺点

- 受噪声影响大
- 只适用二分类，需要做额外处理做多分类



### 决策树

- Gini 不纯度或熵

- CART 算法



### 集成学习

不同的模型有多样性。

- 同样的算法应用到不同的子集

方法：

- Bagging：重抽样
- Pasting：不重抽样



#### 随机森林

基于 bagging 方法。

- 样本和特征（例如，开根号个）都随机选择



#### Boosting and Stacking

- Boosting
- Gradient boosting
- Stacking: 使用模型（blender）处理模型集成的投票（blending），而不是硬投票（选择预测结果最好的子模型的结果）



### 降维

主要有两类方法：

- 映射 projection，如主成分分析 PCA
- 流形学习 manifold learning（非线性降维）
  - t-SNE 属于非线性降维

#### PCA

找到一个离数据最近的超平面，然后将数据映射过去。

- SVG 求解



选择 PC 数量

- elbow plot



修改版本：

- Randomized PCA
- Incremental PCA
- Kernel PCA



### 非监督学习

- K-means
- DBSCAN
- Gaussian mixture model (GMM)



应用例子：

- 聚类

- 异常点检测
- 密度估计



聚类：

- 基于中心点聚类
  - kmeans
- 基于密度大聚类
  - DBSCAN
- 基于分布的聚类
- 层次聚类



#### k-means

迭代算法。

最优数目。

- elbow rule
- silhouette coef



#### DBSCAN

将类看作密度很高的连续区域。



#### 高斯混合模型

拆分为高斯分布的组合。



## 深度学习



### 神经网络

- 生物到人工神经元
- 训练神经网络
- 调参
- 梯度消失/爆炸
- 重复使用预训练层
- 优化器
- 降低过拟合（正则化）



#### 人工神经网络

- Linear threshold unit (LTU)：连续值到离散值的映射

- Perceptron

  - Input layer
  - Output layer
  - Hebb's rule

- Multi-Layer Perceptron (MLP)

  - Input layer
  - Hidden layer
  - Output layer

- Deep neural network (DNN)

  - two or more hidden layers
  - Useful when backpropagation algorithm in 1986

- 反向传播算法

  - Forward pass
  - Reverse pass
  - Gradient descent

  ![image-20200924184706915](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200924184707.png)

- 激活函数：导数特性不一样

  - Logistic function
  - Hyperbolic tangent function
  - ReLU function

![image-20200924184734536](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200924184734.png)

#### 训练神经网络

- 很多步骤

![image-20200924184757456](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200924184757.png)

![image-20200924184815728](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200924184815.png)

#### 调参

![image-20200924184920032](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200924184920.png)



- 隐藏层数
  - 更深的网络需要更多的数据
  - 更深更容易过拟合
  - 更深更强大
- 每一层神经元数量
  - 逐渐增加直至过拟合
  - 使用宽松的模型（数量多），然后利用正则化技术防止过拟合
- 激活函数
  - 隐藏层目前一般使用 ReLU
  - 输出层
    - 分类使用 softmax
- 问题
  - 梯度消失
  - 梯度爆炸
  - 上述原因主要在于权重初始化和激活函数
- Batch Normalization (BN)
  - 做法：及时调整值，避免过大过小
  - 可以使用一些饱和激活函数
  - 对权重初始化不敏感了
  - 可以使用更大的学习率
  - 类似正则化



#### 迁移学习

- 重新利用已构建好的模型



#### 无监督学习做预训练



#### 利用辅助任务进行预训练

- 自监督学习



#### 更快地优化

比梯度下降更快的优化器

- Momentum optimization
- Nesterov Accelerated Gradient (NAG)
- AdaGrad
- RMSProp
- Adam
- Nadam optimization



学习率策略：

- 先高学习率然后降低学习率



#### 过拟合处理

正则化

- dropout（一般是 50%）
- L1 和 L2 正则化
- ...



### CNN

#### Why CNN

- 数据大（图像像素多）会导致模型复杂、难以计算
- receptive field/weight sharing/pooling-subsampling
- 先多层卷积和衔接降采样，随后接上多层神经网络



#### 卷积层激活函数

常用：

- ReLU
- Leaky ReLU



#### 池化层 pooling

- Max pooling



#### LeNet-5

![image-20200929182559376](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20200929182559.png)



#### CNN pipeline

1. Preprocess the data

```
x = x - mean(x)
x = x / sd(x)
```

2. Choose the architecture
3. 用小量样本测试
4. 超参优化 - best to optimize in log space



#### 网络结构

- VGGNet
- GoogLeNet
- ResNet
- SENet



#### 目标检测 CNN

- Fast R-CNN
  - input image > ConvNet > conv feature map of image > Rol Pooling > FCs > Linear + softmax/Linear
- Faster R-CNN
  - Region Proposal Network



#### Sequence modeling

- 将 ACGT 每一个字母转为 4 维向量



将 CNN 看作一个特征提取器



### RNN

-  Suitable for sequence data



#### RNN

![image-20201010182606699](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20201010182606.png)

![image-20201010182534987](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20201010182535.png)

![image-20201010182633762](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20201010182633.png)



#### Long-term Short Term Memory

- LSTM
  - Replacing a vanilla RNN neuron by the LSTM unit
  - Wants the short-term memory to last for a long time period

![image-20201010190243175](https://gitee.com/ShixiangWang/ImageCollection/raw/master/png/20201010190243.png)

