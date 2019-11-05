---
layout: post
title: Stanford CS224n 自然语言处理（三）
date: 2019-11-05 12:00:00
author: "chenjing"
header-img: "img/post-bg-unix-linux.jpg"
catalog: true
mathjax: true
tags:
  - NLP
  - Stanford
  - 深度学习
---

# 0 前言

这篇文章涵盖 CS224n Lecture 2 中的部分内容。主要内容包括：word2vec 的目标函数梯度的计算；实现 word2vec 的两种模型：Skip-Gram 和 Continuous Bag-of-Word；训练方法：负采样 (Negative sampling)和 Hierarchical softmax。

我们回顾一下 word2vec，word2vec 模型实际上包含：

- **两个算法（模型）**：continuous bag-of-words（CBOW）和 skip-gram。CBOW 是根据上下文单词来预测中心词。skip-gram 则相反，是根据中心词预测上下文词。
- **两种训练方法：**negative sampling 和 hierarchical softmax。Negative sampling 通过抽取负样本来定义目标，hierarchical softmax 通过使用一个有效的树结构来计算所有词的概率来定义目标。

# 1 Continuous Bag-of-Word Model

## 1.1 One-word context 

同样，我们首先从简单的例子开始。其实，one-word context 就是 3.2 中任务实施的示例，在这一节，我们不再举具体的示例，而是通过**数学公式**来表征这一过程。我们规定，如若没有特殊说明，向量皆指**列向量**。

|⚠️警告，接下来会出现大篇幅的数学公式推导，若有看不明白的地方，请自己举一些具体的例子，或者参考 Stanford CS224n 自然语言处理（二） 3.2 中的任务实施示例|

![](/img/in-post/post-cs224n/one_word_net_arch_1.png)

上图是 CBOW 的最简化模型（实际上也是 Skip-Gram 的最简化模型），输入只有一个上下文词，且只预测一个目标词（即中心词）。 $\mathbf{W} \in \mathbb{R}^{V \times N}$ 是输入层到隐藏层的权重矩阵，$\mathbf{W}^{\prime} \in \mathbb{R}^{N \times V}$ 是隐藏层到输出层的权重矩阵，$\mathbf{x}$ 是输入 one-hot 向量，$\mathbf{h}$ 是隐藏向量，$\mathbf{y}$ 是输出概率向量， $\{x_{1}, \cdots, x_{V}\}$ ， $\{h_{1}, \cdots, h_{N}\}$ ， $\{y_{1}, \cdots, y_{V}\}$ ，分别代表输入层、隐藏层、输出层的神经元（units），其中 $\{x_{1}, \cdots, x_{V}\}$ ，中只有一个值为 1，其余值都为 0，我们假设 $x_{k}=1$，$x_{k^{\prime}}=0$， ${k^{\prime} \neq k}$。 $\mathbf{v}_w^{\mathrm{T}}$ 为 $\mathbf{W}^{\mathrm{T}}$ 的某一列，也就是说，$\mathbf{v}_w$ 为 $\mathbf{W}$ 的某一行，这里行列转换最好画个简单的矩阵辅助理解。我们有：

$$\mathbf{h}=\mathbf{W}^{\mathrm{T}} \mathbf{x}=\mathbf{W}_{(k, \cdot)}^{ \quad \mathrm{T}} :=\mathbf{v}_{w_{I}}^{\mathrm{T}}  \tag{1}$$

其中$\mathbf{v}_{w_{I}}^{\mathrm{T}}$是个列向量，为$\mathbf{W}^{\mathrm{T}}$第 $k$ 列 ，形式上等于 $\mathbf{W}$ 的第 $k$ 行，这里的 $\mathbf{W}_{(k, \cdot)}^{\quad \mathrm{\mathrm{T}}}$ 的意思是先取 $\mathbf{W}$ 的第 $k$ 行，再转置 ，同样地， $\mathbf{h}$ 是个列向量，为 $\mathbf{W}^{\mathrm{T}}$ 第 $k$ 列 ，形式上等于 $\mathbf{W}$ 的第 $k$ 行。 $\mathbf{v}_{w_{I}}$ 是输入词 ${w_{I}}$ 的向量表示，即词向量。这样解释不知大家明白否。

其实这里暗示了从输入层隐藏层的激活函数是线性的，因为从 $\mathbf{v}_{w_{I}}$ 到 $\mathbf{h}$ 只经过一个转置。

从隐藏层到输出层有一个不同的权重矩阵 $\mathbf{W}^{\prime} \in \mathbb{R}^{N \times V}$ ，通过这个矩阵，对词表中的每一个词，我们最终能计算得到一个分数 $u_{j}，j = 1,2,\cdots,V$。

$$u_{j}=\mathbf{v}_{w_{j}}^{\prime \mathrm{T}} \mathbf{h}  \tag{2}$$

（2）中，$\mathbf{v}_{w_{j}}^{\prime}$ 是 $\mathbf{W}^{\prime}$ 的第 $j$ 列，然后我们使用 softmax 函数将分数转化成后验概率。

$$p\left(w_{j} \mid w_{I}\right)=y_{j}= softmax(u_j)=\dfrac{\exp \left(u_{j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)}  \tag{3}$$

（3）中，$y_{j}$ 是输出层第 $j$ 个单元的输出，将（1）（2）带入（3）得到：

$$p\left(w_{j} \mid w_{I}\right)=\dfrac{\exp \left(\mathbf{v}_{w_{j}}^{\prime \ \mathrm{T}} \mathbf{v}_{w_{I}}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(\mathbf{v}_{w_{j^{\prime}}}^{\prime  \ {\mathrm{T}}} {\mathbf{v}_{w_{I}}}\right)}  \tag{4}$$

通过以上几个式子，我们发现，似乎 $\mathbf{v}_w^{\prime}$ 也可以作为词向量，因为输出的理想状态也是一个 one-hot 向量。因此，$\mathbf{v}_w$ 与 $\mathbf{v}_w^{\prime}$ 是同一个词 $w$ 的两种词向量表示（前者称为**输入词向量**，后者称为**输出词向量**）。又因为 $\mathbf{v}_w$ 来自于 $\mathbf{W}$ 的行，$\mathbf{v}_w^{\prime}$ 来自于 $\mathbf{W}^{\prime}$ 的列，所以，我们称 $\mathbf{W}$ 为**输入词向量矩阵**，$\mathbf{W}^{\prime}$ 为**输出词向量矩阵**。

> 在之后的论述中，我们将 $\mathbf{v}_w$ 与 $\mathbf{v}_{w}^{\mathrm{T}}$ 都称为输入词向量，只不过前者是行向量，后者是列向量。
>
> 将 $\mathbf{v}_w^{\prime}$ 与 $\mathbf{v}_w^{\prime \; \mathrm{T}}$ 都称为输出词向量，只不过前者是列向量，后者是行向量。

### 1.1.1 更新 $\mathbf{W}^{\prime}$

接下来，我们推导该模型权重更新的式子。 虽然实际中应用这样的式子来更新权重不具备可操作性（之后会解释），但这样的推导有利于我们深入理解原始模型而不应用任何技巧。

对于一个训练样本，我们的训练目标是最大化（4）​，即

$$\begin{align} \max p\left(w_{O} \mid w_{I}\right) \tag{5} &=\max y_{j^{*}} \\ &=\max \log y_{j^{*}} \tag{6} \\ &=u_{j^{*}}-\log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right) :=-E \tag{7} \end{align}$$

$w_{O}$ 为实际的输出词（output word），我们记 $j^*$ 为 $w_{O}$ 在输出层的索引。 $E=-\log p\left(w_{O} \mid w_{I}\right)=\log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)-u_{j^{*}}$ 是我们的损失函数，和（7）相差一个负号，所以我们需要最小化 $E$ ， 这个损失函数是交叉熵损失的一种特殊情况，单个样本的二元交叉熵损失如下：

$$L=-[y \log \hat{y}+(1-y) \log (1-\hat{y})]$$

$y$ 是真实标签，$\hat{y}$ 是预测值，当 $y=1$ 时，上式变为 $L=-\log \hat{y}$ ，形式上和我们的损失函数一致。

现在我们来推导从隐藏层到输出层的权重更新公式。我们从损失函数出发，求 $E$ 关于 $u_{j}$ （从隐藏层输入到输出层的第 $j$ 个单元的分数）的偏导数，

$$\dfrac{\partial E}{\partial u_{j}}=y_{j}-t_{j} :=e_{j} \tag{8}$$

其中，$t_{j}=\mathbf{1}\left(j=j^{*}\right)$ ，意思是 $t_{j}$ 只有在第 $j$ 个单元为实际的输出词时才等于1，否则为0，也就是说 $\mathbf{t}=(t_{1} \; t_{2}  \cdots t_{V})^{\mathrm{T}}$ 是一个 one-hot  向量。注意，（8）仅仅是输出层的预测误差。

然后，求求 $E$ 关于 $w_{ij}^{\prime}$ 的偏导数，也就得到了从隐藏层到输出层的权重更新的梯度。

$$\dfrac{\partial E}{\partial w_{i j}^{\prime}}=\dfrac{\partial E}{\partial u_{j}} \cdot \dfrac{\partial u_{j}}{\partial w_{i j}^{\prime}}=e_{j} \cdot h_{i} \tag{9}$$

所以，当使用随机梯度下降（stochastic gradient descent）时，我们得到更新公式：

$$w_{i j}^{\prime \; (\text{new})}=w_{i j}^{\prime \; (\text{old})}-\eta \cdot e_{j} \cdot h_{i} \tag{10}$$

或者，

$$\mathbf{v}_{w_{j}}^{\prime \; (\text{new})}=\mathbf{v}_{w_{j}}^{\prime \; (\text{old})}-\eta \cdot e_{j} \cdot \mathbf{h} \quad \text { for } j=1,2, \cdots, V \tag{11}$$

其中，$\eta>0$ 是学习率，$e_{j}=y_{j}-t_{j}$ ， $h_{i}$ 是隐藏层的第 $i$ 个单元， $\mathbf{v}_{w_{j}}^{\prime }$ 是 $w_j$ 的输出词向量。这个更新公式意味着我们需要遍历整个词汇表，检查每个单词的输出概率 $y_j$ ，并与我们期望的输出 $t_j$ 相比较。

从（11）中还可以看出，如果 $y_j>t_j$ ，我们就从 $\mathbf{v}_{w_{j}}^{\prime }$ 中减去隐藏向量 $\mathbf{h}$ （在这里，指的就是 $\mathbf{v}_{w_{I}}$）的一部分，使得 $\mathbf{v}_{w_{j}}^{\prime }$ 远离 $\mathbf{v}_{w_{I}}$。如果 $y_j<t_j$ （只会在 $t_{j}=1$ 时发生，即 $w_{j}=w_{O}$） ，我们就给 $\mathbf{v}_{w_{j}}^{\prime }$ 中加上隐藏向量 $\mathbf{h}$ 的一部分，使得 $\mathbf{v}_{w_{O}}^{\prime }$ 接近 $\mathbf{v}_{w_{I}}$。如果 $y_j$ 与 $t_j$ 非常接近，那么基本不更新，一次迭代训练完毕。

> 这里的“接近”和“远离”，指的是两个词向量的内积越大越接近，内积越小越远离。其实很好理解，两个相同的词向量的内积最大，此时，这两个词向量完全相同，也就是最“接近”。

### 1.1.2 更新 $\mathbf{W}$ 

我们得到了 $\mathbf{W}^{\prime}$ 更新式之后，现在的目标就是求得 $\mathbf{W}$ 的更新式。现在，我们求 $E$ 关于 $h_{i}$ 的偏导数，

$$\dfrac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \dfrac{\partial E}{\partial u_{j}} \cdot \dfrac{\partial u_{j}}{\partial h_{i}}=\sum_{j=1}^{V} e_{j} \cdot w_{i j}^{\prime} :=\mathrm{EH}_{i} \tag{12}$$

其中，$h_{i}$ 是隐藏层的第 $i$ 个单元， $u_{j}$、 $e_{j}$ 与 1.1.1 中的定义一致， $EH$ 是一个 $n$ 维向量。

我们回顾一下隐藏层做的线性计算操作，根据（1），我们可以知道，

$$h_{i}=\sum_{k=1}^{V} x_{k} \cdot w_{k i} \tag{13}$$

然后我们求 $E$ 关于 $\mathbf{W}$ 每一个元素 $w_{kj}$ 的偏导数，

$$\dfrac{\partial E}{\partial w_{k i}}=\dfrac{\partial E}{\partial h_{i}} \cdot \dfrac{\partial h_{i}}{\partial w_{k i}}=\mathrm{EH}_{i} \cdot x_{k} \tag{14}$$



所以我们得到，

$$\dfrac{\partial E}{\partial \mathbf{W}}=\mathbf{x} \otimes \mathbf{E H}=\mathbf{x} \mathrm{EH}^{\mathrm{T}} \tag{15}$$

>  $\mathbf{·}$ 表示内积（inner product）， $\otimes$ 表示外积（tensor product）。

最终我们获得了一个 $V \times N$ 的矩阵，因为  $\mathbf{x}$ 中只有一个元素不为 0，所以，在 $ \dfrac{\partial E}{\partial \mathbf{W}} $ 中只有一行为非零行，这一行即为 $\mathrm{EH}^{\mathrm{T}}$，是一个 $n$ 维行向量。最终的更新公式为：

$$\mathbf{v}_{w_{I}}^{(\text{new})}=\mathbf{v}_{w_{I}}^{(\text{old})}-\eta \cdot \mathrm{EH}^{\mathrm{T}} \tag{16}$$

其中，$\mathbf{v}_{w_I}$ 是 $\mathbf{W}$ 的行向量，所以 $\mathbf{v}_{w_I}^{\mathrm{T}}$ 也是上下文词的输入词向量，在上下文词只有一个的情况下，在一次迭代中这是唯一导数不为 0 的行。$\mathbf{W}$ 的其他行在这次迭代中保持不变，因为他们的导数都是 0。

## 1.2 Multi-word context 

下面我们扩展一下上面介绍的模型结构，如下图所示：

![](/img/in-post/post-cs224n/CBOW_arc.png)

这里我们认为有多个上下文词，假设有 $C$ 个，来预测一个目标词（即中心词）。和 1.1 中介绍的类似，我们将每个上下文词与 $\mathbf{W}^{\mathrm{T}}$ 做矩阵乘法，然后对这 $C$ 个结果取平均，作为中间层的隐藏向量。

$$\begin{align} \mathbf{h} &= \dfrac{1}{C} \mathbf{W}^{\mathrm{T}} (\mathbf{x}_1+\mathbf{x}_2+ \cdots + \mathbf{x}_C) \tag{17} \\ &= \dfrac{1}{C} (\mathbf{v}_{w_{1}}+\mathbf{v}_{w_{2}}+\cdots+\mathbf{v}_{w_{C}})^{\mathrm{T}}  \tag{18}\end{align}$$

同样这里的 $\mathbf{v}_{w}$ 是输入词向量，$w_1, w_2, ..., w_C$ 是上下文词，所以损失函数为：

$$\begin{align} E&=-\log p\left(w_{O} \mid w_{I,1},w_{I,2},\cdots,w_{I,C}\right)\tag{19}\\ &=-u_{j^{*}} + \log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)\tag{20}\\ &= -\mathbf{v}_{w_{O}}^{\prime \mathrm{T}} \mathbf{h} + \log \sum_{j^{\prime}=1}^{V} \exp \left(\mathbf{v}_{w_{j}}^{\prime \mathrm{T}} \mathbf{h}\right)\tag{21}\end{align}$$

式（21）与式（7）本质相同，因为今后我们基本不会遇到上下文词只有一个的情况，所以，CBOW 的损失函数可确定为式（21）。

### 1.2.1 更新 $\mathbf{W}^{\prime}$

与 1.1.1 中的推导过程相同，从隐藏层到输出层的权重更新式为：

$$\mathbf{v}_{w_{j}}^{\prime \; (\text{new})}=\mathbf{v}_{w_{j}}^{\prime \; (\text{old})}-\eta \cdot e_{j} \cdot \mathbf{h} \quad \text { for } j=1,2, \cdots, V \tag{22}$$

注意，对于每一个训练实例，都需要应用这个更新式来更新 $\mathbf{W}^{\prime}$ 中的每一个元素。

### 1.2.2 更新 $\mathbf{W}$

与 1.1.2 中的推导过程稍微有些不同，从隐藏层到输出层的权重更新式为：

$$\mathbf{v}_{w_{I,c}}^{(\text{new})}=\mathbf{v}_{w_{I,c}}^{(\text{old})}-\dfrac{1}{C}\cdot\eta \cdot \mathrm{EH}^{\mathrm{T}} \quad \text { for } c=1,2, \cdots, C \tag{23}$$

简单地推导一下：

$\text{for  c = 1, 2,..., C}:$

$$\dfrac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \dfrac{\partial E}{\partial u_{j}} \cdot \dfrac{\partial u_{j}}{\partial h_{i}}=\sum_{j=1}^{V} e_{j} \cdot w_{i j}^{\prime} :=\mathrm{EH}_{i}$$

$$h_{i}=\dfrac{1}{C} \sum_{c=1}^C\sum_{k=1}^{V} x_{ck} \cdot w_{k i}^{c}$$

> $h_i$  的表达式与 1.1.2 中的不同

$$\dfrac{\partial E}{\partial w_{k i}^c}=\dfrac{\partial E}{\partial h_{i}} \cdot \dfrac{\partial h_{i}}{\partial w_{k i}^c}=\mathrm{EH}_{i} \cdot \dfrac{1}{C} \cdot x_{ck}$$

$$\dfrac{\partial E}{\partial \mathbf{W}^c}=\dfrac{1}{C}\mathbf{x}_c \otimes \mathbf{E H}=\dfrac{1}{C}\mathbf{x}_c \mathrm{EH}^{\mathrm{T}} $$

最终我们获得了 $C$ 个 $V \times N$ 的矩阵，因为  $\mathbf{x}$ 中只有一个元素不为 0，所以，在 $ \dfrac{\partial E}{\partial \mathbf{W}^c} $ 中只有一行为非零行，这一行即为 $\mathrm{EH}^{\mathrm{T}}$，是一个 $n$ 维行向量。

容易看出，从输入层到隐藏层的权重更新式与式（16）相似，只不过我们需要将式（23）应用于更新每一个上下文词 $w_{I,c}$。式（23）中，$\mathbf{v}_{w_{I,c}}$ 表示第 $c$ 个输入上下文词中的词向量，$\eta$ 为正的学习率，$\mathrm{EH} = \dfrac{\partial E}{\partial h_{i}}$ 由式（12）给出。其中，$\mathbf{v}_{w_I,c}$ 是 $\mathbf{W^c}$ 的行向量，所以 $\mathbf{v}_{w_{I,c}}^{\mathrm{T}}$也是上下文词的输入词向量，在上下文词只有一个的情况下，在一次迭代中这是唯一导数不为 0 的行。$\mathbf{W^c}$ 的其他行在这次迭代中保持不变，因为他们的导数都是 0。

## 2 Skip-Gram Model

Skip-Gram 与 CBOW 相反，输入是中心词，输出是上下文词。

> Skip-Gram 是 一 预测 多。
>
> CBOW 是 多 预测 一。

![](/img/in-post/post-cs224n/SG_arc.png)

我们仍然使用 $\mathbf{v}_{w_I}$ 表示输入层中心词的输入词向量，因此隐藏层状态 $h$ 的定义与式（1）相同。也即意味着 $h$ 只是复制（转置）输入层 $\to$ 隐藏层 权重矩阵 $\mathbf{W}$ 的一行，

$$\mathbf{h}=\mathbf{W}^{\mathrm{T}} \mathbf{x}=\mathbf{W}_{(k, \cdot)}^{ \quad \mathrm{T}} :=\mathbf{v}_{w_{I}}^{\mathrm{T}}  \tag{24}$$

在输出层，我们将输出 $C$ 个概率分布，而不是一个。 使用相同的 隐藏层 $\to$ 输出层 矩阵计算每个输出：

$$p\left(w_{c,j}=w_{O,c} \mid w_{I}\right)=y_{c, j}= softmax(u_{c,j})=\dfrac{\exp \left(u_{c,j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)}  \tag{25}$$

其中，$w_{c,j}$ 是第 $c$ 个 $\mathbf{W}^{\prime}$ 中的第 $j$ 列的输出词向量所代表的词； $w_{O,c}$ 输出上下文词中的第 $c$ 个词； $w_{I}$ 是输入词； $y_{c,j}$ 是第 $c$ 个 $\mathbf{W}^{\prime}$ 中的第 $j$ 个单元的输出； $u_{c,j}$ 第 $c$ 个 $\mathbf{W}^{\prime}$ 的输出中第 $j$ 个分数；因为输出权重矩阵共享参数，因此：



$$u_{c,j}=u_{j}=\mathbf{v}_{w_{j}}^{\prime \quad \mathrm{T}} \cdot \mathbf{h,} \quad \text{for c = 1, 2, ... , C} \tag{26}$$

其中，$\mathbf{v}_{w_{j}}^{\prime}$ 是第 $j$ 个输出词向量对应的词，也是 $\mathbf{W}^{\prime}$ 第 $j$ 列。

> 这里提一句，模型初始化时，生成相同的 C 个分布，但是在反向传播时，由于每个词不同，各自减去的比例也不同，因此 $\mathbf{W}^{\prime}$ 也会根据不同的词发生不同的变化。

## 2.1 更新 $\mathbf{W}^{\prime}$ 

隐藏层 $\to$ 输出层的参数更新方程的推导与 1.1 中的没有太大不同。损失函数改为：

$$\begin{align} E&=-\log p\left(w_{O} \mid w_{I,1},w_{I,2},\cdots,w_{I,C}\right)\tag{27}\\ &= -\log[p\left(w_{O,1} \mid w_{I}  \right) p\left(w_{O,2} \mid w_{I} \right) \cdots p\left(w_{O,C} \mid w_{I}  \right)] \tag{28} \\ &=-\log \prod_{c=1}^{C} \frac{\exp \left(u_{c, j_{c}^{*}}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)}\tag{29}\\ &= -\log [\prod_{c=1}^{C} \exp \left(u_{c, j_{c}^{*}}\right)] + -\log [\prod_{c=1}^{C} \sum_{j^{\prime}=1}^{V} \exp \left(u_{ j^{\prime}}\right)] \tag{30} \\ &= -\sum_{c=1}^{C} u_{j_{c}^{*}}+C \cdot \log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)\tag{31}\end{align}$$

其中， $u_{j_{c}^{*}}$ 是第 $c$ 个输出上下文词的索引。

我们求输出层的每个 panel 上的每个单元的输入 $u_{c,j}$ 对 $E$ 的偏导数，得到：

$$\dfrac{\partial E}{\partial u_{c, j}}=y_{c, j}-t_{c, j}:=e_{c, j} \tag{32}$$

> 这里的 panel 的意思是 “面板” ， 可以看 Skip-Gram 模型的输出部分加以理解。

和（8）类似， $e_{c, j}$ 是预测误差。为了简化标记，我们定义一个 $V$ 维的向量 $\mathrm{EI}=\left\{\mathrm{EI}_{1}, \cdots, \mathrm{EI}_{V}\right\}$ ，作为所有输出上下文词的预测误差的和。

$$\mathrm{EI}_{j}=\sum_{c=1}^{C} e_{c, j} \tag{33}$$

然后，我们求 $\mathbf{W}^{\prime}$ 对 $E$ 的偏导数：

$$\dfrac{\partial E}{\partial w_{i j}^{\prime}}=\sum_{c=1}^{C} \dfrac{\partial E}{\partial u_{c, j}} \cdot \dfrac{\partial u_{c, j}}{\partial w_{i j}^{\prime}}=\mathrm{EI}_{j} \cdot h_{i} \tag{34}$$

最终得到 $\mathbf{W}^{\prime}$ 的更新式：

$$w_{i j}^{\prime \quad \mathrm{(new)}} = w_{i j}^{\prime \quad \mathrm{(old)}}-\eta \cdot \mathrm{EI}_{j} \cdot h_{i} \tag{35}$$

或，

$$\mathbf{v}_{w_{j}}^{\prime \quad \mathrm{(new)}} =\mathbf{v}_{w_{j}}^{\prime \quad \mathrm{(old)}}-\eta \cdot \mathrm{EI}_{j} \cdot \mathbf{h} \quad \text { for } j=1,2, \cdots, V \tag{36}$$

除了预测误差是输出层中所有上下文词中累加而成的外，对更新方程（36）的直观理解与（11）相同。 注意，我们需要为每个训练实例的 hidden $\to$ output 矩阵的每个元素应用此更新方程。

## 2.2 更新 $\mathbf{W}$ 

输入层 $\to$ 隐藏层矩阵的更新公式的推导与（12）至（16）相同，不同之处在于将预测误差 $e_j$ 替换为 $\mathrm{EI}_j$ 。 我们直接给出更新公式：

$$\mathbf{v}_{w_{I}}^{(\text {new})}=\mathbf{v}_{w_{I}}^{(\text {old})}-\eta \cdot \mathrm{EH}^{\mathrm{T}} \tag{37}$$

其中 $\mathrm{EH}$ 是一个 $N$ 维向量。由（12）：

$$\dfrac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \sum_{c=1}^{C} \dfrac{\partial E}{\partial u_{c, j}} \cdot \dfrac{\partial u_{c, j}}{\partial h_{i}} = \sum_{j=1}^{V} \underbrace{\sum_{c=1}^{C} e_{c,j}}_{\mathrm{EI}_{j}} \cdot w_{ij}^{\prime} = \mathrm{EH}_i \tag{38}$$

所以 $\mathrm{EH}$ 的每个元素为：

$$\mathrm{EH}_i = \sum_{j=1}^{V} e_{c,j} \cdot w_{ij}^{\prime} \tag{39}$$

对（37）的理解与（16）处的相同。

# 3 小结

这篇文章论述了 word2vec 的两个模型 CBOW 和 Skip-Gram 的数学表示，包含详细的数学推导及验证，如果有难以理解的地方，请一定要举出具体的例子，比如，举出一个 $\mathbf{W}$ 和 $\mathbf{W}^{\prime}$ 去走一遍数学推导。由于篇幅不宜过长，两种可以提高训练效率的方法将在下一篇文章中详细说明。

## 参考文献

[word2vec Parameter Learning Explained ](https://arxiv.org/pdf/1411.2738.pdf)