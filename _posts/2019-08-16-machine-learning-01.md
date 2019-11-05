---
layout: post
title: 机器学习-白板推导系列（一）
date: 2019-08-16 17:00:00
author: "chenjing"
header-img: "img/post-bg-universe.jpg"
catalog: false
mathjax: true
tags:
  - 机器学习
  - 公式推导
---

# 绪论

再开一个坑。

前段时间突然想巩固机器学习的相关内容，幸而在b站看到一位大佬的[《机器学习-白板推导》系列课程](https://space.bilibili.com/97068901/video)，花了大概20天左右把课程听完了。这个系列的课程对 “小白”（学过线性代数、概率论以及微积分）非常友好，因为你面对的不是枯燥的、满是公式的专业教材，而是一个愿意花时间做出细致且形象的解释的大佬😏，而且大佬一步一步非常清晰地 纯 · 手推公式，大佬请受我一拜。大佬在讲一个模型时会引申出很多模型和算法，并找出他们的相同点和不同点，到后来你会发现很多模型都有内在的联系，豁然开朗，越听越有味，一天不听浑身发痒。。。希望你也有这样的体会😬

好了，开这个系列是为了保存笔记，其间可能会加一些自己对模型或算法的理解，但大部分以大佬的为主。

一开始，介绍了机器学习的两个派别，**频率派**和**贝叶斯派**，频率派渐渐发展出了**统计机器学习**，贝叶斯派渐渐发展出了**概率图模型**。

### 参考书籍：

李航的《统计学习方法》、周志华的《机器学习》（西瓜书）、《Deep Learning》（花书），以及机器学习三大神书《Pattern Recognition and Machine Learning》（PRML）、《Machine Learning : A Probabilistic Perspective》（MLAPP）、《The Elements of Statistical Learning》（ESL）。其中，李航的《统计学习方法》和《ESL》属于频率派；《PRML》属于贝叶斯派；《MLAPP》有点像《PRML》和《ESL》的结合体，是百科全书性质的书，但主要以贝叶斯的角度来写；周志华的机器学习更像是一本手册，没有深入的公式推导，但介绍的很全面；《Deep Learning》就是大名鼎鼎的花书了，看书名就知道是讲深度学习的。

> 李航的《统计学习方法》中讲了10个算法，用一句口诀来记：感K朴决逻，支提E隐条。《PRML》的主要内容也可总结为一句口诀：回分神核稀，图混近采连，顺组。

### 视频资料：

1  台湾大学林轩田的《机器学习基石》：VC Theory、正则化、线性模型等；《机器学习技法》：SVM、决策树、随机森林、神经网络等。

2  张志华的《机器学习导论》：主要是以频率派的角度阐述；《统计机器学习》：主要讲统计上的一些理论，以贝叶斯的角度阐述，偏数学方面。这两门课是张志华老师在上海交通大学时开的，现在张志华老师已经去了北大。

3  斯坦福大学 Andrew Ng（吴恩达）: Stanford CS229 2017，非常有名，不介绍了。

4  悉尼科技大学徐亦达的《机器学习》：阐述一些列概率模型，EM、MCMC、Calman Filter，粒子滤波，狄利克雷过程。GitHub上有笔记，很全！

5  台湾大学李宏毅的《机器学习》：CNN、DNN；《MLDS》：优化、正则化、实践优化、自然语言处理等。

### 符号约定

我们先规定一些符号：$\mathbf{X}$ 表示数据（data），是一个样本矩阵，每一行表示一个样本（随机变量），$\theta$ 表示参数（parameter），多数情况下是一个向量。

$$\mathbf{X}=(X_1\ X_2\ \cdots \ X_N)^{\mathrm{T}}=\begin{pmatrix} X_1^{\mathrm{T}}\\ X_2^{\mathrm{T}} \\ \vdots \\ X_N^{\mathrm{T}} \\ \end{pmatrix}\\ = \begin{pmatrix} x_{11} & x_{12} & \cdots &x_{1p}\\ x_{21} & x_{22} & \cdots &x_{2p}\\ \vdots & \vdots & \ddots & \vdots \\ x_{N1}& x_{N2} & \cdots & x_{Np} \\ \end{pmatrix}\\$$ 

其中，$\mathbf{X} \in \mathbb{R}^{ N \times p}$， $X_i \in \mathbb{R}^{ p \times 1}, i = 1,2,\cdots,N$。我们用大写粗体的 $\mathbf{X}$ 表示由 $N$ 个随机变量组成的矩阵，用 大写细体的 $X$ 表示随机变量，用小写细体的 $x$ 表示随机变量的具体取值，$x$ 可以是标量或向量，都用相同类型的字母表示，除特别声明外，本书中的向量均为列向量，$x$ 的特征向量记作：

$$(x^{(1)}\ x^{(2)}\ \cdots \ x^{(n)})^{\mathrm{T}}$$

$x^{(i)}$ 表示 $x$ 的第 $i$ 个特征，注意，$x^{(i)}$ 与 $x_i$ 不同，后者表示多个随机变量的第 $i$ 个取值，即，

$$(x_i^{(1)}\ x_i^{(2)}\ \cdots \ x_i^{(n)})^{\mathrm{T}}$$

若 $X$ 服从于一个概率分布，记为 $ X\sim P(X \mid \theta)$，这里我们用大写的 $P(·)$ 表示概率分布，用小写的 $p(·)$ 表示概率密度函数或离散分布律。此处，当 $\theta$ 为参数时， 以下两种表示方式等价： $P(X \mid \theta) \iff P(X;\theta)$ 。今后若不特殊说明，我们都用左侧的表示方式。

今后的符号都依照以上规则。

### 频率派 VS 贝叶斯派

#### 频率派

$\theta$ 为未知常量，$X$ 为随机变量，在这里我们要估计的是 $\theta$。最常用的方法是极大似然估计（Maximum Likelihood Estimation），

$$\theta_{MLE}=\underset{\theta}{\operatorname{argmax}}\; \log  \underbrace{P(X \mid \theta)}_{L(\theta)}$$ 

$L(\theta)$ 是似然函数，$L(\theta)=P(X \mid \theta)=\prod_{i=1}^N p(x_i \mid \theta) $， 加 $\log$ 是为了简化运算，利用对数的运算性质，将连乘变为连加，$\prod \to \sum$ ，即 $\log P(X \mid \theta)=\sum_{i=1}^N p(x_i \mid \theta)$。我们的目的是求一个 $\hat{\theta}$ 使得 $P(X \mid \hat{\theta})$ 最大。

所以，以频率派的视角，最终要解决的是**优化问题**。

#### 贝叶斯派

$\theta$ 为随机变量，$\theta \sim p(\theta)$，是先验概率分布（prior）。在这种情况下，我们最终要求的是后验概率 $P(\theta \mid X)$，由贝叶斯公式可得：

$$P(\theta \mid X)= \dfrac{P(X \mid \theta)P(\theta)}{P(X)} \propto P(X \mid \theta)P(\theta)$$

其中，$P(\theta \mid X)$ 为后验概率（posterior）也就是我们要求的，$P(X \mid \theta)$ 为似然（likelihood），$P(\theta)$ 为先验（prior），$P(X)$ 是 $P(X, \theta)$ 的边缘分布，依据边缘概率的求法， $P(X)=\int_{\theta} {P(X \mid \theta)P(\theta)} \,{\rm d}\theta$ 是可以算出来的，可以认为是一个常值。

因此，我们使用最大后验概率（Maximum  A  Posteriori）：

$$\theta_{MAP}=\underset{\theta}{\operatorname{argmax}}\; P(X \mid \theta)P(\theta)$$

**贝叶斯估计：**

$$P(\theta \mid X)= \dfrac{P(X \mid \theta)P(\theta)}{\int_{\theta} {P(X \mid \theta)P(\theta)} \,{\rm d}\theta}$$

**贝叶斯预测：**

已知 $X$，预测 $\widetilde{X}$，首先，我们通过 $X$ 去预测 $\theta$，再通过 $\theta$ 预测 $\widetilde{X}$，因此：

$$P(\widetilde{X} \mid X)=\int_{\theta} {P(\widetilde{X},\theta \mid X)} \,{\rm d}\theta =\int_{\theta} {P(\widetilde{X} \mid \theta) \underbrace{P(\theta \mid X)}_{posterior}} \,{\rm d}\theta$$

所以，以贝叶斯派的视角，最终要解决的是**求积分问题**。

---

