---
layout: post
title: Stanford CS224n 自然语言处理（一）
date: 2019-08-04 15:04:00
author: "chenjing"
header-img: "img/post-bg-unix-linux.jpg"
catalog: true
tags:
  - NLP
  - Stanford
  - 深度学习
---

> Begining

## 前言

我是在4月初入坑自然语言处理的，并在4月初到6月初的2个月时间里面学习了斯坦福的自然语言处理课程。接下来我会大致介绍一下这门课程，讲讲每节课的大纲，以及我觉得缺少的东西。

首先给出课程的[官方网站](http://web.stanford.edu/class/cs224n/)，所有的课程资料都可以在这个网页上下载，包括：课程ppt、作业说明文档、手册、部分笔记等。配套的网络课程在[这里](https://www.bilibili.com/video/av46065585?from=search&seid=12274272452479951197)，如果你希望系统地学习 NLP，那么这门课程是你的首选。(6月下旬斯坦福又开放了一门课程 CS224u 自然语言理解，我会另开一个系列介绍)

---

## 课程大纲

#### Lecture 1 : Introduction and Word Vectors

首先介绍了课程的教授者，课程资料以及作业的评分细则，然后列出了和2017年的 CS2224n 课程的不同点：

* 课程涵盖新的模型和方法，如，字符级模型( Character Model )、Transformer、多任务学习( Multitask Learning )

* 作业涵盖了新内容，如，NMT with attention、卷积网络( ConvNets )、子词模型( subword model )
* 使用 PyTorch 而不是 TensorFlow

接着介绍了自然语言以及词义；词向量、word2vec 概览；word2vec 的目标函数梯度的计算以及梯度下降算法。



#### Lecture 2 : Word Vectors and Word Senses

1. 实现 word2vec 的两种模型
   * Skip-grams (SG) 
   *  Continuous Bag of Words (CBOW)
2. 训练技巧
   * 负采样 (Negative sampling) 
   * Hierarchical softmax
3. 共现矩阵( co-occurrence matrix )，以及降维方法( SVD )
4. GloVe
5. 词向量的评价指标
6. 词义：歧义( word sense ambiguity )、一词多义( polysemy )等



#### Lecture 3 : Word Window Classification, Neural Networks and Matrix Calculus

1. 分类问题概览
2. 神经网络简介
3. 命名实体识别( Named Entity Recognition )
4. 词窗口
5. 矩阵求导



#### Lecture 4 : Backpropagation and computation graphs

1. 推导矩阵反向传播的梯度
2. 计算图( computation graphs )
3. 神经网络的一些需要知道的问题：
   * 向量化表达( Vectorization ) / 矩阵化表达( Matrixization )的计算速度优势
   * 过拟合( overfitting )
   * 非线性( nonlinearities )
   * 初始化( initialization )
   * 优化器( optimizers )
   * 学习率( learning rates )



#### Lecture 5 : Dependency Parsing

1. 两种句法结构：
   * 一致性( Consistency ) ，即短语结构语法，也即无上下文语法( context-free grammars )，18课会单独讲
   * 依存性( Dependency )，即依存结构
2. 依存语法和依存结构
3. 依存关系解析( Dependency Parsing )
   * 基于转换的解析( transition-based )
   * 基于神经网络的解析( nueral )



#### Lecture 6 : Language Models and Recurrent Neural Networks

1. 语言模型简介
2. n-gram 语言模型
3. 神经网络（RNN）语言模型
4. 评估语言模型



#### Lecture 7 : Vanishing Gradients and Fancy RNNs

1. 梯度消失问题和梯度爆炸问题及其解决方法
2. LSTM （ Long Short-Term Memory ）
3. GRU（ Gated Recurrent Unit ）
4. RNN variant：
   * Bidirectional RNN
   * Multi-layer RNN



#### Lecture 8 : Machine Translation, Sequence-to-sequence and Attention

1. 统计机器翻译（ SMT ）
   * 对齐（alignment）
   * 解码（decodeing）
2. 神经机器翻译（ NMT ）
   * seq2seq
   * beam search decoding
3. Attention 机制



#### Lecture 9 : Practical Tips for Final Projects

1. 最终项目的一些建议
2. 回顾 GRU、LSTM
3. 机器翻译的评价指标：BLEU



#### Lecture 10 : (Textual) Question Answering

1. QA —>机器阅读理解
2. SQuAD 数据集
3. 斯坦福 Attentive Reader
4. BiDAF
5. 一些先进的模型：Co-attention、FusionNet 等
6. ELMo 和 BERT 概览



#### Lecture 11 : ConvNets for NLP

1. 从 RNN 到 CNN
2. 模型比较 



#### Lecture 12 : Information from parts of words : Subword Models

1. 语言学概览
   * 发音（Phonetics）
   * 音韵（Phonology）
   * 词的形态（Morphology）
   * 书写系统

2. 字符级的模型
3. 子词模型
   * Byte Pair Encoding
   * Wordpiece
   * Sentencepiece

4. 混合神经网络翻译
5. fastText



#### Lecture 13 : Contextual Word Representations and Pretraining

1. 词表示
2. pre-ELMo & EMLo
3. ULMfit 
4. Transformer 架构
   * self-attention
5. BERT



#### Lecture 14 : Self-Attention for Generative Models

1. Self-Attention
2. 文本生成
3. 图像生成
4. 音乐生成



#### Lecture 15 : Natural Language Generation

1. 回顾语言模型和解码算法及技巧
   * Greedy decoding
   * Beam search
   * Sampling methods
   * Softmax temperature
2. 自然语言生成（NLG）任务以及神经网络方法
   * 文本总结
   * 对话系统
   * 创意写作

3. NLG 的评估
4. NLG 的趋势和发展



#### Lecture 16 : Coreference Resolution

1. 共指解析及其应用
2. 提及检测（Mention Detection）
3. 一些语言学概念
   * anaphora
   * cataphora

4. 共指模型
   * Rule-based Model
   * Mention Pair Model
   * Mention Ranking Model
   * Mention Clustering Model

5. 评估指标



#### Lecture 17 : Multitask Learning as QA

这节课相当于一个讲座，介绍了多任务学习的若干模型、训练策略，以及多任务学习的优势



#### Lecture 18 : Tree Recursive Neural Networks, Constituency Parsing, and Sentiment

1. 递归神经网络（Recursive Neural Networks）
2. 短语句法解析
3. 树递归神经网络（TreeRNN）



#### Lecture 19 : Bias in the Vision and Language of Artificial Intelligence

讲座，介绍了 AI 的偏见以及多任务学习



#### Lecture 20 : The Future of Deep Learning + NLP

讲座，介绍了大规模深度学习在各个领域取得的成就，无标签数据集，无监督机器翻译，大模型（如 BERT、GPT-2），多任务学习，更加难的自然语言理解，NLP 在工业上的应用，以及未来的挑战。

---

## 后记

以上就是Stanford CS224n 自然语言处理课程的课程概览，我会在今后陆续完成每节课的分析与笔记（如果有时间的话 hhh）。

自然语言处理真的是一门非常庞大的学科，其中包含了很多内容，想要在一门课上学习完所有的内容是不现实的，个人所见，这门课的的意义就在于为刚开始学习自然语言处理的学生做一个相对详细的 review ，接下去要做什么方向，按照自己的兴趣来。

在前言中我提到过，这门课程还有许多没有提及的方面，比如分词方法、TF-IDF、关系提取、情感分析等，如果对这些感兴趣的话，Stanford 为大家又提供了一门课程（[CS224u 自然语言理解](http://web.stanford.edu/class/cs224u/)），感谢斯坦福。

写这些文章的初衷是为了对抗遗忘，所以某种程度上是给自己温故用的。因此，我不会把课上的内容完完整整地记录下来，如果有什么疑问，欢迎指出。

**PS** ：博客的评论区还没有调试成功，如果我的文章有什么错误，烦请点击下面的微博按钮，私信给我，不胜感激。

---

