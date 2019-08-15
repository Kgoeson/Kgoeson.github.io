---
layout: post
title: Stanford CS224n 自然语言处理（二）
date: 2019-08-15 20:00:00
author: "chenjing"
header-img: "img/post-bg-unix-linux.jpg"
catalog: true
mathjax: true
tags:
  - NLP
  - Stanford
  - 深度学习
---

> You shall know a word by the company it keeps.						—— J. R. Firth

# 0 前言

这篇文章涵盖 CS224n 的 Lecture 1 和 Lecture 2 中的部分内容。抛却开头的课程介绍，我们直入正题。主要内容包括：自然语言以及词义；词向量；共现矩阵( co-occurrence matrix ) ；降维方法( SVD )；word2vec 概览；实现 word2vec 的两种模型：Skip-Gram 和 Continuous Bag-of-Word。

因为是第一篇正式介绍内容的文章，概念有点多，中途遇到看不懂的地方可以先略过，可能在后面会有解释，再返回去看，就会串起来。

# 1 自然语言处理介绍

## 1.1 自然语言

什么是自然（人类）语言？狭义地来说，自然语言通常是指一种自然地随文化演化的语言，即我们所知的汉语、英语、日语等。广义地来说，自然语言还包括人造语言，比如世界语。

> 此处的“人造语言” 世界语，区别于编程语言等人造语言。

人类语言是一个专门用来表达意义（meaning）的系统，而不是由任何物理表现产生的。以这种方式来说，它与视觉或任何形式的机器学习任务都不同。也就是说，图像是一种物理表现，这很好理解，日常生活中的照片，画册，亦或是计算机中的 RGB 色图，就是图像的物理表现，脱离了这种传递信息的介质，图像便不能很好地传递信息。而自然语言可以脱离这种方式，以自身直接传递信息，一般来说，你只需要通过若干个单词的描述，就能够准确无误地表达意义。

大多数单词只是一个超语言实体（extra-linguistic entity）的符号：单词是一种符号，该符号映射到一个想法或事物。这种映射也称作指代语义（Denotational semantics）。

**signifier ( symbol )  $\Longleftrightarrow$  signified ( idea or thing )**

> **signifier:** a sign's physical form (such as a sound, printed word, or image) as distinct from its meaning.
>
> **signified:** the meaning or idea expressed by a sign, as distinct from the physical form in which it is expressed.

举个例子。**signifier ( symbol )  $\Rightarrow$  signified ( idea or thing )** ，就是在纸上写着“椅子”（符号），你就能联想到具体的1号椅子，2号椅子...。

**signifier ( symbol )  $\Leftarrow$  signified ( idea or thing )**，指着具体的椅子给你看，你能抽象出椅子这一记号。

还有，这些语言的符号可以被编码成几种形式：声音、手势、文字等。然后通过*连续*信号传输给大脑，大脑本身似乎也能以一种连续的方式编码这些信号。

那么，我们现在已经知道我们要 *“处理”* 的对象了，那么该如何处理呢？这就是 CS224n 这个系列的主题，接下去的一系列文章将会带你领略 **自然语言处理** 这门融语言学、计算机科学、数学于一体的学科。

## 1.2 自然语言处理的任务

自然语言处理的 目标 是**通过设计算法来使计算机能够“理解”语言，从而能够执行某些特定的任务**，而不同的任务的难度是不一样的：

**简单：**

* 拼写检查
* 关键词搜索
* 同义词查找



**中等：**

* 解析来自网站文档等的信息



**困难：**

* 机器翻译
* 语义分析（例如，“陈述”这个词是什么意思？）
* 共指（例如，“他”和“它”在文档中分别指代什么？）
* 问答系统

# 2 词的表示

## 2.1 如何表示词

那么我们该如何使计算机明白一个词、一句话，乃至一篇文章的意思呢？

我们先举个实际例子，大家或许都知道 WorldNet ，他是自然语言处理工具包 NLTK 包含一个同义词集（synonym sets）和上位词（hypernyms）的词库。

> **hypernyms :**  “is a” 的关系。如，Rose is a flower。Flower is a plant。即 ”花” 是 ”玫瑰” 的上位词，”植物” 是 ”花” 的上位词。

![](/img/in-post/post-cs224n/worldnet_demo.png)

可以看到它是一个可以很好地进行同义词比对和上位词查找的词库。那么，像这种基于人工统计的词库有没有缺点呢？答案是肯定的。

首先，它不能判别词的细微差别，如，“proficient” 是 “good” 的同义词，但是在一些文章中并不是。第二，它缺少了很多新词，如，badass, nifty, wizard, ninja 等，再进一步，通过 WorldNet 的组织形式可以看出，要加入新词几乎是不可操作的。第三，由于是人工统计的，显得太主观。第四，需要大量的人力和物力去创建和维护。还有，它不能定量地计算词与词之间的相似度。

为了让大多数的自然语言处理任务能有更好的表现，我们先需要了解单词之间的相似和不同，于是词向量应运而生。

## 2.2 词向量

在所有的 NLP 任务中，最重要的是我们如何将单词表示为任意模型的输入。而模型的输入是一组数字，于是很自然地想到用向量来表示一个词，即把单词映射为实数向量，这种向量就叫做词向量。

> **词向量（word vector）** 有时又被称作**词嵌入 （word embedding）** 或者**词表示（word representation）**。

一共有1千3百万个左右的英语单词，它们其中很多都是有关系的。例如 “feline” 和 “cat”，“hotel” 和 “motel” 。因此，我们希望用词向量编码单词使它嵌入到词组空间中，代表其中的一个点（这也是词向量称为词嵌入的原因）。这样做最直观的原因是，在实际中可能存在 $N$ 维空间（$N$ $\ll$ $13 million$）足以编码所有单词以及语义。每个维度都会编码一些使用言语传达的意思。例如，语义维度可能表示时态（过去、现在和未来），计数（单数和复数）和性别（男性和女性）。

> 这里先提前区别几组名词：
>
> **distributed representation** (密集型表示) 与 **symbolic representation**（localist representation、one-hot representation）相对。
>
> **discrete representation** (离散表示) 与 **symbolic representation** (符号表示) 及 **denotation** 的意思相似。
>
> 切不可搞混 **distributed** 和 **discrete** 这两个词。

我们从最简单的开始，传统的 NLP 的做法是将词看作是一组离散的符号，将词表示成 **one-hot 向量**：每个词都是一个 $\mathbb{R}^{\mid V\mid \times 1}$ 向量，其中除了该单词所在的索引为 1 外其他索引都是 0。在这个定义下， $\mid V\mid$ 是词汇表的大小，即词表中词的个数。这时词向量的可以表示为

$hotel$  = [ 0 0 0 0 1 0 0 ··· 0 ]

$motel$ = [ 0 1 0 0 0 0 0 ··· 0 ]

  $cat$    = [ 0 0 0 1 0 0 0 ··· 0 ]

但是，这样的表示无法给出词之间的相似性，因为根据相似度的一种定义（内积）：

$$\left(w^{\text { hotel }}\right)^{T} w^{\text { motel }}=\left(w^{\text { hotel }}\right)^{T} w^{\text { cat }}=0$$

很不可思议吧，$hotel$ 与 $motel$ 的相似度竟然等于 $hotel$ 和 $cat$ 的相似度。这是因为维数越大，数据越稀疏，比较相似度往往缺乏实际意义。

那么接下来很明确，就是降低维度，使数据变得稠密，找到一个更低维度的向量空间来编码词与词之间的关系。

## 2.3 共现矩阵

在介绍降维的方法之前，我先引入共现矩阵（co-occurrence）的概念。共现矩阵也叫共现计数矩阵，是基于统计方法得到的矩阵，记作 $X$。

### 2.3.1 基于文档的共现矩阵

我们猜想，有关联的词经常会出现在同一个文档中，例如，“banks”，“stocks”，“shares” 等，出现在同一篇的文档的概率较高，而 “banks”，“banana”，“phone”出现在同一篇文档的概率较小。根据这种情况，我们可以建立一个词-文档矩阵 $X$ ，$X$ 按照以下方式构建：遍历文档，当词 $i$ 出现在文档 $j$ ，我们对 $X_{ij}$ 加一，遍历结束，我们便可得到一个词-文档矩阵。但这显然是一个很大的矩阵$X\in\mathbb{R}^{\mid V\mid \times M}$ ，它的规模是和文档数 $M$ 成正比的，而 $M$ 通常非常大（$billion$级），因此，不考虑这种方式。

### 2.3.2 基于窗口的共现矩阵

使用基于窗口的方式，我们直接通过单词与单词之间的共现来计算共现矩阵。在这种方法中，我们统计每个单词在感兴趣单词的附近特定大小的窗口中出现的次数。我们按照这个方法对语料库中的所有单词进行统计。当窗口大小为2时，窗口是这样移动的，直至最后一个单词。

![](/img/in-post/post-cs224n/window_based.png)

举个例子，语料库由三句话组成，窗口的大小是 1（考虑中心词左右两侧的第一个单词）：

I enjoy flying.

I like NLP.

I like deep learning.

共现矩阵如下：

![](/img/in-post/post-cs224n/co_occurrence_matrix.png)

## 2.4 SVD

这里先提两个概念：

>  **Denotational semantics:** The concept of representing an idea as a symbol (a word or a one-hot vector). It is sparse and cannot capture similarity. This is a "localist" representation.

> **Distributional semantics:** The concept of representing the meaning of a word based on the context in which it usually appears. It is dense and can better capture similarity.

简而言之，就是 **Denotational semantics** 是用符号表示词，稀疏，无法获得相似性，而 **Distributional semantics** 是用上下文表示词，稠密，可以很好地获取相似性。我们在上面说的 one-hot 向量表示词就是一种 **Denotational semantics**，而以下要说的 SVD 方法就是一种  **Distributional semantics**。

SVD即奇异值分解，是矩阵分解的一种。SVD 方法是一种找到词向量的方法，首先遍历一个很大的数据集和统计词的共现计数矩阵 $X$，然后对矩阵 $X$ 进行 SVD 分解得到 $USV^{T}$ （$U,S,V\in\mathbb{R}^{\mid V\mid \times \mid V\mid}$）。通过选择前 $k$ 个奇异值来降低维度，然后使用 $U_{1:\mid V\mid,1:k} \in \mathbb{R}^{\mid V\mid \times k}$ 的行向量作为词汇表中所有词的词向量，词向量的维度为 $k$。

SVD 方法能让我们的词向量编码充分的语义和句法（词性标注）的信息，但是也会存在许多问题：

* 共现矩阵的维度会经常发生改变（经常增加新的单词和语料库的大小会改变）。

- 共现矩阵会非常的稀疏，因为很多词不会共现。
- 共现矩阵的维度一般会非常高（$ \approx 10^{6} \times 10^{6}$ ）。
- 基于 SVD 的方法的计算复杂度一般为 $O\left(mn^{2}\right)$ 。
- 需要在 $X$ 上加入一些技巧处理来解决词频的不平衡。

对上述讨论中存在的问题有以下的解决方法：

- 忽略功能词，例如 “the”，“he”，“has” 等等。

- 使用 ramp window ，即根据中心词与上下文词之间的距离远近，赋予共现计数不同的权重。

  > 如，和中心词最相邻的上下文词的计数权重为1，相隔5个位置的计数权重为0.5。

- 使用皮尔逊相关系数将负数的计数设为 0，而不是使用原始的计数。

在下一部分，基于迭代的方法可以用一种更为优雅的方式解决大部分上述问题。

# 3 word2vec

我们回顾一下文章开头的话：You shall know a word by the company it keeps。这是 J. R. Firth 大佬说的，他是现代统计自然语言处理最成功的思想之一。简单来说，就是我们要到具体的上下文当中去理解一个词。上一节中提出的共现矩阵其实就是一种捕捉单词上下文词信息的方法。

![](/img/in-post/post-cs224n/context_word.png)

**word2vec** 是一种从大量文本语料中学习语义知识的模型，它被大量地用在自然语言处理（NLP）中。它的模型参数就是词向量，用词向量表征词的语义信息，词向量可作为下游任务的输入。

word2vec 模型实际上分为了两个部分，第一部分为建立模型，第二部分是通过模型获取嵌入词向量。word2vec的整个建模过程实际上与自编码器（auto-encoder）的思想很相似，即先基于训练数据构建一个神经网络，当这个模型训练好以后，我们并不会用这个训练好的模型处理新的任务，我们真正需要的是这个模型通过训练数据所学得的参数，例如隐层的权重矩阵——后面我们将会看到这些权重在 word2vec 中实际上就是我们试图去学习的“词向量”。基于训练数据建模的过程，我们给它一个名字叫“Fake Task”，意味着建模并不是我们最终的目的。

word2vec 模型主要有 Skip-Gram 和 Continuous Bag-of-Word 两种模型，从直观上理解，Skip-Gram 是给定中心词（center word）来预测上下文词（context word），而 CBOW 是给定上下文词（context word），来预测中心词（center word）。

![](/img/in-post/post-cs224n/CBOW_arc.png)

<center>CBOW 模型</center>



![](/img/in-post/post-cs224n/SG_arc.png)

<center>Skip-Gram 模型</center>

## 3.1 Language Model

在正式介绍 word2vec 模型之前，我们需要了解一些语言模型（language model）相关的概念，我们从一个简单的例子开始：

“ The cat jumped over the puddle. ”​

一个好的语言模型会给这个句子很高的概率，因为在句法和语义上这是一个完全有效的句子。相似地，句子 “ stock boil fish is toy. ” 会得到一个很低的概率，因为这是一个无意义的句子。在数学上，我们可以称给定 n 个词的序列的概率是：

$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)$$

我们可以使用 unigram 语言模型方法，通过假设单词的出现是符合独立同分布假设的：

$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=1}^n P\left(w_{i}\right)$$

但是我们知道这是不合理的，因为下一个单词是高度依赖于前面的单词序列的。如果使用上述的语言模型，可能会让一个无意义的句子具有很高的概率（只需要选取那些出现频率高的单词组成一个句子）。所以我们可以让序列的概率 等于 序列中每个单词和其旁边的单词组成单词对的概率的乘积。我们称之为 bigram 模型：

$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=2}^{n} P\left(w_{i} | w_{i-1}\right)$$

虽然这个方法会有提升，但还是过于简单，不能捕获完整的语义信息，因为我们只关心一对邻近的单词，而不是针对整个句子来考虑。考虑在词-词共现矩阵中，共现窗口为 1，我们基本上能得到这样的成对的概率。但是，这需要计算和存储大量数据集的全局信息。

类似的方法还有 trigram 模型、four-gram 模型等，统称为 n-gram 模型。这些模型的计算非常复杂，在实际情况中最多取 four-gram，原则上能用 trigram 解决的问题绝对不用 four-gram。

现在我们知道该如何计算一个序列的概率，接下来就是 word2vec 中一些可以计算这些概率的模型。使用迭代的方法简化模型使得训练模型速度非常快，而不是像 n-gram 模型一样去计算和维护一个庞大数据集得到全局信息。

## 3.2 The Fake Task

### 3.2.1 任务介绍

我们在介绍 wordvec 的时候提到，训练模型的真正目的是获得模型基于训练数据学得的隐层权重。为了得到这些权重，我们首先要构建一个完整的神经网络作为我们的“Fake Task”，后面再返回来通过“Fake Task”间接地得到这些词向量。

接下来我们来看看如何训练我们的神经网络。假如我们有一个句子： $\text {“The dog barked at the mailman”}$。

- 首先我们选句子中间的一个词作为我们的输入词，例如我们选取“dog”作为中心词；

  > 这里做一个规定：对于 CBOW 模型，input word 是上下文词，output word 是中心词，而对于 Skip-Gram 模型，input word 是中心词，output word 是上下文词。标记有点多，千万不要混淆。

- 我们再定义一个 window_size 参数，它代表着我们从当前中心词的一侧（左边或右边）选取词的数量。如果我们设置 window_size = 2 ，那么我们最终获得窗口中的词（包括中心词在内）就是['The', 'dog'，'barked', 'at']。window_size = 2 代表着选取中心词左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小 span = 4。另一个参数叫 num_skips，它代表着我们从整个窗口中选取多少个**不同的词**作为我们的 output word，当 window_size = 2，num_skips = 2 时，我们将会得到两组 (input word, output word) 形式的训练数据，即 ('dog', 'barked')，('dog', 'the')。

- 神经网络基于这些训练数据将会输出一个概率分布，这个概率代表在给定 input word 的情况下词表中的每个词出现的概率。再用交叉熵损失将这个概率分布与 output word 的 one-hot 向量的误差反向传播。

模型的输出概率代表词表中每个词有多大可能性跟 input word 同时出现。举个栗子，如果我们向神经网络模型中输入一个单词“Soviet“，那么最终模型的输出概率中，像“Union”， ”Russia“这种相关词的概率将远高于像”watermelon“，”kangaroo“非相关词的概率。因为”Union“，”Russia“更有可能在”Soviet“的窗口中出现。
我们将通过给神经网络输入文本中成对的单词来训练它，完成上面所说的概率计算。下面的图中给出了一些我们的训练样本的例子。

假设有句子：$\text{ “The quick brown fox jumps over lazy dog. ”}$ ，设定我们的窗口大小为2（window_size = 2）。下图中，蓝色代表input word，方框内代表位于窗口内的单词。

![](/img/in-post/post-cs224n/training_data.png)

模型将会从每对单词出现的次数中学习得到统计结果。例如，我们的神经网络可能会得到更多类似（“Soviet“，”Union“）这样的训练样本对，而（”Soviet“，”Sasquatch“）这样的训练样本对却很少。因此，当我们的模型完成训练后，给定一个单词”Soviet“作为输入，输出的结果中”Union“或者”Russia“要比”Sasquatch“的概率大得多。

### 3.2.2  任务实施

**1  输入层**

我们首先从最简单情形开始，input word 只有一个，即给定一个 input word，预测 output word（有点像 bigram 模型）。假设词表大小为$V$，隐藏层神经元个数为$N$，输入层到隐藏层、隐藏层到输出层都是全连接，样本是（input word，output word）单词对，输入为 input word 的 one-hot 向量。

假设从我们的训练文档中抽取出 10000 个不重复的单词组成词汇表。我们对这 10000 个单词进行 one-hot 编码，得到的每个单词都是一个 10000 维的向量，向量每个维度的值只有 0 或者 1，假如单词 “ants“ 在词汇表中的出现位置为第 3 个，那么 ‘’ants“ 的向量就是一个第三维度取值为 1，其他维都为 0 的 10000 维的向量（$\text {ants}=[0,0,1,0, \ldots, 0]$ ）

模型的输入如果为一个 10000 维的向量，那么输出也是一个 10000 维度（词汇表的大小）的向量，它包含了 10000 个概率，每一个概率代表着给定 input word 词表中的每个词出现在 input word 附近的概率大小。

下图是我们神经网络的结构：

![](/img/in-post/post-cs224n/one_word_net_arch.png)

隐层没有使用任何激活函数，但是输出层使用了sotfmax。

我们使用成对的单词来对神经网络进行训练，训练样本是 ( input word, output word ) 这样的单词对，input word 和 output word 都是 one-hot 编码的向量。最终模型的输出是一个概率分布，每一个输出神经元都是一个概率值，表示在给定 input word 的情况下词表中每个词出现的概率，即 $p\left(w_{j} \mid w_{I}\right)$，$w_{I}$ 表示 input word，$w_{j}$，表示词表中的词，$\text j=1,2, \ldots, V$ 。

**2  隐藏层**

说完单词的编码和训练样本的选取，我们来看下我们的隐层。如果我们现在想用300个特征来表示一个单词（即每个词可以被表示为300维的向量）。那么隐层的权重矩阵应该为10000行（词汇表中的每个单词），300列（每个隐藏神经元）。

> Google在最新发布的基于Google news数据集训练的模型中使用的就是300维的词向量。词向量的维度是一个可以调节的超参数，可以根据不同的任务调节，实践中一般300维为佳。

看下面的图片，左右两张图分别从不同角度代表了输入层-隐层的权重矩阵。左图中每一列代表一个10000维的词向量和隐层单个神经元连接的权重向量。从右边的图来看，每一行实际上代表了每个单词的词向量。

![](/img/in-post/post-cs224n/word2vec_weight_matrix_lookup_table.png)

现在我们已经完成了“Fake Taks”，我们最终的目标就是学习到隐层的权重矩阵。

我们现在回来，接着通过模型的定义来训练我们的这个模型。

上面我们提到，input word 和 output word 都会被我们进行 one-hot 编码。仔细想一下，我们的输入被 one-hot 编码以后大多数维度上都是0（实际上仅有一个位置为1），所以这个向量相当稀疏，那么会造成什么结果呢。如果我们应用**矩阵乘法**将一个1 x 10000的向量和10000 x 300的矩阵相乘，它会消耗相当大的计算资源，为了高效计算，它仅仅会选择矩阵中对应的向量中维度值为1的索引行，看图就明白。

![](/img/in-post/post-cs224n/matrix_mult_w_one_hot.png)

为了有效地进行计算，这种稀疏状态下不会进行矩阵乘法计算，可以看到计算的结果实际上是矩阵的某一行：先根据 input word 的 one-hot 向量中元素 1 的索引，再由这个索引取得矩阵对应的行。上面的例子中，左边向量中取值为 1 的对应维度为 3（下标从0开始），那么计算结果就是矩阵的第 3 行（下标从0开始）—— [10, 12, 19]，这样模型中的隐层权重矩阵便成了一个”查找表“（lookup table），进行矩阵计算时，直接去查输入向量中取值为1的维度下对应的那些权重值。隐层的输出就是每个输入单词的“嵌入词向量”，也称为输入词向量。

> 上面提到的隐层权重矩阵是从输入层到隐藏层的权重矩阵，称为**输入词向量矩阵**，在接下来的章节中我们还会看到，从隐藏层到输出层的权重矩阵，称为**输出词向量矩阵**。

**3  输出层**

经过神经网络隐层的计算，“ants“ 这个词会从一个 1 x 10000 的向量变成 1 x 300 的向量，再被输入到输出层。输出层是一个softmax回归分类器，它的每个结点将会输出一个 0-1 之间的值（概率），这些所有输出层神经元结点的概率之和为1。

下面是一个例子，训练样本为 (input word: “ants”， output word: “car”) 的计算示意图：

![](/img/in-post/post-cs224n/output_weights_function.png)

> 乘号右侧的 output weights for “car” 即是 “car” 的输出词向量，“car”在作为 input word 时也有相应的输入词向量。

直觉上的理解，如果两个不同的单词有着非常相似的“上下文”（也就是窗口单词很相似，比如“Kitty climbed the tree”和“Cat climbed the tree”），那么通过我们的模型训练，这两个单词的嵌入向量将非常相似。

> 在word2vec 中，规定两个单词的**相似度**就是各自词向量的内积，如，(input word: “ants”， output word: “car”) 的计算结果就是“ants”和“car”的相似度。

那么两个单词拥有相似的“上下文”到底是什么含义呢？比如对于同义词“intelligent”和“smart”，我们觉得这两个单词应该拥有相同的“上下文”。而例如”engine“和”transmission“这样相关的词语，可能也拥有着相似的上下文。

实际上，这种方法实际上也可以帮助你进行词干化（stemming），例如，神经网络对”ant“和”ants”两个单词会习得相似的词向量。

> 词干化（stemming）就是去除词缀得到词根的过程。

以上便是 word2vec 模型的最简化形式，CBOW 和 Skip-Gram 无非就是对这个模型的推广，到这里，大家应该在直观上理解了这个模型，如若要深究下去（数学推导👀），我会在下一篇文章中具体分析。

# 4 小结

这篇文章从自然语言开始讲起，通过如何表示一个词，讲述了从词的离散表示到稠密表示的发展过程，引入了共现矩阵、词向量、SVD、word2vec 等方法将一个具体的单词（符号）表示成可以喂给任意模型处理的数据。在下一篇文章中，我将介绍 word2vec 的两种具体的模型：CBOW 和 Skip-Gram，以及它们各自的目标函数梯度的计算和梯度下降算法，还有它们的训练方法：负采样 (Negative sampling)和 Hierarchical softmax。
