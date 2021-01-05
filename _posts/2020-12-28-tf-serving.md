---
layout: post
title: 基于TF Serving部署TensorFlow模型
description: ""
modified: 2020-10-01
tags: [Deep Learning, Seq2Seq]
categories: [算法]
image:
  path: /images/tensorflow/tf_serving/tf_serving_2.png
  feature: /tensorflow/tf_serving/tf_serving_2.png
  credit: x1
  creditlink: https://deeplearn.org/arxiv/69226/a-comparative-study-on-hierarchical-navigable-small-world-graphs
---
目录

* TOC 
{:toc}

# 1. TensorFlow Serving的基本结构
Tensorflow Serving是一个针对生产环境设计的灵活、
高性能的机器学习模型服务系统。
TensorFlow Serving使得我们能够在保持相同服务架构和API
的情况下很容易的部署新算法和实验。

TensorFlow Serving针对TensorFlow模型提供开箱即用的集成，
同时又能很容易地扩展到其他类型的模型服务。

## 1.1 核心概念

为了更好的理解TensorFlow Serving，我们需要了解以下的一些基本概念。

**1. Servables**

Servables是TensorFlow Serving中的核心抽象，
是客户端用来执行计算(例如:一个lookup操作或者是一个inference)的底层对象。。

Servable具有非常灵活的大小和粒度，可以包含任何东西。
Servable可以是一个lookup table操作的切片，
也可以包含单独一个模型推断的tuple。
为了确保灵活性和未来的可扩展性，Servables可以是任何一种类型和接口。
例如：
* streaming results
* experimental APIs
* asynchronous models of operation

Servables不管理自身的生命周期。

典型的Servables包含以下几种:

* 一个TensorFlow SavedModelBundle(Tensorflow:：Session)

* 一个用于embedding或者vocabulary查询的lookup table

**2. Servable Version**

在单个服务的实例生命周期中，
TensorFlow Serving可以处理一个或者多个版本的Servable。
也就是说在这段时间内可以加载新算法、配置、权重、其他数据。
Servable Version使得TensorFlow Serving可以并行加载多个模型，
支持渐进式回滚和实验。在服务的时候，client可以请求最新版本
或者通过模型的id指定特定的模型版本。

**3. Servable Streams**

一个Servable Streams是一个Servable的version序列，通过递增的版本号进行排序。

**4. Models**

TensorFlow Serving中的一个Models包含一个Servable或者多个Servable。
一个机器学习模型可以包含一个或者多个算法(包含学习到的权重)以及lookup或者
embedding tables。

在TensorFlow可以将一个复合模型表示成以下两种方式之一:

* 多个独立的Servable
* 单个复合Servable

一个Servable也可以对应一个模型的一部分。
例如，一个大的lookup table可以在多个TensorFlow Serving实例共享。

**5. Loaders**

Loaders管理着Servable的生命周期。
Loader API使得公共结构独立于特定的机器学习算法，数据或者产品用例。
例如Loaders使用标准化API来加载或者卸载一个Servable。

**6. Sources**

Sources是用来查找和提供Servables的插件。每个Source提供零个或者多个
Servable Streams。对于每一个Servable Stream，
一个Source会为每个Version提供一个Loader实例，
这个Loader使得Servables Stream可以被加载。
(一个Source实际上会使用零或多个SourceAdapter链接到一起，
链上的最后一个item会触发(emits)该Loaders。)

Tensorflow Serving的Source接口能够从存储系统中查找并发现Servables。
TensorFlow Serving中包含了Source公共索引的实现。
例如，Source可以访问RPC等机制并且可以轮询文件系统。

Sources可以维持跨多个Servable和跨多个Version的状态。
这对于使用delta(diff)来在
versions间进行更新的Servables很有作用。
这对于不同版本之间使用增量的模型更新的Servables来说非常有用。


**7. Aspired Versions**

Aspired Versions表示需要被加载和准备的Servable Version集合。
Source会与该Servable Version集合进行通讯，一次一个Servable Stream。
当一个Source给出一个aspired versions的新列表到Manager时，
它会为该Servables stream取代之前的列表。该Manager会unload任何之前
已加载的versions，使它们不再出现在该列表中。

**8. Managers**

Managers处理了Servable的完整生命周期，包括:
* loadding Servable
* serving Servable
* unloading Servable
Managers会监听Source，并跟踪所有的versions。Mananger会尝试满足Source的请求，
但如果需要的resources不提供的话，也可以拒绝加载一个aspired version。
Managers也可以延期一个"unload"操作。
例如，一个Manager可以等待unload直到一个更新的version完成loading，
基于一个策略(Policy)来保证所有时间内至少一个version被加载。

TensorFlow Serving Manager提供一个单一的接口——GetServableHandle()——给client来访问已加载的Servable实现。

**9. Core**

TensorFlow Serving Core(通过标准TensorFlow Serving API)管理着一下的Servables：
* lifecycle
* metrics
TensorFlow Serving Core将Servables和loaders看成是透明的对象。

## 1.2 Servable的生命周期

<div align="center">
<image src="/images/tensorflow/tf_serving/tf_serving_1.svg"/>
</div>

<div align="center">
图1&nbsp;&nbsp;&nbsp;&nbsp;servable的生命周期
</div>

通俗地讲：

* source会为Servable Version创建一个Loaders。
* Loaders会作为Aspired Versions被发送给Manager，manager会加载和提供服务给客户端请求。

更详细的:
1. 一个Source插件会为一个特定的Version创建一个Loader。
该Loaders包含了一个加载Servable时所需的任何元数据。
2. Source会使用一个callback来通知该Aspired Version的Manager。
3. 该Manager应用该配置过的Version Policy来决定下一个要采用的动作，
它会被unload一个之前已经加载过的Version，或者加载一个新的Version。
4. 如果该Manager决定它是否安全，它会给Loader所需的资源，
并告诉该Loader来加载新的Version。
5. 一个Client会向最新Version的模型请求一个Handle，
接着Dynamic Manager返回一个handle给Servables的新version。

## 1.3 Extensibility

Tensorflow serving提供了一些扩展点，使你可以添加新功能。

**1. Version Policy**

Version Policy可以指定version序列，在单个servable stream中加载或卸载。

tensorflow serving包含了两个策略（policy）来适应大多数已知用例。
分别是：Availability Preserving Policy(避免没有version加载的情况；
在卸载一个老version前通常会加载一个新的version)， 
Resource Preserving Policy（避免两个version同时被加载，这需要两倍的资源；
会在加载一个新version前卸载老version）。对于tensorflow serving的简单用法，
一个模型的serving能力很重要，资源消耗要低，Availability Preserving 
Policy会确保在新version加载前卸载老version。对于TensorFlow Serving的复杂用法，
例如跨多个server实例管理version，Resource Preserving 
Policy需要最少的资源(对于加载新version无需额外buffer)

**2. Source**

新的资源(Source)可以支持新的文件系统、云供应商、算法后端。
TensorFlow Serving提供了一些公共构建块来很方便地创建新的资源。
例如，TensorFlow Serving包含了围绕一个简单资源polling行为的封装工具类。
对于指定的算法和数据宿主Servables，Source与Loaders更紧密相关。

**3. Loaders**

Loaders是用于添加算法和数据后端的扩展点。Tensorflow就是这样的一个算法后端。
例如，你实现了一个新的Loader来进行加载、访问、卸载一个新的servable类型的机器学习模型实例。
我们会为lookup tables和额外的算法创建Loaders。

**4. Batcher**
会将多个请求打包（Batching）成单个请求，可以极大减小执行inference的开销，
特别是像GPU这样的硬件加速存在的时候。Tensorflow Serving包含了一个请求打包组件
（request batching widget），使得客户端（clients）
可以很容易地将特定类型的inferences跨请求进行打包成一个batch，
使得算法系统可以更高效地处理。详见Batching Guide。


[[1] 使用 TensorFlow Serving 和 Flask 部署 Keras 模型](https://docle.github.io/2019/04/06/Deploying-keras-models-using-tensorflow-serving-and-flask/)

[[2] Google TFX 文档](https://www.tensorflow.org/tfx/guide/serving)