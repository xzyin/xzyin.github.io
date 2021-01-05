---
layout: post
title: YouTube DNN实践过程的一些细节
description: "浅显看YouTube DNN实现细节"
modified: 2020-10-01
tags: [Deep Learning, Seq2Seq]
categories: [算法]
image:
  path: /images/algorthim/youtube_dnn/youtube-1.png
  feature: /algorthim/youtube_dnn/youtube-1.png
  credit: x1
  creditlink: https://deeplearn.org/arxiv/69226/a-comparative-study-on-hierarchical-navigable-small-world-graphs
---
目录

* TOC 
{:toc}


这篇文章中主要记录一下YouTube DNN实现的具体细节和迭代步骤。
在整个YouTube DNN的迭代过程中我们主要分为以下几个过程:

* BaseLine模型(history 版本)

* Feature版本

* Share Weight 版本

* Online Serving版本

* Filter版本

* 

评估指标的修昔底德陷阱？

推荐系统到底是什么？

Long Tail和Hot Top之间的取舍？

群众的力量？



# 1. 
<div align="center">
<image src="/images/algorthim/l1_l2/regular-2.png"/>
</div>

<div align="center">
图1&nbsp;&nbsp;&nbsp;&nbsp;模型训练中的不同状态
</div>