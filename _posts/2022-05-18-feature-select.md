---
layout: post
title: 怎么样做有效的特征选择
description: "以House Price为例快速构建一个Base Line模型"
modified: 2020-10-01
tags: [Contrastive Learning]
categories: [算法]
image:
  path: /images/algorthim/info_nce/1.jpeg
  feature: /algorthim/info_nce/1.jpeg
  credit: x1
  creditlink: https://cxl.com/blog/bandit-tests/
---


* 什么是特征工程当中的无量纲化？ 为什么要做无量纲化？

从原始数据中提取特征以供算法和模型使用。通过总结和归纳，人们认为特征工程包括以下几个方面:


无量纲化使不同规格的数据转化到同一规格, 常用方法:
* 标准化和区间缩放法
* 标准化的前提是特征服从正态分布。

为什么标准化要求区间符合符合正态分布？


[[1]机器学习中，有哪些特征选择的工程方法？](https://www.zhihu.com/question/28641663)