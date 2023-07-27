---
layout: post
title: EmbedX 上Graph Sage的迭代
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


# 需要准备的数据文件

1. 节点关系数据
2. 节点特征数据
3. 邻居节点特征数据
4. 随机游走之边数据

这4份数据里面, 数据1和数据2是原始数据, 数据3和数据4是派生数据。

## 1. 节点关系数据的生成

主要在于节点编码:
[ns_config]
node_name=["", "", "", ""]
ugc_node=["", "", "", ""]


[ns_code]


[【使用指南】Venus 平台 EmbedX 模型训练](https://iwiki.woa.com/pages/viewpage.action?pageId=1440219038)