---
layout: post
title: 常用评估指标的原理和计算方式
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


记录一次数据下跌的排查

# ROC曲线

ROC曲线的全称是(receiver operating characteristic curve), 反应敏感性和特异性连续变量的的综合指标。

那么什么是敏感度(sensitivity)? 什么是特异度(specificity)?
对负样本的分类准确率, 对正样本的分类准确率。

[特异度（specificity）与灵敏度（sensitivity](https://www.jianshu.com/p/7919ef304b19)
[糖尿病人的例子](https://www.zhihu.com/question/30750849)

specificity: 是指对负样本识别的特异度, 负样本识别的比率高, 特异度越高。
TN / (TN + FP)
sensitivity: 表示对正样本识别的敏感度, 正样本识别的比率高,  敏感度越高。

ROC曲线的横轴是:
1 - specificity
ROC曲线的纵轴是:
sensitivity

TP / (TP + FN)

ROC曲线是怎么绘制的

选择不同的分类阈值, 然后根据这些阈值分别计算sensitivity和 specificity

我们假设两种极端情况:
1. 一个特别糟糕的模型, sensitivity的升高伴随着specificity的下跌幅度是1:1 这个时候, 模型完全随机。
2. 一个特别好的模型, sensitivity的升高于specificity无关。
3. 一个真实的优秀模型, sensitivity的升高, specificty降低，但是斜率逐渐减少。

那么, 我们可以得到ROC曲线下的面积其实可以表示模型分类的效果。

# AUC的原理

AUC的全称(Area under curve)

# AUC的计算方式

1. 统计ROC曲线下的面积
2. 在所有的正负样本对中, 正样本的概率大于负样本对的概率。
3. 根据排序计算

$$AUC=\frac{\sum_{ins_{i}\in pos} rank_{ins_i} - \frac{M \times (M+1)}{2}}{M \times N}$$

其中$rank_{ins_i}$表示正样本在所有样本中的排序值, 
$M$ 表示正样本个数, $N$ 表示负样本个数。

```C
struct RECORD {
    float label;
    float predict;
    float weight;

    RECORD(float label_y, float predict_y, float sample_weight = 1.0):
        label(label_y), predict(predict_y), weight(sample_weight) {}

    bool operator<(RECORD &i) { return predict < i.predict; }
  };
```

首先定义了一个`RECORD`的结构体, 这个结构体包含三个部分:
* label: 表示positive和negative样本
* predict: 表示预测的得分
* weight: 表示每个样本的权重, weight的默认值是1.0


```C
for (size_t i = 0; i < predicts.size(); ++i) {
    if (weights.empty()) {
      id_list_result[group_ids[i]].push_back(
          AucCalc::RECORD(labels[i], predicts[i]));
    } else {
      id_list_result[group_ids[i]].push_back(
          AucCalc::RECORD(labels[i], predicts[i], weights[i]));
    }
  }

```

```C
struct RetAucInfo {
  RetAucInfo() : area(0.0), auc(0.0), weighted_postive_num(0.0),
      weighted_negative_num(0.0) {}
  // area is the intermediate value for distributed auc
  double area;
  double auc;
  double weighted_postive_num;
  double weighted_negative_num;
  SampleStatisticInfo sample_info;
};

struct SampleStatisticInfo {
    SampleStatisticInfo() : predict_sum(0.0), postive_num(0), negative_num(0) {}
    void Reset() {
      predict_sum = 0.0;
      postive_num = 0;
      negative_num = 0;
    }
    double predict_sum;
    uint64_t postive_num;
    uint64_t negative_num;
  };
```
在`RetAucInfo`这个结构体里面有三个字段, 在不考虑权重的情况下:
* area: 表示负样本的索引之和。
* auc: 
* weighted_postive_num: 表示正样本的个数
* weighted_negative_num: 表示负样本的个数
* sample_info.postive_num: 表示正样本个数
* sample_info.negative_num: 表示负样本个数
* sample_info.predict_sum: 表示整体预测的数目



整体AUC的计算在以下代码中完成。

```C
bool AucCalc::CalcLocalAuc(
    const std::list<RECORD> &list_result, RetAucInfo& out) {
  double weighted_postive_num = 0;
  double weighted_negative_num = 0;
  double predict_sum = 0.0;
  uint64_t postive_num = 0;
  uint64_t negative_num = 0;

  for (auto next = list_result.begin(); next != list_result.end(); next++) {
    if (next->label == 1) {
      weighted_postive_num += next->weight;
      postive_num++;
    } else {
      weighted_negative_num += next->weight;
      negative_num++;
    }
    predict_sum += next->predict;
  }  // for next

  double total_area = weighted_postive_num * weighted_negative_num;
  out.weighted_postive_num = weighted_postive_num;
  out.sample_info.predict_sum = predict_sum;
  out.sample_info.postive_num = postive_num;
  out.sample_info.negative_num = negative_num;
  if (total_area == 0) {
    NUMEROUS_ERROR << "fail to calculate auc, weighted_postive_num: "
                   << weighted_postive_num << ", weighted_negative_num: "
                   << weighted_negative_num;
    return false;
  }

  double h = 0;
  double area = 0;
  std::string label_before = "";
  std::string next_weight = "";
  for (auto next = list_result.rbegin(); next != list_result.rend(); next++) {
    label_before += std::to_string(next->label) + " ";
    next_weight += std::to_string(next->weight) + " ";
    if (next->label == 1) {
      h += next->weight;
    } else {
      area += (h * next->weight);
    }
  }

```


# AUC为什么可以这么计算

[AUC的计算方法](https://blog.csdn.net/qq_22238533/article/details/78666436)


# COPC的计算方式

[AUC和COPC](https://www.cnblogs.com/Lee-yl/p/15061680.html)

什么是ECPM

[ctr预估中的评估指标及校准](https://blog.csdn.net/u013019431/article/details/102473137)