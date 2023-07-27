---
layout: post
title: DIN Attention的代码解析
description: "以House Price为例快速构建一个Base Line模型"
modified: 2020-10-01
tags: [Contrastive Learning]
categories: [心理学]
image:
  path: /images/algorthim/info_nce/1.jpeg
  feature: /algorthim/info_nce/1.jpeg
  credit: x1
  creditlink: https://cxl.com/blog/bandit-tests/
---



# 代码

```python
print("[INFO] build mlp module")
#queries [B, D]
queries = input_layer[0]
#keys/facts shape=[B, sequence_length, embedding_size]
keys = input_layer[1]



# get dimension of query
queries_hidden_units = queries.get_shape().as_list()[-1]
# tile queries for sequence length， [B, queries_D] -> [B, queries_D * sequence_length]

print("tf.shape(keys):", keys.get_shape())
print("tf.shape(queries):", queries.get_shape())
queries = tf.tile(queries, [1, tf.shape(keys)[1]])
# [B, sequence_length, embedding_size]

# query and fact are of the same dimension
queries = tf.reshape(queries,
                     [-1, tf.shape(keys)[1], queries_hidden_units])


if len(input_layer) == 3:
  timegap = input_layer[2]  # [B, sequence_length, d]
  din_all = tf.concat([queries, keys, queries - keys, queries * keys, timegap],
                      axis=-1)
else:
  din_all = tf.concat([queries, keys, queries - keys, queries * keys],
                      axis=-1)
  print("din attention mlp input shape: ", din_all.get_shape())
  # units = 80,  输出的维度大小，改变inputs的最后一维
  # 即在 [B, sequence_length, concat_embedding] 的最后一维度 变成 80
  # mlp:[80, 40, 1]
  d_layer_1_all = tf.layers.dense(din_all,
                                  80,
                                  activation=tf.nn.sigmoid,
                                  name="{0}_f1_att".format(self.name),
                                  reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(
    d_layer_1_all,
    40,
    activation=tf.nn.sigmoid,
    name="{0}_f2_att".format(self.name),
    reuse=tf.AUTO_REUSE,
  )
  d_layer_3_all = tf.layers.dense(d_layer_2_all,
                                  1,
                                  activation=None,
                                  name="{0}_f3_att".format(self.name),
                                  reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all
  # Mask

  key_masks = tf.sequence_mask(
    #lenghths = [B,B,B,B,B,B,B...B]
    # maxlen = T, 控制最大长度
    tf.ones(
      shape=tf.shape(keys)[0],
      dtype=tf.int32) * tf.shape(keys)[1], tf.shape(keys)[1])  # [B, T]
  # [B, 1, T] all true
  key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-(2**32) + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1]**0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  # [B, 1, T] * [B, T, D]
  output = tf.matmul(outputs, keys)  # [B, 1, D]
  # tf.squeeze 与 tf.expand_dims互为逆操作
  output = tf.squeeze(output, [1])

  return output
```

# 输入部分

* Keys: 表示待查询的key，其结构为[B, Sequence_length, Embedding_size]
  * B: 表示Batch Size的大小
  * Sequence_length: 表示序列的长度
  * Embedding_size:表示序列嵌入的大小
* Query: 表示查询的Qurey，在DIN中表示Target Item的Embedding。其结构为[B, Sequence_length, D]
  * B: 表示Batch Size的大小。
  * Sequence_length:表示序列的长度。
  * D: 表示Query序列的Embedding长度。

这里先不考虑Time Gap的操作。



# 操作部分

* tf.tile(queries, [1, tf.shape(keys)[1]):

> 这里执行了一个tile操作, 我们来看一下tile操作的功能是什么。
>
> 这里假设Batch Size = 3 sequence_length = 5 embedding_size = 4 Queries_D = 6

```python
import tensorflow as tf
a = [[[1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0],
     [1.0, 2.0, 3.0, 4.0]]]

b = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
keys = tf.convert_to_tensor(a)
queries = tf.convert_to_tensor(b)

queries_hidden_units = queries.get_shape().as_list()[-1] #获取embedding size
print(tf.shape(keys)[1])
print(queries)
queries = tf.tile(queries, [1,tf.shape(keys)[1]]) # 在第一维上不扩增，在第二维度上扩增embedding size的大小
```

结果输出如下:

```
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(
[[1. 2. 3. 4. 5. 6.]
 [1. 2. 3. 4. 5. 6.]
 [1. 2. 3. 4. 5. 6.]], shape=(3, 6), dtype=float32)
tf.Tensor(
[[1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6.
  1. 2. 3. 4. 5. 6.]
 [1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6.
  1. 2. 3. 4. 5. 6.]
 [1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6. 1. 2. 3. 4. 5. 6.
  1. 2. 3. 4. 5. 6.]], shape=(3, 30), dtype=float32)
```

* 将Queries 转换成[B, Sequence_length, queries_D]的大小

```python
queries = tf.reshape(queries,
                             [-1, tf.shape(keys)[1], queries_hidden_units])
print(queries)
```

```
tf.Tensor(
[[[1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]]

 [[1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]]

 [[1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]
  [1. 2. 3. 4. 5. 6.]]], shape=(3, 5, 6), dtype=float32)
```

* 
