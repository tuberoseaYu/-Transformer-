# 手撕 Transformer：从基本组件到完整架构

记录 基于PyTorch 实现 Transformer 的学习笔记。[`5_Transformer.ipynb`](./5_Transformer.ipynb) 从零实现了经典的 Encoder–Decoder Transformer，重点展示各模块的计算逻辑、张量形状以及三类注意力掩码的作用。

![经典 Transformer 架构](./images/transformer/architecture.png)

## Notebook 内容

整体实现按照“基本组件 → Encoder/Decoder Block → 完整 Transformer”的顺序展开：

1. **输入与位置编码**
   - 使用 `nn.Embedding` 将 token 映射为向量；
   - 实现正弦、余弦位置编码 `PositionalEncoding`；
   - 将 token embedding 与 positional encoding 相加。
2. **注意力机制**
   - 实现 Scaled Dot-Product Attention；
   - 实现 Multi-Head Attention；
   - 讲解 Query、Key、Value 的变换与多头拼接过程。
3. **核心网络组件**
   - Feed Forward Network；
   - 残差连接；
   - Layer Normalization；
   - Dropout。
4. **Encoder 与 Decoder**
   - 组装 `EncoderLayer` 和 `Encoder`；
   - 组装带掩码自注意力、交叉注意力的 `DecoderLayer` 和 `Decoder`。
5. **完整 Transformer**
   - 组合 Encoder、Decoder 与输出线性层；
   - 统一生成并应用三类 attention mask。

## 核心公式

### 位置编码

正弦和余弦函数为序列中的每个 token 提供位置信息：

![位置编码公式](./images/transformer/positional-encoding-formula.png)

低维方向变化较快，高维方向变化较慢，使模型能够感知绝对位置和相对距离。输入核心网络前，会将位置编码与词向量相加：

```text
input = token_embedding + positional_encoding
shape = [batch, seq_len, d_model]
```

### Scaled Dot-Product Attention

![缩放点积注意力公式](./images/transformer/scaled-dot-product-attention.png)

Notebook 中对应的主要计算过程为：

```python
scores = query @ key.transpose(-2, -1) / sqrt(d_k)
scores = scores.masked_fill(mask, -1e9)
weights = softmax(scores)
output = weights @ value
```

被 mask 的位置会在 Softmax 前替换为极小值，因此其注意力权重接近 0。

## 三类 Attention Mask

| Mask | 使用位置 | 遮挡内容 | 作用 |
| --- | --- | --- | --- |
| `src_mask` | Encoder Self-Attention | 源序列中的 `<pad>` | 避免 Encoder 关注补齐位置 |
| `dst_mask` | Decoder Masked Self-Attention | 目标序列中的 `<pad>` 和未来 token | 避免关注补齐位置并防止“偷看未来” |
| `src_dst_mask` | Decoder Cross-Attention | 源序列中的 `<pad>` | Decoder 读取 Encoder 输出时跳过补齐位置 |

![三类 Mask 在 Transformer 中的位置](./images/transformer/attention-masks.png)

Mask 的形状为 `[batch, 1, seq_q, seq_k]`，其中第二维可以广播到所有注意力头。

## 模型结构

```text
Transformer
├── Encoder
│   ├── Token Embedding + Positional Encoding
│   └── N × EncoderLayer
│       ├── Multi-Head Self-Attention
│       ├── Add & LayerNorm
│       ├── Feed Forward Network
│       └── Add & LayerNorm
├── Decoder
│   ├── Token Embedding + Positional Encoding
│   └── N × DecoderLayer
│       ├── Masked Multi-Head Self-Attention
│       ├── Cross-Attention
│       ├── Feed Forward Network
│       └── Residual Connection & LayerNorm
└── Linear Output Layer
```

### Feed Forward Network

FFN 对每个位置独立进行两次线性映射，中间使用 ReLU 激活，并将维度从 `d_model` 扩展到 `d_ff` 后再映射回来。

![前馈神经网络](./images/transformer/feed-forward-network.png)

### Decoder

与 Encoder 相比，Decoder 额外包含：

- 防止看到未来 token 的 Masked Self-Attention；
- 读取 Encoder 输出的 Cross-Attention。

![Decoder 结构](./images/transformer/decoder-architecture.png)

## 如何运行

建议使用 Python 3.10 或更高版本，并安装 Jupyter 与 PyTorch：

```bash
pip install torch jupyter
```

在项目根目录启动：

```bash
jupyter notebook notes/5_Transformer.ipynb
```

也可以直接使用 VS Code 的 Jupyter 扩展打开并逐单元格运行。

## 文件说明

```text
notes/
├── 5_Transformer.ipynb       # Transformer 原理讲解与 PyTorch 实现
├── README.md                 # 本说明文档
└── images/transformer/       # README 使用的结构图和公式图片
```

## 学习建议

阅读时可以重点跟踪以下张量形状的变化：

```text
[batch, seq_len, d_model]
→ [batch, heads, seq_len, d_k]
→ attention
→ [batch, seq_len, d_model]
```

理解这些维度变化，以及 `src_mask`、`dst_mask`、`src_dst_mask` 分别在何处生效，是掌握这份实现的关键。

## 参考资料

- Vaswani et al., *Attention Is All You Need* (2017)
- PyTorch 官方文档：`nn.Embedding`、`nn.Linear`、`nn.LayerNorm`
