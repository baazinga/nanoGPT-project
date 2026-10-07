# 模型参数详解

## 一、配置参数 (Config)

### 数据相关
- **`block_size = 2048`**
  - 最大上下文长度（最大序列长度）
  - 位置编码表的大小上限
  - KV cache 的最大缓存长度
  - 超过此长度会报错或截断

- **`train_split = 0.9`**
  - 训练集占总数据的比例（90%）
  - 剩余 10% 用于验证集

- **`vocab_size = None`**
  - 词汇表大小（token 数量）
  - 训练时从数据中自动确定
  - 推理时需要手动指定

### 训练相关
- **`batch_size = 12`**
  - 每个 batch 并行处理的序列数量
  - 影响显存占用和训练速度

- **`max_iters = 5000`**
  - 训练循环的最大迭代次数
  - 每个 iteration 处理一个 batch

- **`eval_interval = 500`**
  - 每训练多少步进行一次验证
  - 用于监控模型性能

- **`learning_rate = 1e-3`**
  - 学习率（0.001）
  - 控制参数更新的步长

- **`eval_iters = 200`**
  - 验证时评估多少个 batch
  - 用于计算平均验证损失

### 模型架构参数
- **`n_embd = 256`**
  - 嵌入维度（Embedding Dimension）
  - 每个 token 被编码为 256 维向量
  - 也是模型的主隐藏层维度
  - 影响模型容量和参数量

- **`n_head = 4`**
  - 注意力头数量（Multi-Head Attention）
  - 每个头独立学习不同的注意力模式
  - 总参数量与头数成正比

- **`n_layer = 4`**
  - Transformer 层数（Block 数量）
  - 模型深度，影响表达能力
  - 每层包含：Self-Attention + FeedForward

- **`dropout = 0.2`**
  - Dropout 比率（20%）
  - 训练时随机丢弃 20% 的神经元
  - 防止过拟合
  - 推理时自动关闭（eval 模式）

- **`max_new_tokens = 500`**
  - 生成时的最大新 token 数量
  - 控制生成长度上限

### 设备相关
- **`device`**
  - 自动选择：CUDA > MPS > CPU
  - 优先使用 GPU 加速

### KV Cache 相关
- **`use_kv_cache = True/False`**
  - 是否启用 KV cache
  - True: 推理时缓存 K/V，加速生成
  - False: 每次重算，速度慢但内存占用少

---

## 二、模型架构参数

### 1. Head (单注意力头)

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
```

**参数：**
- **`head_size`**: 每个头的维度 = `n_embd // n_head = 256 // 4 = 64`
  - 每个头将 256 维输入映射到 64 维
  - 4 个头拼接后恢复 256 维

- **`self.key`**: Key 投影层
  - 输入: (B, T, 256)
  - 输出: (B, T, 64)
  - 参数量: 256 × 64 = 16,384

- **`self.query`**: Query 投影层
  - 输入: (B, T, 256)
  - 输出: (B, T, 64)
  - 参数量: 256 × 64 = 16,384

- **`self.value`**: Value 投影层
  - 输入: (B, T, 256)
  - 输出: (B, T, 64)
  - 参数量: 256 × 64 = 16,384

- **`self.tril`**: 因果掩码（下三角矩阵）
  - 形状: (block_size, block_size) = (2048, 2048)
  - 确保 token 只能看到之前的 token（因果性）

- **`self.dropout`**: Dropout 层
  - 比率: 0.2

- **`self.cache_k`, `self.cache_v`**: KV cache 缓冲区
  - 形状: (B, past_len, head_size)
  - 存储历史 K/V 值，避免重复计算

### 2. MultiHeadAttention (多头注意力)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
```

**参数：**
- **`num_heads = 4`**: 注意力头数量
- **`head_size = 64`**: 每个头的维度
- **`self.proj`**: 输出投影层
  - 输入: (B, T, 256) [4 个头拼接]
  - 输出: (B, T, 256)
  - 参数量: 256 × 256 = 65,536

**总参数量（一个 MultiHeadAttention）：**
- 4 个 Head: 4 × (16,384 × 3) = 196,608
- 输出投影: 65,536
- **总计: 262,144 参数**

### 3. FeedForward (前馈网络)

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      # 扩展层
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),     # 压缩层
            nn.Dropout(Config.dropout),
        )
```

**参数：**
- **`4 * n_embd = 1024`**: 中间层维度（扩展 4 倍）
  - 先扩展到 1024 维，再压缩回 256 维
  - 增加模型非线性表达能力

**参数量：**
- 扩展层: 256 × 1024 = 262,144
- 压缩层: 1024 × 256 = 262,144
- **总计: 524,288 参数**

### 4. Block (Transformer 层)

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.n1 = nn.LayerNorm(n_embd)
        self.n2 = nn.LayerNorm(n_embd)
```

**参数：**
- **`self.sa`**: 自注意力层（262,144 参数）
- **`self.ffwd`**: 前馈网络（524,288 参数）
- **`self.n1`, `self.n2`**: LayerNorm 层
  - 参数量: 2 × 256 × 2 = 1,024（每个 LayerNorm 有 scale 和 bias）

**每层总参数量: 262,144 + 524,288 + 1,024 = 787,456**

### 5. BigramLanguageModel (完整模型)

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(...) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

**参数：**

1. **`self.token_embedding_table`**
   - Token 嵌入表
   - 形状: (vocab_size, 256)
   - 参数量: vocab_size × 256
   - 例如 vocab_size=5000: 5000 × 256 = 1,280,000

2. **`self.position_embedding_table`**
   - 位置嵌入表
   - 形状: (2048, 256)
   - 参数量: 2048 × 256 = 524,288

3. **`self.blocks`** (4 层)
   - 每层: 787,456 参数
   - 总计: 4 × 787,456 = 3,149,824

4. **`self.ln_f`**
   - 最终 LayerNorm
   - 参数量: 512

5. **`self.lm_head`**
   - 语言模型头（输出层）
   - 输入: (B, T, 256)
   - 输出: (B, T, vocab_size)
   - 参数量: 256 × vocab_size
   - 例如 vocab_size=5000: 256 × 5000 = 1,280,000

**总参数量（以 vocab_size=5000 为例）：**
- Token Embedding: 1,280,000
- Position Embedding: 524,288
- 4 个 Block: 3,149,824
- Final LayerNorm: 512
- LM Head: 1,280,000
- **总计: 6,234,624 参数（约 6.2M）**

---

## 三、形状变化流程

### 输入到输出的维度变化：

```
输入: idx (B, T)  # B=batch_size, T=sequence_length
  ↓
Token Embedding: (B, T, 256)
  ↓
Position Embedding: (B, T, 256)
  ↓
相加: (B, T, 256)
  ↓
Block 1:
  - Self-Attention: (B, T, 256) → (B, T, 256)
  - FeedForward: (B, T, 256) → (B, T, 256)
  ↓
Block 2-4: 同样处理
  ↓
Final LayerNorm: (B, T, 256)
  ↓
LM Head: (B, T, 256) → (B, T, vocab_size)
  ↓
输出: logits (B, T, vocab_size)
```

---

## 四、关键参数对模型的影响

| 参数 | 增加的影响 | 减少的影响 |
|------|-----------|-----------|
| **n_embd** | 模型容量↑, 参数量↑↑, 显存↑↑ | 训练速度↓, 过拟合风险↑ |
| **n_layer** | 表达能力↑, 参数量↑, 训练时间↑ | 梯度消失风险↑, 显存↑ |
| **n_head** | 注意力模式多样性↑, 参数量↑ | 计算量↑, 显存↑ |
| **block_size** | 上下文长度↑, 位置编码表↑ | 显存↑↑, 训练速度↓ |
| **dropout** | 正则化↑, 过拟合风险↓ | 模型容量↓, 训练损失↑ |
| **batch_size** | 训练稳定性↑, 显存↑↑ | 梯度估计更准确 |

---

## 五、你的模型配置总结

- **模型大小**: 约 6.2M 参数（vocab_size=5000 时）
- **架构**: 4 层 Transformer，256 维，4 头
- **上下文长度**: 最大 2048 tokens
- **适用场景**: 小型语言模型，适合学习和实验
