# 学习大模型和 Transformer

——依靠 [Transformers快速入门](https://transformers.run/) 这篇教程，配合 [huggingface/transformers](https://github.com/huggingface/transformers) 这个库和 NLP 场景

## 代码

环境

```bash
uv sync
uv pip install -e .
```

- src/llm_transformer_nlp/
    - config.py 获取 Bert 配置和分词器
    - sample_token_embedding.py 词嵌入，供没有自己实现嵌入层时使用
    - attention/ 注意力
        - scaled_dot_product_attention.py 缩放点积注意力
        - multi_head_attention.py 多头自注意力
    - encoder/ 编码器
        - encoder.py 完整的编码器实现
        - feed_forward.py 前馈神经网络
        - layer_normalization.py 层归一化
        - positional_embedding.py 位置嵌入

transformers 库：[huggingface/transformers](https://github.com/huggingface/transformers)，文档：[Transformers](https://huggingface.co/docs/transformers/index)

## Transformer

Transformer 模型本质上都是**预训练**语言模型，大都采用**自监督学习** (Self-supervised learning) 的方式在大量生语料上进行训练，也就是说，训练这些 Transformer 模型完全不需要人工标注数据。下面两个常用的预训练任务：
- **因果语言建模** (causal language modeling)：因果语言建模 (causal language modeling)
- **遮盖语言建模** (masked language modeling)：基于上下文（周围的词语）来预测句子中被遮盖掉的词语 (masked word)

虽然可以对训练过的语言产生统计意义上的理解，但是如果直接拿来完成特定任务，效果往往并不好。因此，还会采用**迁移学习** (transfer learning) 方法，使用特定任务的标注语料，**以有监督学习的方式对预训练模型参数进行微调** (fine-tune)，以取得更好的性能

**迁移学习**：在绝大部分情况下，我们都应该尝试找到一个尽可能接近我们任务的预训练模型，然后微调它

结构：
- Encoder：负责理解输入文本，为每个输入构造对应的语义表示（语义特征）
- Decoder：负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列

按模型结构对 Transformer 模型分类
- 纯 Encoder 模型（例如 BERT），又称**自编码 (auto-encoding)** Transformer 模型
    - 适用于只需要理解输入语义的任务，例如句子分类、命名实体识别
- 纯 Decoder 模型（例如 GPT），又称**自回归 (auto-regressive)** Transformer 模型
    - 适用于生成式任务，例如文本生成
- Encoder-Decoder 模型（例如 BART、T5），又称 **Seq2Seq (sequence-to-sequence)** Transformer 模型
    - 适用于需要基于输入的生成式任务，例如翻译、摘要

mask
- decoder 的训练过程
- encoder、decoder 的填充字符处理


纯 Encoder 模型通常通过破坏给定的句子（例如随机遮盖其中的词语），然后让模型进行重构来进行预训练，最适合处理那些需要理解整个句子语义的任务，例如句子分类、命名实体识别（词语分类）、抽取式问答

### 注意力

对输入文本进行编码，转换为一个由词语向量组成的矩阵 $\boldsymbol{X} = (\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n)$，其中 $\boldsymbol{x}_i$ 就表示第 $i$ 个词语的词向量，维度为 $d$，故 $\boldsymbol{X} \in \mathbb{R}^{n \times d}$。
1. 首先对句子进行分词
2. 然后将每个词语 (token) 都转化为对应的词向量 (token embeddings)

对 token 序列编码方式（2 老 1 新）
- RNN（例如 LSTM） $\boldsymbol{y}_t = f(\boldsymbol{y}_{t-1}, \boldsymbol{x}_t)$
    - 递归的结构导致其无法并行计算，因此速度较慢
    - 本质是一个马尔科夫决策过程，难以学习到全局的结构信息
- CNN： $\boldsymbol{y}_t = f(\boldsymbol{x}_{t-1}, \boldsymbol{x}_t, \boldsymbol{x}_{t+1})$ （核尺寸为 3）
    - 能够并行地计算，因此速度很快
    - 由于是通过窗口来进行编码，所以更侧重于捕获局部信息，难以建模长距离的语义依赖
- Attention（Google《Attention is All You Need》） $\boldsymbol{y}_t = f(\boldsymbol{x}_t, \boldsymbol{A}, \boldsymbol{B})$
    - $\boldsymbol{A},\boldsymbol{B}$ 是另外的词语序列（矩阵），如果取 $\boldsymbol{A} = \boldsymbol{B} = \boldsymbol{X}$ 就称为 Self-Attention

#### Scaled Dot-product Attention

$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left( \frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}} \right) \boldsymbol{V}$$

1. 计算注意力权重
2. 更新 token embeddings

当 Q、K 相同时，相同单词会分配较大分数

#### Multi-head Attention

$$
head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \dots, head_h)
$$

每个注意力头负责关注某一方面的语义相似性，多个头就可以让模型同时关注多个方面。因此与简单的 Scaled Dot-product Attention 相比，Multi-head Attention 可以捕获到更加复杂的特征信息

### Transformer Encoder

#### feed-forward

Transformer Encoder/Decoder 中的前馈子层实际上就是**两层全连接神经网络**，它**单独地处理序列中的每一个词向量**，也被称为 **position-wise** feed-forward layer

#### Layer Normalization

- **Post layer normalization**：Transformer 论文中使用的方式，将 Layer normalization 放在 Skip Connections 之间。 但是因为梯度可能会发散，这种做法很难训练，还需要结合学习率预热 (learning rate warm-up) 等技巧
- **Pre layer normalization**：目前主流的做法，将 Layer Normalization 放置于 Skip Connections 的范围内。这种做法通常训练过程会更加稳定，并且不需要任何学习率预热

#### Positional Embeddings

Positional Embeddings 用于添加词语的位置，因为注意力机制无法捕获词语之间的位置信息

思路是**使用与位置相关的值模式来增强词向量**

- **模型自动学习位置嵌入**：预训练数据集足够大时，可以让模型自己学习位置嵌入，当前项目代码中实现的就是这个
- **绝对位置表示**：使用由调制的正弦和余弦信号组成的静态模式来编码位置。没有大量训练数据可用时，这种方法尤其有效
- **相对位置表示**：需要在模型层面对注意力机制进行修改，而不是通过引入嵌入层来完成

### Transformer Decoder

与 Encoder 最大的不同在于 Decoder 有两个注意力子层

1. Masked multi-head self-attention layer：确保在每个时间步生成的词语仅基于过去的输出和当前预测的词，否则 Decoder 相当于作弊了
2. Encoder-decoder attention layer：以解码器的中间表示作为 queries，对 encoder stack 的输出 key 和 value 向量执行 Multi-head Attention

## 名词

- **NLP**（Natural Language Processing，自然语言处理）
- **LSTM**（Long Short Term Memory，长短时记忆网络）
- **CNN**（Convolutional Neural Networks，卷积神经网络）
- **乔姆斯基形式语言**（Chomsky Formal languages）：基于语法与规则的 NLP 方向
- **统计语言模型**：基于统计方法为自然语言建立数学模型
    - **马尔可夫（Markov）假设**：文字序列是否合理，就是句子 S 出现的概率 
        - $P(S) = P(w_1, w_2, \dots, w_n) = P(w_1) \prod_{i=2}^{n} P(w_i | w_{i - N + 1}, w_{i - N + 2}, \dots, w_{i - 1})$
        - 假设每个词语仅与它前面的 $N-1$ 个词语有关
        - 对应的语言模型称为 N 元模型（N-gram）
            - $N=2$ 时称为二元模型（Bigram）
            - $N=1$ 时就是上下文无关模型
- **NNLM**（Neural Network Language Model，神经网络语言模型)，2003 年本吉奥（Bengio）提出
    - 思路与统计语言模型保持一致，它通过输入词语前面的 $N-1$ 个词语来预测当前词
    - 不仅能够能够根据上文预测当前词语，同时还能够给出所有词语的词向量（Word Embedding）
- **Word2Vec** 模型：2013 年 Google 提出
    - Word2Vec 模型提供的词向量在很长一段时间里都是自然语言处理方法的标配，即使是后来出现的 Glove 模型也难掩它的光芒
    - Word2Vec 分为 CBOW (Continuous Bag-of-Words) 和 Skip-gram 两种
        - CBOW 使用周围的词语来预测当前词 
        - Skip-gram 则正好相反，它使用当前词来预测它的周围词语
- **互信息**（Mutual Information）：20 世纪 90 年代初，雅让斯基（Yarowsky）
    - 对于多义词，可以使用文本中与其同时出现的互信息最大的词语集合来表示不同的语义
        - 例如对于“苹果”，当表示水果时，周围出现的一般就是“超市”、“香蕉”等词语；而表示“苹果公司”时，周围出现的一般就是“手机”、“平板”等词语
        - 互信息就来自于信息论
            - 1948 年，香农（Claude Elwood Shannon）在他著名的论文《通信的数学原理》中提出了“信息熵”（Information Entropy）的概念，解决了信息的度量问题，并且量化出信息的作用
- **ELMo** 模型（Embeddings from Language Models）：2018
    - ELMo 模型首先对语言模型进行预训练，使得模型掌握编码文本的能力
    - 然后在实际使用时，对于输入文本中的每一个词语，都提取模型各层中对应的词向量拼接起来作为新的词向量
    - ELMo 模型采用双层双向 LSTM 作为编码器
- **BERT** 模型（Bidirectional Encoder Representations from Transformers）：2018
- **涌现**（Emergent Abilities）能力：在保持模型结构以及预训练任务基本不变的情况下，仅仅通过扩大模型规模就可以显著增强模型能力，尤其当规模达到一定程度时，模型甚至能够解决未见过的复杂问题
    - 规模扩展定律（**Scaling Laws**）
- **ChatGPT** 模型（Chat Generative Pre-trained Transformer）：2022 年 11 月 30 日 OpenAI 
- **微调**（Instruction Tuning）
- **思维链**提示（Chain-of-Thought Prompting）

## `pipelines`

[Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

`pipelines` 封装了预训练模型和对应的前处理和后处理环节。我们只需输入文本，就能得到预期的答案

pipeline 模型会自动完成以下三个步骤：
1. 预处理（preprocessing）将文本预处理为模型可以理解的格式
2. 将预处理好的文本送入模型
3. 后处理（postprocessing）对模型的预测值进行后处理，输出人类可以理解的格式

示例
- 分类
    - 情感分析 [sentiment_analysis.py](src/pipelines/sentiment_analysis.py)
    - 零训练样本分类 [zero_shot_classification.py](src/pipelines/zero_shot_classification.py)
    - 命名实体识别（NER） [ner.py](src/pipelines/ner.py)
- 文本生成
    - 前缀模板（Preﬁx Prompt） [text_generation.py](src/pipelines/text_generation.py)
    - 填充模板（Cloze Prompt） [fill_mask.py](src/pipelines/fill_mask.py)
- 问答，根据上下文回答问题 [question_answering.py](src/pipelines/question_answering.py)
    - 抽取式 QA（extractive QA） 假设答案就包含在文档中，因此直接从文档中抽取答案
    - 多选 QA（multiple-choice QA） 从多个给定的选项中选择答案，相当于做阅读理解题
    - 无约束 QA（free-form QA） 直接生成答案文本，并且对答案文本格式没有任何限制


使用 `transformers.AutoModel.from_pretrained` 可以加载模型。预训练模型的本体只包含基础的 Transformer 模块，对于给定的输入，它会输出一些神经元的值，称为 hidden states 或者特征 (features)。对于 NLP 模型来说，可以理解为是文本的高维语义表示。这些 hidden states 通常会被输入到其他的模型部分（称为 head），以完成特定的任务，例如送入到分类头中完成文本分类任务

Transformers 库封装了很多不同的结构，常见的有：

- `*Model` （返回 hidden states）
- `*ForCausalLM` （用于条件语言模型）
- `*ForMaskedLM` （用于遮盖语言模型）
- `*ForMultipleChoice` （用于多选任务）
- `*ForQuestionAnswering` （用于自动问答任务）
- `*ForSequenceClassification` （用于文本分类任务）
- `*ForTokenClassification` （用于 token 分类任务，例如 NER）

Transformer 模块的输出是一个维度为 (Batch size, Sequence length, Hidden size) 的三维张量，其中 Batch size 表示每次输入的样本（文本序列）数量，即每次输入多少个句子；Sequence length 表示文本序列的长度，即每个句子被分为多少个 token；Hidden size 表示每一个 token 经过模型编码后的输出向量（语义表示）的维度

对于情感分析任务，很明显我们最后需要使用的是一个文本分类 head。因此，实际上我们不会使用 AutoModel 类，而是使用 `AutoModelForSequenceClassification`，其对于 batch 中的每一个样本，模型都会输出一个两维的向量（每一维对应一个标签，positive 或 negative）。要将他们转换为概率值，还需要让它们经过一个 SoftMax 层。最后，为了得到对应的标签，可以读取模型 config 中提供的 id2label 属性

## 模型与分词器

### 模型

`具体Model.from_pretrained` 也可以加载模型，一般是通过 `AutoModel`，这样方便换

保存模型： `.save_pretrained(目录)`

- `config.json` 模型配置文件，存储模型结构参数，例如 Transformer 层数、特征空间维度等
- `pytorch_model.bin` 又称为 state dictionary，存储模型的权重

### 分词器

将文本转换为数字的过程称为编码（Encoding），步骤：
1. **分词**：使用分词器（tokenizer）将文本切分成 tokens
2. **映射**：将所有 token 映射到对应的 token ID

分词策略
- 按词切分（Word-based）
    - 按空格切分
    - 按符号切分
    - 缺点：所有出现过的独立片段都作为不同的 token，从而产生巨大的词表。而实际上很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果给它们赋予不同的编号就无法表示出这种关联性
    - 如果分词结果中包含很多 `[UNK]` 就意味着丢失了很多文本信息
- 按字符切分（Character-based）
    - 只会产生一个非常小的词表，并且很少会出现词表外的 tokens
    - 切分出的 tokens 会很多
- 按子词切分（Subword-based）：高频词直接保留，低频词被切分为更有意义的子词
    - 只用一个较小的词表就可以覆盖绝大部分文本，基本不会产生 unknown token

分词器的加载与保存使用 `Tokenizer.from_pretrained()` 和 `Tokenizer.save_pretrained()`

- `special_tokens_map.json` 映射文件，里面包含 unknown token 等特殊字符的映射关系
- `tokenizer_config.json` 分词器配置文件，存储构建分词器需要的参数
- `vocab.txt` 词表，一行一个 token，行号就是对应的 token ID（从 0 开始）

文本解码 (Decoding) 与编码相反，负责将 token IDs 转换回原来的字符串。**注意，解码过程不是简单地将 token IDs 映射回 tokens，还需要合并那些被分为多个 token 的单词**。分词器都有 `encode`、`decode` 函数