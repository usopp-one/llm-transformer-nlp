from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer, BertConfig

if __name__ == "__main__":
    print("\n 将文本分词为 token 序列（词索引 int 序列）")

    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_ckpt)
    print(f"{type(tokenizer) = }")

    text = "time flies like an arrow"
    # return_tensors="pt" 返回 tensor 的类型，pt： pytorch
    # .venv.lib.python3.13.site-packages.transformers.utils.generic.TensorType
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(f"{type(inputs) = }")
    print(f"{inputs.input_ids = }")
    print(f"{inputs.input_ids.shape = }")

    print("\n把 token 转为词向量/嵌入向量")

    config: BertConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_ckpt
    )
    print(f"{type(config) = }")
    # vocab_size 是词汇表大小
    # hidden_size 就是词向量维度，也是模型维度， bert-base-uncased 中是 768
    # https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json
    print(f"{config.vocab_size = }, {config.hidden_size = }")

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    # nn.Embedding 就是一个形状为 (词汇表大小, 嵌入向量/词向量维度) 的查找矩阵，
    # 会将输入词索引直接映射为对应的词向量
    #   矩阵中的数据即参数是可以训练的
    print(f"词嵌入器（查找矩阵）形状 {token_emb = }")
    inputs_embeds: torch.Tensor = token_emb(inputs.input_ids)
    print(f"{type(inputs_embeds) = }")
    print(f"输入形状 (b, n, d_model) {inputs_embeds.size() = }")
    print(inputs_embeds)
    print()

    print("\n计算 Q、K、V 矩阵")

    # (b, n, d_model)
    # Q: (b, m, d_k) K: (b, n, d_k) V: (b, n, d_v)
    Q = K = V = inputs_embeds
    dim_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)
    print(f"QK^T 的形状 (b,m,n)，由于m=n，这里就是 (b,n,n) {scores.size() = }")
    print(f"{scores = }")

    weights = F.softmax(scores, dim=-1)
    print(f"在最后一个维度上 softmax 之后，和应该为 1, {weights.sum(dim=-1) = }")

    # 注意力权重与 value 相乘，得到的每一行都是之前所有 value 向量的加权求和
    # 所以结果形状是 (b,n,d_model)
    attn_outputs = torch.bmm(weights, V)
    print(f"注意力输出形状： {attn_outputs.shape=}")


# 封装一个函数
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_mask: Tensor | None = None,
    key_mask: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """
    query:    (b, m, d_k)
    key:      (b, n, d_k)
    value:    (b, n, d_v)

    return:   (b, m, d_v)
    """
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        # mask 值为 0 的地方置为 -inf，经过 softmax 变为 0
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
