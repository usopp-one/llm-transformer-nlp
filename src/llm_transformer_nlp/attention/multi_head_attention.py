import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer, BertConfig, BertTokenizerFast

from llm_transformer_nlp.attention.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


class AttentionHead(nn.Module):
    # 实践中一般把 head_dim 设置成 embed_dim // num_heads
    # 最终在多头注意力结果拼接后长度 = embed_dim
    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()

        # 每个都能把输入处理成 (b, seq_len, head_dim) 的形状
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor | None = None,
        key_mask: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        embed_dim: int = config.hidden_size
        num_heads: int = config.num_attention_heads
        head_dim: int = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )

        # in_features = num_head * head_dim
        #             = num_head * (embed_dim / num_head)
        #             = embed_dim
        # out_features = embed_dim
        self.output_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor | None = None,
        key_mask: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        x = torch.cat(
            [h(query, key, value, query_mask, key_mask, mask) for h in self.heads],
            dim=-1,
        )
        x = self.output_linear(x)
        return x


if __name__ == "__main__":
    model_ckpt = "bert-base-uncased"
    # huggingface 的 transformers 库，自动从本地目录/huggingface.co找到你要加载的模型
    config: BertConfig = AutoConfig.from_pretrained(model_ckpt)
    print(f"模型配置 {type(config) = }")
    print(f"\t词汇表大小 {config.vocab_size = }")
    print(f"\t模型核心参数 {config.hidden_size = }")
    print(f"\t注意力头数 {config.num_attention_heads = }")

    # 这里的分词器就只是把字符映射到对应的 id 上
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_ckpt)
    print(f"\n分词器层 {type(tokenizer) = }")

    # 嵌入层：一个简单的查找表
    token_emb = nn.Embedding(
        num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
    )
    print(f"\n词嵌入层 {token_emb = }")

    multihead_attn = MultiHeadAttention(config)
    print(f"\n多头注意力层 {multihead_attn = }")

    # 输入有五个词，这五个都在词汇表里 seq_len = 5
    text = "time flies like an arrow"
    print(f"\n1. 输入文字 {text = }")
    # inputs.input_ids 形状： (b, seq_len)   b=1
    inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False)
    print("2. 分词后的 inputs:", inputs, type(inputs))
    # inputs_embeds 形状：(b, seq_len, d_model)    b=1
    inputs_embeds: Tensor = token_emb(inputs.input_ids)
    print(
        "3. 词嵌入后 inputs_embeds:",
        inputs_embeds.shape,
        inputs_embeds,
        type(inputs_embeds),
    )
    # query、key、value 都取输入，这就是“自”注意力了
    query = key = value = inputs_embeds
    # 多头自注意力最终输出形状 (b, n, d_model)
    attn_output: Tensor = multihead_attn(query, key, value)
    print("4. 多头自注意力输出 attn_output:", attn_output.shape)
