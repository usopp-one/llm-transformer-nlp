from torch import Tensor, nn
from transformers import BertConfig


class FeedForward(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        # 中间层一般是网络大小的 4 倍
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear_2(self.gelu(self.linear_1(x))))


if __name__ == "__main__":
    from llm_transformer_nlp.attention.multi_head_attention import MultiHeadAttention
    from llm_transformer_nlp.config import get_bert_config, get_bert_tokenizer

    config = get_bert_config()
    print(f"配置：{config = }")

    tokenizer = get_bert_tokenizer()
    emb = nn.Embedding(config.vocab_size, config.hidden_size)
    mulithead_attn = MultiHeadAttention(config=config)
    feed_forward = FeedForward(config)
    print()
    print(f"分词器：{tokenizer = }")
    print(f"词嵌入层：{emb = }")
    print(f"多头自注意力：{mulithead_attn = }")
    print(f"前馈网络：{feed_forward = }")

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs_emb: Tensor = emb(inputs.input_ids)
    query = key = value = inputs_emb
    # 多头注意力输出形状 (b, seq_len, d_model)
    attn_output: Tensor = mulithead_attn(query, key, value)
    print()
    print(f"多头注意力输出形状：{attn_output.shape = }")
    # 前馈网络输出形状 (b, seq_len, d_model)
    feed_output: Tensor = feed_forward(attn_output)
    print(f"前馈网络输出形状：{feed_output.shape = }")
