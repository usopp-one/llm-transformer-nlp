from torch import Tensor, nn
from transformers import BertConfig

from llm_transformer_nlp.attention.multi_head_attention import MultiHeadAttention
from llm_transformer_nlp.encoder.feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    """
    pre LN
    前置层归一化
    """

    def __init__(self, config: BertConfig):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # 先做层归一化，再进多头自注意力
        hidden_state = self.layer_norm_1(x)

        # 经过多头自注意力，然后做残差连接
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)

        # 先做第二次层归一化，然后进前馈神经网络
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x


if __name__ == "__main__":
    from llm_transformer_nlp.config import get_bert_config, get_bert_tokenizer
    from llm_transformer_nlp.sample_token_embedding import get_embedding

    config = get_bert_config()
    tokenizer = get_bert_tokenizer()

    embedding = get_embedding(config)
    encoder_layer = TransformerEncoderLayer(config)
    print("-" * 80 + f"\n词嵌入 {embedding = }\n编码器层 {encoder_layer = }")

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs_embes = embedding(inputs.input_ids)
    encoder_output = encoder_layer(inputs_embes)
    print(
        "-" * 80 + f"\n词嵌入 (B, L, D) {inputs_embes.shape = }"
        f"\n编码器输出 (B, L, D) {encoder_output.shape = }"
    )
