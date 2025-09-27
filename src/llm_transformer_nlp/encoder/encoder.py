from torch import Tensor, nn
from transformers import BertConfig

from llm_transformer_nlp.encoder.layer_normalization import TransformerEncoderLayer
from llm_transformer_nlp.encoder.positional_embedding import Embedding


class TransformerEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.embedding = Embedding(config)
        # 注意有多个模型构成的列表要用 nn.ModuleList 来封装，遮阳才能进行训练
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


if __name__ == "__main__":
    from llm_transformer_nlp.config import get_bert_config, get_bert_tokenizer

    config = get_bert_config()
    tokenizer = get_bert_tokenizer()

    encoder = TransformerEncoder(config)
    print("-" * 80 + f"\n编码器 {encoder = }")

    text = "time flies like an arrow"
    # 分词结果 (B, L)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    # 编码器结果 (B, L, D)
    encoder_output = encoder(inputs.input_ids)
    print(
        "-" * 80 + f"\n分词结果 {inputs.input_ids.shape = }"
        f"\n编码器输出 {encoder_output.shape = }"
    )
