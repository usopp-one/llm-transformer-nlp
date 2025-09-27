import torch
from torch import Tensor, nn
from transformers import BertConfig


class Embedding(nn.Module):
    """
    让模型自己学习位置嵌入的方式（预训练数据集足够大的情况下比较好）
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        # 词嵌入就是普通的词嵌入
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        # 位置嵌入也是普通的嵌入层（查找表）
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids: Tensor) -> Tensor:
        seq_length = input_ids.shape[1]
        # 就是下标代表的位置信息 0, 1, ... 形状是 (B, L)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)

        emb = token_emb + position_emb
        emb = self.dropout(self.layer_norm(emb))

        return emb


if __name__ == "__main__":
    from llm_transformer_nlp.config import get_bert_config, get_bert_tokenizer

    config = get_bert_config()
    tokenizer = get_bert_tokenizer()
    embedding = Embedding(config)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs_emb = embedding(inputs.input_ids)
    print(
        "-" * 80 + f"\n分词结果 {inputs.input_ids.shape = }"
        f"\n嵌入层结果 {inputs_emb.shape = }"
    )
