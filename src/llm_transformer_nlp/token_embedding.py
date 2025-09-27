from torch import nn
from transformers import BertConfig

__all__ = ["get_embedding"]


def get_embedding(config: BertConfig) -> nn.Embedding:
    return nn.Embedding(
        num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
    )


if __name__ == "__main__":
    from llm_transformer_nlp.config import get_bert_config, get_bert_tokenizer

    config = get_bert_config()
    tokenizer = get_bert_tokenizer()

    embedding = get_embedding(config)
    print("-" * 80 + f"\n词嵌入: {embedding = }")

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs_emb = embedding(inputs.input_ids)
    print(
        "-" * 80 + f"\n输入序列大小 (B, L) : {inputs.input_ids.shape = }"
        f"\n词嵌入结果 (B, L, D) : {inputs_emb.shape = }"
    )
