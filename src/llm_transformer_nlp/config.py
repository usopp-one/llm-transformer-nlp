from transformers import AutoConfig, AutoTokenizer, BertConfig, BertTokenizerFast

MODEL_BERT_CKPT = "bert-base-uncased"


def get_bert_config() -> BertConfig:
    # 根据 huggingface 模型名加载配置
    # google/bert-base-uncased
    # 结果存储在 ~/.cache/huggingface/hub/models--bert-base-uncased
    config = AutoConfig.from_pretrained(MODEL_BERT_CKPT)
    assert isinstance(config, BertConfig)

    print(
        "-" * 80 + f"\n加载模型配置: {MODEL_BERT_CKPT} {type(config)}"
        f"\n词汇表大小: {config.vocab_size = }"
        f"\n模型大小 D : {config.hidden_size = }"
        f"\n注意力头数 : {config.num_attention_heads = }"
    )
    return config


def get_bert_tokenizer() -> BertTokenizerFast:
    # 根据 huggingface 模型名加载分词器
    # google/bert-base-uncased
    # 结果存储在 ~/.cache/huggingface/hub/models--bert-base-uncased
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(MODEL_BERT_CKPT)
    return tokenizer


if __name__ == "__main__":
    config = get_bert_config()
    print(config)

    tokenizer = get_bert_tokenizer()
    print(tokenizer)

    text = "time flies like an arrow"

    # return_tensors="pt" 返回 tensor 的类型，pt： pytorch
    # .venv.lib.python3.13.site-packages.transformers.utils.generic.TensorType
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(inputs)
    print("-" * 80 + f"\n输入序列大小 (B, L) : {inputs.input_ids.shape = }")
