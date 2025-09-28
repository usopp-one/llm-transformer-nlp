from transformers import AutoTokenizer, pipeline

# pipeline 会自动选择合适的预训练模型来完成任务
# 这里自动选择的是 distilbert/distilbert-base-uncased-finetuned-sst-2-english
# 第一次执行会下载模型
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life")
print(result)
results = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(results)


tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
inputs = tokenizer(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ],
)
print(inputs)


tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
inputs = tokenizer(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ],
    padding=True,
)
print(inputs)
