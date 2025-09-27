from transformers import pipeline

# 这里使用的默认模型为 facebook/bart-large-mnli
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(result)
