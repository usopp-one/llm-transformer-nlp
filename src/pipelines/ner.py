from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# [
#   {'entity_group': 'PER', 'score': np.float32(0.9981694),
#       'word': 'Sylvain', 'start': 11, 'end': 18},
#    # grouped_entities=True 的作用让 Hugging 和 Face 两个 token 合并成了一个词输出
#   {'entity_group': 'ORG', 'score': np.float32(0.97960204),
#       'word': 'Hugging Face', 'start': 33, 'end': 45},
#   {'entity_group': 'LOC', 'score': np.float32(0.9932106),
#       'word': 'Brooklyn', 'start': 49, 'end': 57}
# ]
print(result)
