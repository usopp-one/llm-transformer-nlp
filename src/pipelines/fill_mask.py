from transformers import pipeline

# No model was supplied, defaulted to distilbert/distilroberta-base and revision fb53ab8
filler = pipeline("fill-mask")

results = filler("This course will teach you all about <mask> models.", top_k=3)
print(results)

results = filler("I come from <mask>, I like play football.", top_k=5)
print(results)
