from transformers import pipeline

filler = pipeline("fill-mask")

# No model was supplied, defaulted to distilbert/distilroberta-base and revision fb53ab8
results = filler("This course will teach you all about <mask> models.", top_k=3)
print(results)
