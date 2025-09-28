from transformers import pipeline

answer = pipeline("question-answering")

result = answer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn.",
)

print(result)


result = answer(
    question="Which city do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn.",
)
print(result)
