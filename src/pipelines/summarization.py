from transformers import pipeline

summarization = pipeline("summarization")
results = summarization(
    """The model is pre-trained by UER-py, which is introduced in this paper.
Besides, the model could also be pre-trained by TencentPretrain introduced
in this paper, which inherits UER-py to support models with parameters above
one billion, and extends it to a multimodal pre-training framework.

The model is used to generate Chinese ancient poems. You can download the model
from the UER-py Modelzoo page, or GPT2-Chinese Github page, or via HuggingFace from
the link gpt2-chinese-poem.

Since the parameter skip_special_tokens is used in the pipelines.py, special 
tokens such as [SEP], [UNK] will be deleted, the output results of Hosted 
inference API (right) may not be properly displayed."""
)

print(results)
