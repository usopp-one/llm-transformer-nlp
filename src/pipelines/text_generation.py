from transformers import pipeline

# No model was supplied, defaulted to openai-community/gpt2 and revision 607a30d
generator = pipeline("text-generation")

results_1 = generator("In this course, we will teach you how to")
print(results_1)

# 这个最大长度并没有生效
# Both `max_new_tokens` (=256) and `max_length`(=20) seem to have been set.
# `max_new_tokens` will take precedence.
# Please refer to the documentation for more information.
# (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
results_2 = generator(
    "In this course, we will teach you how to", num_return_sequences=2, max_length=20
)
print(results_2)

# 默认的这个 openai-community/gpt2 中文支持的不好
results_3 = generator("天上的星星不说话，地上的", num_return_sequences=2, max_length=20)
print(results_3)

# 在 huggingface 上左侧有标签可以筛选，筛选 「Text Generation」 的就能找到其他支持中文的
# uer/gpt2-chinese-poem 是教程作者示例中使用的
# "[CLS]梅 山 如 积 翠 ，" 是 uer/gpt2-chinese-poem 说明文档中的示例
chinese_generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results_4 = chinese_generator(
    "[CLS]梅 山 如 积 翠 ，", max_new_tokens=30, max_length=30, do_sample=True
)
print(results_4)
