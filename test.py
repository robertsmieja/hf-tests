from transformers import pipeline
import torch

sample_input = "Is this working? Hello world?"

# classifier = pipeline("sentiment-analysis")
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
)

# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-9b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",  # replace with "mps" to run on a Mac device
# )

generator = pipeline(
    task="text-generation",
    model="openai-community/gpt2",
    revision="607a30d",
    # model="google/gemma-2-9b-it",
    # model_kwargs={"torch_dtype": torch.bfloat16},
    # device="cuda",  # replace with "mps" to run on a Mac device
)

# result = classifier(sample_input)
# print(result)


import torch.utils.benchmark as benchmark

sentiment_timer = benchmark.Timer(
    stmt="classifier(sample_input)",
    globals={"classifier": classifier, "sample_input": sample_input},
)

print(sentiment_timer.timeit(100))

text_gen_timer = benchmark.Timer(
    stmt="generator(sample_input, pad_token_id=generator.tokenizer.eos_token_id)",
    globals={"generator": generator, "sample_input": sample_input},
)

print(text_gen_timer.timeit(10))
