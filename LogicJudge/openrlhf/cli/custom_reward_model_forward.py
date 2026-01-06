import torch
import torch.nn.functional as F

def extract_prompt(prompt):
    pass

def forward_func(model, tokenizer, queries, prompts, labels):
    responses = [query.replace(prompt) for query, prompt in zip(queries, prompts)]
    prompts = []
    texts = [tokenizer.apply_chat_template(
      [
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': response}
      ],
      tokenize=False,
      add_generation_prompt=False
    ) for query, response in zip(queries, responses)]





    pass
