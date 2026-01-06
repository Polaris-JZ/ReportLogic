import re
import torch

CHOICES = ['A>B', 'A=B', 'A<B', 'both_bad']

def check_output_format(text):
    analysis = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    pred = re.findall(r'</think>\n\n(.*?)<\|im_end\|>', text, re.DOTALL)
    if len(analysis) == 1 and bool(analysis[0].strip()) and len(pred) == 1 and pred[0] in CHOICES:
        return pred[0]
    else:
        return None

def get_reward(query, prompt, label):
    """
    query:
        <|im_start|>user
        <prompt><|im_end|>
        <|im_start|>assistant
        <think><analysis></think>
        
        <result><|im_end|>
    prompt:
        <|im_start|>user
        <prompt><|im_end|>
        <|im_start|>assistant

    label:
        <result>
    """
    output = query.replace(prompt, '')
    valid_output = check_output_format(output)
    if valid_output is None:
        return -1.0
    
    if valid_output == label:
        return 1.0
    else:
        return -0.5

def reward_func(queries, prompts, labels):
    return torch.tensor([
        get_reward(query, prompt, label)
        for query, prompt, label in zip(queries, prompts, labels)
    ])

if __name__ == '__main__':
    queries = [
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<think><analysis></think>\n\nA<B<|im_end|>',
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<analysis></think>\n\nA=B<|im_end|>',
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<think><analysis></think>\n\nA<B<|im_end|>',
    ]
    prompts = [None] * len(queries)
    labels = ['A<B', 'A=B', 'A>B']
    print(reward_func(queries, prompts, labels))