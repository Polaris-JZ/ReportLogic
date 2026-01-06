import re
import torch
import json
from itertools import islice, zip_longest
from transformers import AutoTokenizer
from aistudio_checkpoint.aistudio_base_checkpointer import lazy_load_checkpoint
import re
from openrlhf.utils.utils import recover_original_order, distributed_order
import os

if os.path.exists("/path/to/llm_pretrain/model_hub/Qwen3-30B-A3B"):
    path = "/path/to/llm_pretrain/model_hub/Qwen3-30B-A3B"
else:
    path = lazy_load_checkpoint("aistudio://10172182/Qwen3-30B-A3B")
tokenizer = AutoTokenizer.from_pretrained(path)


#
# mixed_language_template = """## 任务:
#     给你一个答案（[答案开始]和[答案结束]中间的部分）
#     请以专业审校员身份，按以下流程对答案内容进行结构化审查，判断答案不合理中英文混排的数量
# ## 约束条件:
#     - 遵循逻辑性和清晰性原则，避免生成与事实不符的信息。
#     - 严格按照审查中的要求，先找出所有的英文单词、英文短语、英文简称，然后进行判断；
#
# ## 第一步审查，审查是否存在中英文混排，比如包含但不限于以下问题：
#     1、内容存在无意义的拼音混排，例如：九月九日 sanjuxi、来自王zhao的
#     2、内容存在特殊的英文混排，例如：这个app的UI设计很user-friendly
#     3、内容存在一些专业术语，但是和中文混在一起而无法区分其含义，例如：他在篮球场上是PF，很擅长Backboard以及Fast Break得分
#     4、有一些前后不搭的中英文连在一起，例如：你好啊look
#     5、内容存在一些突兀的英文语句或段落，和上下文完全不搭，与中文部分也毫无关联
#     6、如果存在5个以下的英文单词和中文连在一起，即使符合语义，也认为是中英文混排
#
# ## 第二步审查，如果存在下面的特殊情况。则认为是正常的：
#     1、内容是一些大模型或人工智能方面的，涉及一些专业术语的简称或简称是易于理解而且和上下文相关的，例如AI、SFT等；注意不能把完整的非中文词汇也当成简称！
#     2、内容存在一些拼音、英文单词、英文短语等，但是属于翻译、文字或语句解释、文章解释、诗词解释、歌名、书名、地名、公司名称、知名品牌名称、名人姓名、知名产品名称、习惯性的口语单词等
#     3、内容有些上述没列到，但是合理出现的英文单词或短语，或者被括号、书名号、引号等结构包括的
#
# ## 约束条件:
#     - 避免生成与事实不符的信息。
#     - 遵循逻辑性和清晰性原则。
#     - 质量评分理由必须使用中文表达，不超过200字。
# ## 思考流程要求，你需要严格遵循以下思考路径：
#     1.步骤一：逐段的解析回答，判断存在中英混排，以及判断中英混排是否合理
#     2.步骤二：将所有的不合理的中英混排列举并计数
#     3.步骤三：将最终的不合理的中英混排数放在\\boxed{{}}中
# ## 下面是要审查的答案部分：
#     [答案开始]
#     {response}
#     [答案结束]"""


judge_template = judge_prompt = """# 指令

您是一位专业评估专家。您的任务是评估两个AI模型生成回答的质量。
我们将为您提供用户问题及一对AI生成的回答（回答A和回答B）。
您应当首先仔细阅读用户问题，然后根据以下提供的规则评估回答的质量。

# 用户与AI之间的对话

## 用户问题
<|begin_of_query|>

{question}

<|end_of_query|>

## 回答A
<|begin_of_response_A|>

{prediction1}

<|end_of_response_A|>

## 回答B
<|begin_of_response_B|>

{prediction2}

<|end_of_response_B|>

# 评估

## 检查清单
检查清单分为两个部分：
核心清单与非核心清单，前三条为核心清单。如果任意回复违反了前三条检查清单，其属于较差的回答。

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

请参考此检查清单来评估回答的质量，但不要局限于此检查清单。

## 规则

您应当基于用户查询，分析比较上述两种回答。
您应当基于检查清单写下您的分析，然后提供您的评价。
有五个选项供您做出最终评估：["A>>B", "A>B", "A=B", "B>A", "B>>A"]，它们对应如下含义：

- `A>>B`：回答A远胜于回答B。
- `A>B`：回答A略优于回答B。
- `A=B`：回答A和回答B质量相同。请谨慎使用此选项。
- `B>A`：回答B略优于回答A。
- `B>>A`：回答B远胜于回答A。


## 输出格式
首先，请输出您对每个模型回答的分析，
然后对比总结您的评估，最后做出您对最终评估的选择。

请按照以下json格式提供您的评估结果，通过填充[]中的占位符：
```
    "回答A的分析": "[依照checklist逐条分析回答A的内容质量]",
    "回答B的分析": "[依照checklist逐条分析回答B的内容质量]",
    "总结分析": "[对回答A和回答B的分析进行总结，选择一个更好的回答。]",
    "choice": "[A>>B or A>B or A=B or B>A or B>>A]",
```
"""

# GROUP_SIZE=8
# WORLD_SIZE=8
# def extract_mixed_lang_cnt(text):
#     matches = re.findall(r'\\boxed\{(\d+)\}', text)
#     if len(matches) > 0:
#         return int(matches[-1])
#     else:
#         return 0

def extract_raw_prompt(prompt):
    # 定义正则表达式
    pattern = r'<role>HUMAN</role>(.*?)<role>ASSISTANT</role>'
    # 使用 re.search 查找匹配
    match = re.search(pattern, prompt, re.DOTALL)
    assert match is not None
    return match.group(1).strip()  # group(1) 是捕获组中的内容

# 准备vllm 推理内容
def vllm_reward_format_func(queries, prompts, labels, args):
    # batch_size * n
    # 2 * 4

    # query: prompt + response
    # prompt: prompt
    # labels:
    WORLD_SIZE = args.actor_num_nodes * args.actor_num_gpus_per_node
    GROUP_SIZE = args.n_samples_per_prompt

    # print(f"...sxq...queries:{type(queries)}, prompts:{type(prompts)}, labels:{type(labels)}")
    # print(f"...sxq...queries:{queries[0]}, prompts:{prompts[0]}, labels:{labels[0]}")

    print(f"...sxq...before len:{len(queries)}, WORLD_SIZE:{WORLD_SIZE}, GROUP_SIZE:{GROUP_SIZE}")  #...sxq...before len:3200, WORLD_SIZE:8, GROUP_SIZE:8
    queries = recover_original_order(WORLD_SIZE, queries)
    prompts = recover_original_order(WORLD_SIZE, prompts)
    labels = recover_original_order(WORLD_SIZE, labels)
    formated_prompt_for_llm = []
    # print(f"...sxq...after len:{len(queries)}")  #...sxq...after len:3200  上来先做个eval，200*16（rollout answer）=3200  3200/4(vllm_num_engines)=800  judge:位置2次+语言1次，3200*2=6400 6400/4(vllm_num_engines)=1600

    label_json = json.loads(labels[0])
    if 'type' in label_json:
        GROUP_SIZE = 8
        rbs = int(len(queries) // GROUP_SIZE)
    else:
        rbs = int(len(queries) // GROUP_SIZE)
        assert args.rollout_batch_size == rbs


    # for query, prompt, label in zip(queries, prompts, labels):
    #     formated_prompt_for_llm.append(
    #         tokenizer.apply_chat_template(
    #             [{
    #                 'role': 'user',
    #                 'content': mixed_language_template.format(response=query.replace(prompt, "").strip().strip("<|endoftext|>"))
    #             }],
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             enable_thinking=False
    #         )
    #     )

    for index in range(rbs):
        group_queries = queries[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_prompts = prompts[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_labels = labels[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        assert group_prompts[0]==group_prompts[-1]
        assert group_labels[0]==group_labels[-1]
        raw_prompt = extract_raw_prompt(group_prompts[0])
        label_json = json.loads(group_labels[0])
        checklist_string = ""
        for c_index, checkpoint in enumerate(label_json['checklist']):
            checklist_string += f"{c_index+1}. {checkpoint}\n"

        if 'type' in label_json and label_json['type'] == 'eval':
            for i in range(GROUP_SIZE):
                rsp_i = group_queries[i].replace(group_prompts[i], "").strip().strip("<|endoftext|>")
                rsp_j = label_json['ground_truth_response']

                # fomated_prompt = prompt_template.format(dialog=raw_prompt, response1=rsp_i, response2=rsp_j)
                fomated_prompt = judge_template.format(question=raw_prompt, prediction1=rsp_i, prediction2=rsp_j, checklist=checklist_string)
                formated_prompt_for_llm.append(
                    tokenizer.apply_chat_template(
                        [{
                            'role': 'user',
                            'content': fomated_prompt
                        }],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                )
                fomated_prompt = judge_template.format(question=raw_prompt, prediction1=rsp_j, prediction2=rsp_i, checklist=checklist_string)
                formated_prompt_for_llm.append(
                    tokenizer.apply_chat_template(
                        [{
                            'role': 'user',
                            'content': fomated_prompt
                        }],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                )
        else:
            for i in range(GROUP_SIZE):
                for j in range(i+1, GROUP_SIZE):
                    rsp_i = group_queries[i].replace(group_prompts[i], "").strip().strip("<|endoftext|>")
                    rsp_j = group_queries[j].replace(group_prompts[j], "").strip().strip("<|endoftext|>")
                    fomated_prompt = judge_template.format(question=raw_prompt, prediction1=rsp_i, prediction2=rsp_j, checklist=checklist_string)
                    formated_prompt_for_llm.append(
                        tokenizer.apply_chat_template(
                            [{
                                'role': 'user',
                                'content': fomated_prompt
                            }],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    )
                    fomated_prompt = judge_template.format(question=raw_prompt, prediction1=rsp_j, prediction2=rsp_i, checklist=checklist_string)
                    formated_prompt_for_llm.append(
                        tokenizer.apply_chat_template(
                            [{
                                'role': 'user',
                                'content': fomated_prompt
                            }],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    )
    print(f"len of formated_prompt_for_llm: {len(formated_prompt_for_llm)}")
    prompt_type = 'eval' if 'type' in label_json else 'train'
    # with open(f"{args.save_path}/mixed_lang_cnt_judge_prompt_{prompt_type}.jsonl", 'a+') as wf:
    #     for prompt in formated_prompt_for_llm[:len(queries)]:
    #         wf.write(prompt + "\n")
    with open(f"{args.save_path}/judge_prompt_{prompt_type}.jsonl", 'a+') as wf:
        for prompt in formated_prompt_for_llm:
            # wf.write(prompt + "\n")
            item={'prompt': prompt}
            wf.write(json.dumps(item, ensure_ascii=False)+"\n")

    return formated_prompt_for_llm  #[bs * n]

def extract_score(judgment, response_1, response_2, exist_cnt_1, exist_cnt_2, ):
    if exist_cnt_1-exist_cnt_2 <= -3:
        score_1, score_2 = 2, -2
    elif exist_cnt_1-exist_cnt_2 >= 3:
        score_1, score_2 = -2, 2
    else:
        pattern = r'\"choice\": \"(.*?)\"'
        pattern = re.compile(pattern)
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            score_1, score_2 = 0, 0
        elif len(set(matches)) == 1:
            judge = matches[0].strip("\n")
            all_results = ['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']
            all_present = []
            decision = ""
            for result in all_results:
                if result in judge:
                    all_present.append(1)
                    decision = result
                else:
                    all_present.append(0)
            if sum(all_present) != 1:
                # 确保存在且仅存在一种情况
                score_1, score_2 = 0, 0
            elif decision in ['A>>B', 'B>>A']:
                score_1, score_2 = {
                    'A>>B': (2, -2),
                    'B>>A': (-2, 2)
                }[decision]
            elif decision == 'A>B':
                score_1, score_2 = 1, -1
                # length_reward = min(0.5, max(-0.5, 1-len(response_1)/len(response_2)))
                # score_1 = score_1 + length_reward
                # score_2 = score_2 - length_reward
            elif decision == 'B>A':
                score_1, score_2 = -1, 1
                # length_reward = min(0.5, max(-0.5, 1-len(response_2)/len(response_1)))
                # score_1 = score_1 - length_reward
                # score_2 = score_2 + length_reward
            else:
                score_1, score_2 = 0, 0
                # length_reward = min(0.5, max(-0.5, 1-len(response_1)/len(response_2)))
                # score_1 = score_1 + length_reward
                # score_2 = score_2 - length_reward
        else:
            score_1, score_2 = 0, 0

    # mixed_lang_score_1 = min(1, exist_cnt_1*0.2)
    # mixed_lang_score_2 = min(1, exist_cnt_2*0.2)
    #
    # score_1 = score_1 - mixed_lang_score_1
    # score_2 = score_2 - mixed_lang_score_2
    return score_1, score_2

def vllm_reward_post_process_func(vllm_reward_responses, queries, prompts, labels, args):
    """
    计算reward值
    :param vllm_reward_responses: reward_vllm_engines output: bs * n
    :param queries: prompt + response
    :param prompts: prompt
    :param labels: 标签
    :return: reward值，需要为tensor float类型

    在这里组装prompts，然后输入到generate_func中，获取生成式模型的输出
    """
    WORLD_SIZE = args.actor_num_nodes * args.actor_num_gpus_per_node
    GROUP_SIZE = args.n_samples_per_prompt

    print(f"...sxq2...before len:{len(queries)}, WORLD_SIZE:{WORLD_SIZE}, GROUP_SIZE:{GROUP_SIZE}")
    queries = recover_original_order(WORLD_SIZE, queries)
    prompts = recover_original_order(WORLD_SIZE, prompts)
    labels = recover_original_order(WORLD_SIZE, labels)
    # print(f"...sxq2...after len:{len(queries)}")

    label_json = json.loads(labels[0])
    if 'type' in label_json:
        GROUP_SIZE = 8
        rbs = int(len(queries) // GROUP_SIZE)
    else:
        rbs = int(len(queries) // GROUP_SIZE)
        assert args.rollout_batch_size == rbs

    # vllm_mixed_lang_responses = vllm_reward_responses[:len(queries)]
    # mixed_lang_cnt = [extract_mixed_lang_cnt(response) for response in vllm_mixed_lang_responses]
    # print(f"{sum(mixed_lang_cnt)=}")

    # vllm_reward_responses = vllm_reward_responses[len(queries):]
    if 'type' not in label_json:
        assert len(vllm_reward_responses) == rbs * GROUP_SIZE * (GROUP_SIZE-1)

    rewards = []
    overall_gr = []
    for index in range(rbs):
        group_queries = queries[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_prompts = prompts[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_labels = labels[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_responses = [query.replace(prompt, "").strip().strip("<|endoftext|>") for query, prompt in zip(group_queries, group_prompts)]
        # group_mixed_lang_cnt = mixed_lang_cnt[index * GROUP_SIZE: (index+1) * GROUP_SIZE]

        GROUP_SIZE_VLLM = int(len(vllm_reward_responses) // rbs)
        group_vllm_reward_responses = vllm_reward_responses[index * GROUP_SIZE_VLLM: (index+1) * GROUP_SIZE_VLLM]


        assert group_prompts[0]==group_prompts[-1]
        assert group_labels[0]==group_labels[-1]
        raw_prompt = extract_raw_prompt(group_prompts[0])
        label_json = json.loads(group_labels[0])
        group_rewards = [[] for _ in range(GROUP_SIZE)]

        if 'type' in label_json and label_json['type'] == 'eval':
            assert len(group_vllm_reward_responses) == GROUP_SIZE*2
            for i in range(GROUP_SIZE):
                vllm_reward_response = group_vllm_reward_responses[i*2]
                rsp_i_score, rsp_j_score = extract_score(vllm_reward_response, group_responses[i], label_json['ground_truth_response'], 0, 0)
                group_rewards[i].append(rsp_i_score)

                vllm_reward_response = group_vllm_reward_responses[i*2+1]
                rsp_j_score, rsp_i_score = extract_score(vllm_reward_response, label_json['ground_truth_response'], group_responses[i], 0, 0)
                group_rewards[i].append(rsp_i_score)
        else:
            vllm_index = 0
            for i in range(GROUP_SIZE):
                for j in range(i+1, GROUP_SIZE):
                    vllm_reward_response = group_vllm_reward_responses[vllm_index]
                    vllm_index += 1
                    rsp_i_score, rsp_j_score = extract_score(vllm_reward_response, group_responses[i], group_responses[j], 0, 0)
                    # For train
                    group_rewards[i].append(rsp_i_score)
                    group_rewards[j].append(rsp_j_score)

                    vllm_reward_response = group_vllm_reward_responses[vllm_index]
                    vllm_index += 1
                    rsp_j_score, rsp_i_score = extract_score(vllm_reward_response, group_responses[j], group_responses[i], 0, 0)
                    # For train
                    group_rewards[i].append(rsp_i_score)
                    group_rewards[j].append(rsp_j_score)
            assert vllm_index == len(group_vllm_reward_responses)

        # print(f"...sxq...group_rewards:{group_rewards}")
        for gr in group_rewards:
            rewards.append(sum(gr))
        overall_gr.extend(group_rewards)

    prompt_type = 'eval' if 'type' in label_json else 'train'

    with open(f"{args.save_path}/v5_lang_judge_result_{prompt_type}.jsonl", 'a+') as wf:
        for index in range(8):
            group_queries = queries[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
            group_prompts = prompts[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
            group_labels = labels[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
            group_responses = [query.replace(prompt, "").strip().strip("<|endoftext|>") for query, prompt in zip(group_queries, group_prompts)]
            # group_mixed_lang_cnt = mixed_lang_cnt[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
            group_rewards = rewards[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
            GROUP_SIZE_VLLM = int(len(vllm_reward_responses) // rbs)
            group_vllm_reward_responses = vllm_reward_responses[index * GROUP_SIZE_VLLM: (index+1) * GROUP_SIZE_VLLM]
            group_overall_gr = overall_gr[index * GROUP_SIZE: (index+1) * GROUP_SIZE]

            response_info = []
            for in_idx in range(len(group_responses)):
                response_info.append(
                    {
                        'response': group_responses[in_idx],
                        # 'mixed_lang_cnt': group_mixed_lang_cnt[in_idx],
                        'reward': group_rewards[in_idx],
                        'llm_judge': group_overall_gr[in_idx]
                    }
                )
            group_info = {
                'prompt': group_prompts[0],
                'label': group_labels[0],
                'vllm_reward_responses': group_vllm_reward_responses,
                'response_info': response_info
            }
            wf.write(json.dumps(group_info, ensure_ascii=False) + "\n")

    rewards = distributed_order(WORLD_SIZE, rewards)
    return torch.tensor(rewards).to(torch.float32)


class ArgsDemo:
    def __init__(self):
        self.actor_num_nodes = 1
        self.actor_num_gpus_per_node = 4
        self.n_samples_per_prompt = 8
        self.rollout_batch_size = 1024
