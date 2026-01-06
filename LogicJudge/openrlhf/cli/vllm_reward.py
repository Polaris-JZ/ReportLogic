import re
import torch
import json

from itertools import islice, zip_longest


PROMPT = """请告诉我你的名字是百灵"""


import concurrent.futures

import ast
import traceback


PROMPT =  """
    请判断给定的回复是否遵循指令中的约束, 其中已经逐条从指令中提取了约束：
    
    [指令](instruction):
    %s

    [约束](checkers):
    %s

    [回复](response):
    %s
    
    先简要分析约束遵循情况，判断给定的回复(response)是否遵循指令中的约束(checkers)，比如风格、格式等约束，然后在回答的最后一行用[[YES|No]]输出你的判断。
    如果回复遵循了所有的约束，请输出[[YES]]；如果有约束没有满足，则输出[[NO]]，回答长度不超过300字:
    """

def extract_response(query: str, prompt_end_marker: str) -> str:
    """
    通过查找prompt结束和response开始的标记来提取response。
    
    Args:
        query (str): 完整的 "prompt + response" 字符串。
        prompt_end_marker (str): 标志着prompt结束的唯一字符串，
                                 例如 '<role>ASSISTANT</role>' 或 'ASSISTANT:'。
    
    Returns:
        str: 提取出的response内容。
    """
    # 使用 rfind 从右侧查找，确保找到的是最后一个标记，
    # 这通常是分隔用户和助手回合的那个。
    marker_index = query.rfind(prompt_end_marker)
    
    if marker_index == -1:
        # 如果找不到标记，说明可能格式有问题，返回空字符串或抛出错误
        print(f"警告：在query中找不到分割标记 '{prompt_end_marker}'")
        return ""
        
    # response 的内容从标记之后开始
    response_start_index = marker_index + len(prompt_end_marker)
    response = query[response_start_index:]
    
    # 清理可能存在的特殊结尾标记和前后空格
    return response.strip().strip("<|endoftext|>")

# 准备vllm 推理内容
def vllm_reward_format_func(queries, prompts, labels):

    formated_prompt_for_llm = []
    for query, prompt, label in zip(queries, prompts, labels):
        # 解析 label 字段
        label_json = json.loads(label)
        # 提取 gold 字段
        label = label_json["functions"]
        llm_checkers = []
        for function in label:
            if "llm_call" in function:
                # llm_checkers.append(checker)
                # llm_functions.append(function)
                llm_checkers.append(function.split("return llm_score(instruction, response, ")[1])
        # print(prompt, llm_checkers, query)

        clean_prompt = re.sub(r'^(?:\s*(?:<role>|</role>)[^<]*)+', '', prompt, flags=re.DOTALL)
        # response = extract_response(query, "<role>ASSISTANT</role>")
        response = query.replace(prompt, "").strip().strip("<|endoftext|>")

        formated_prompt_for_llm.append(
            PROMPT % (
                clean_prompt.strip("<role>ASSISTANT</role>"), 
                llm_checkers,
                response
            )
        )
    return formated_prompt_for_llm


def extract_imports(code):
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                imports.append(f"from {module} import {alias.name}")
    return imports

def execute_code(instruction, response, function=None):
    global_context = {}
    local_vars = {"response": response}
    
    try:
        import_statements = extract_imports(function)
        for statement in import_statements:
            exec(statement, global_context)

        # 执行传入的 function 代码
        exec(function, global_context, local_vars)

        # 检查 check_following 是否存在且可调用
        if 'check_following' in local_vars and callable(local_vars['check_following']):
            result = local_vars['check_following']("instruction", response)
            return result, None
        else:
            return None, "Function 'check_following' is missing or not callable"

    except Exception as e:
        error_message = f"Execution error: {e}\n{traceback.format_exc()}"
        print(function + "\n" + error_message)
        return None, error_message


def mean_func(numbers):
    mean_num = sum(numbers) / len(numbers)
    return mean_num



def _evaluate_reward(instruction, response, functions, reduction="mean"):
    def process_function(function):
        result, _ = execute_code(instruction, response, function)
        return 1 if result else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_function, functions))
    if reduction == "mean":
        return mean_func(results)
    else:
        return results

def extract(text):
    if text is None:
        return None
        
    if not isinstance(text, (str, bytes)):
        try:
            text = str(text)  # Attempt conversion
        except:
            return None
    
    pattern = r"(YES|NO)"
    match = re.search(pattern, text)
    return match.group(1).upper() if match else None  # Ensure uppercase



def vllm_reward_post_process_func(vllm_reward_responses, queries, prompts, labels):
    """
    计算reward值
    :param vllm_reward_responses: reward_vllm_engines output
    :param queries: prompt + response
    :param prompts: prompt
    :param labels: 标签
    :return: reward值，需要为tensor float类型

    在这里组装prompts，然后输入到generate_func中，获取生成式模型的输出
    """

    rewards = []
    judge_data_list = []
    

    for idx, llm_judge_output in enumerate(vllm_reward_responses):
        query, prompt, label = queries[idx], prompts[idx], labels[idx]
        clean_prompt = re.sub(r'^(?:\s*(?:<role>|</role>)[^<]*)+', '', prompt, flags=re.DOTALL)
        # response = extract_response(query, "<role>ASSISTANT</role>")
        response = query.replace(prompt, "").strip().strip("<|endoftext|>")


        llm_checkers = []
        rule_functions = []
        # 解析 label 字段
        label_json = json.loads(label)
        # 提取 gold 字段
        label = label_json["functions"]
        llm_checkers = []
        for function in label:
            if "llm_call" in function:
                # llm_checkers.append(checker)
                # llm_functions.append(function)
                llm_checkers.append(function.split("return llm_score(instruction, response, ")[1])

        formated_prompt_for_llm = PROMPT % (
                clean_prompt.strip("<role>ASSISTANT</role>"), 
                llm_checkers,
                response
            )

        for function in label:
            if "llm_call" not in function:
                # rule_checkers.append(checker)
                rule_functions.append(function)
            else:
                pass
                # single processing
        if len(rule_functions) != 0:
            rule_reward = _evaluate_reward(prompt, response, rule_functions, reduction="mean")
        else:
            rule_reward = 0.0

        judge_data = {
            "raw_prompt":prompt, 
            "clean_prompt": clean_prompt.strip("<role>ASSISTANT</role>"),
            "raw_query": query,
            "response": response,
            "label": label,
            "rule_functions": rule_functions,
            "formated_prompt_for_llmjudge":formated_prompt_for_llm,
            "llm_judge_output": llm_judge_output,
            "rule_output": rule_reward,


        }
        
        judge_res = extract(llm_judge_output)
        if judge_res is None:  # 如果没有提取到结果，则采用rule reward进行计算
            judge_data['reward_func'] = 'rule'
            reward = rule_reward
        else:
            judge_data['reward_func'] = 'gen_llm and rule'
            if judge_res == 'YES':
                reward = 1.0 * rule_reward
            else:
                reward = 0.0 * rule_reward
        rewards.append(reward)
        judge_data['reward'] = reward
        judge_data['rule_reward'] = rule_reward
        judge_data['judge_res'] = judge_res
        judge_data_list.append(judge_data)
    
    # 保存judge数据
    try:
        with open(JUDGE_LOG_FILE, 'a', encoding='utf-8') as fw:
            for item in judge_data_list:
                fw.write(json.dumps(item, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"ERROR: Could not rite to judge log file: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during logging: {e}")

    return torch.tensor(rewards)

    