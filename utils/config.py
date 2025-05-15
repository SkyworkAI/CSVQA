'''
This file is config file to save the model informations, system prompts and the chat templates of each model.
When you want to add new model for evaluation, you should first set the information of the new model in this config file.
'''

# model name and the local save path
model_name_to_path = {
    'internvl2_5_78b': '/mnt/data_vlm/models/InternVL2_5-78B',
    'internvl3_78b': '/mnt/data_vlm/models/InternVL3-78B',
    'qwen2_5_vl_72b_instruct': '/mnt/data_vlm/models/Qwen2.5-VL-72B-Instruct',
}


# key 0 zh
# key 1 en
input_files = {
    0: '/mnt/datasets_vlm/jianai/Benchmark/finaldata/csvqa_data_zh.jsonl',
    1: '/mnt/datasets_vlm/jianai/Benchmark/finaldata/csvqa_data_en.jsonl',
}

# system reference:
# 1. see the official Huggingface repo
system_prompts = {
    'internvl2_5_78b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl3_78b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'qwen2_5_vl_72b_instruct': 'You are a helpful assistant.',
}


# template reference:
# 1. see the official Huggingface repo
# 2. reference to the vllm repo: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py

model_name_to_template = {
    'internvl2_5_78b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl3_78b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'qwen2_5_vl_72b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
}