'''
This file is config file to save the model informations, system prompts and the chat templates of each model.
When you want to add new model for evaluation, you should first set the information of the new model in this config file.
'''

# model name and the local save path
model_name_to_path = {
    'internvl2_5_8b': '',
    'internvl2_5_26b': '',
    'internvl2_5_38b': '',
    'internvl2_5_78b': '',
    'internvl3_9b': '',
    'internvl3_14b': '',
    'internvl3_38b': '',
    'internvl3_78b': '',
    'deepseek_vl2_tiny': '',
    'deepseek_vl2_small': '',
    'deepseek_vl2': '',
    'llava_7b': '',
    'llava_13b': '',
    'idefics3_8b': '',
    'fuyu_8b': '',
    'qwen2_vl_7b_instruct': '',
    'qwen2_vl_72b_instruct': '',
    'qwen2_5_vl_7b_instruct': '',
    'qwen2_5_vl_32b_instruct': '',
    'qwen2_5_vl_72b_instruct': '',
    'qvq_72b': '',
    'gemma_3_4b_it': '',
    'gemma_3_27b_it': '',
    'Phi_3_5_vision_instruct': '',
    'Phi_4_multimodal_instruct': '',
    'pixtral_12b': '',
    'mono_internvl_2b': '',
    'llava_onevision_72b': ''
}

# paritial models need to override the architecture
# please to refer to the Huggingface repo, to find the config.json
model_name_to_hf_override = {
    'deepseek_vl2_tiny': 'DeepseekVLV2ForCausalLM',
    'deepseek_vl2_small': 'DeepseekVLV2ForCausalLM',
    'deepseek_vl2': 'DeepseekVLV2ForCausalLM',
}

# key 0 zh
# key 1 en
input_files = {
    0: 'Path to normal input file for Chinese version',
    1: 'Path to normal input file for English version'
}

input_files_direct = {
    0: 'path to direct input file for Chinese version',
    1: 'path to direct input file for English version',
}

# system reference:
# 1. see the official Huggingface repo
system_prompts = {
    'internvl2_5_8b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl2_5_26b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl2_5_38b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl2_5_78b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl3_9b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl3_14b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl3_38b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'internvl3_78b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'deepseek_vl2_tiny': '',
    'deepseek_vl2_small': '',
    'deepseek_vl2': '',
    'llava_7b': '',
    'llava_13b': '',
    'idefics3_8b': '',
    'fuyu_8b': '',
    'qwen2_vl_7b_instruct': 'You are a helpful assistant.',
    'qwen2_vl_72b_instruct': 'You are a helpful assistant.',
    'qwen2_5_vl_7b_instruct': 'You are a helpful assistant.',
    'qwen2_5_vl_32b_instruct': 'You are a helpful assistant.',
    'qwen2_5_vl_72b_instruct': 'You are a helpful assistant.',
    'qvq_72b': 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.',
    'gemma_3_4b_it': '',
    'gemma_3_27b_it': '',
    'Mistral_Small_3_1_24B_Instruct_2503': '',
    'NVLM_D_72B': 'Answer the questions.',
    'Phi_3_5_vision_instruct': '',
    'Phi_4_multimodal_instruct': '',
    'pixtral_12b': '',
    'mono_internvl_2b': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    'llava_onevision_72b': ''
}


# template reference:
# 1. see the official Huggingface repo
# 2. reference to the vllm repo: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py

model_name_to_template = {
    'internvl2_5_8b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl2_5_26b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl2_5_38b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl2_5_78b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl3_9b': ' <|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl3_14b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl3_38b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'internvl3_78b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'deepseek_vl2_tiny': '<|User|>: <|text|>\n\n<|Assistant|>:',
    'deepseek_vl2_small': '<|User|>: <|text|>\n\n<|Assistant|>:',
    'deepseek_vl2': '<|User|>: <|text|>\n\n<|Assistant|>:',
    'llava_7b': 'USER: <|text|>\nASSISTANT:',
    'llava_13b': 'USER: <|text|>\nASSISTANT:',
    'idefics3_8b': '<|begin_of_text|>User:<|text|><end_of_utterance>\nAssistant:',
    'fuyu_8b': '<|text|>\n',
    'qwen2_vl_7b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'qwen2_vl_72b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'qwen2_5_vl_7b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'qwen2_5_vl_32b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'qwen2_5_vl_72b_instruct': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'qvq_72b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|text|><|im_end|>\n<|im_start|>assistant\n',
    'gemma_3_4b_it': '<bos><start_of_turn>user\n<start_of_image><|text|><end_of_turn>\n<start_of_turn>model\n',
    'gemma_3_27b_it': '<bos><start_of_turn>user\n<start_of_image><|text|><end_of_turn>\n<start_of_turn>model\n',
    'Mistral_Small_3_1_24B_Instruct_2503': '<s>[INST]<|text|>\n[IMG][/INST]',
    'NVLM_D_72B': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'Phi_3_5_vision_instruct': '<|user|>\n<|image_1|>\n<|text|><|end|>\n<|assistant|>\n',
    'Phi_4_multimodal_instruct': '<|user|><|image_1|><|text|><|end|><|assistant|>',
    'pixtral_12b': '<s>[INST]<|text|>\n[IMG][/INST]',
    'mono_internvl_2b': '<|im_start|>system\nsys_prompt<|im_end|>\n<|im_start|>user\n<|text|><|im_end|>\n<|im_start|>assistant\n',
    'llava_onevision_72b': '<|im_start|>user <|text|><|im_end|><|im_start|>assistant\n'
}


# For caption + q evalutation

caption_dir = ''
