'''
This code is used to load the model with vllm tool and introduce the batch inference.
'''


from vllm import LLM, SamplingParams
import torch
from PIL import Image
import base64
from io import BytesIO
import os
import json


class VlmModel:
    def __init__(
        self, model_path, model_name, temperature=1.0, top_p=1, max_new_tokens=8192, n=1, sys_prompt=None, template=None, hf_override='', q_type=0, is_en=0, caption_file=None
    ) -> None:
        self.n = n
        self.model_name = model_name
        if not sys_prompt:
            self.sys_prompt = "You are a helpfull assistant."
        else:
            self.sys_prompt = sys_prompt
        self.q_type = q_type
        self.is_en = is_en
        self.template = template
        
        # load caption info
        if caption_file:
            if os.path.exists(caption_file):
                with open(caption_file, 'r') as f:
                    self.captions = json.load(f)
        else:
            self.captions = {}
        
        self.gen_params = SamplingParams(
            max_tokens=max_new_tokens,
            n=self.n,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.0,

        )
        world_size = torch.cuda.device_count()
        if model_name in ['qwen2_vl_7b_instruct', 'qwen2_5_vl_7b_instruct']:
            world_size = 4
        if model_name == 'deepseek_vl2_tiny':
            world_size = 1
        if hf_override == '':
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=world_size,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1}
            )
        else:
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=world_size,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                hf_overrides={"architectures": [hf_override]},
                limit_mm_per_prompt={"image": 1}
            )

    def __call__(self, json_datas):
        prompts = []
        for info_id, text, image in json_datas:
            # idefics_8b does not need the <image> placeholder when decoding the multi-modality message.
            if self.model_name == 'idefics_8b':
                text = text.replace('<image>\n', '<image>')
            # <image> is only used when the question type is visual + question.
            if self.q_type == 1 or self.q_type == 2:
                text = text.replace('<image>', '')
            # visual + question evaluation
            if self.q_type == 0:
                message = self.template.replace('sys_prompt', self.sys_prompt).replace('<|text|>', text)
                if not image:
                    prompts.append({
                        'prompt': message
                    })
                elif type(image) == str:
                    prompts.append({
                        "prompt": message,
                        "multi_modal_data": {"image": Image.open(image)},
                    })
                elif type(image) == list:
                    image_objects = [Image.open(img_path) for img_path in image]
                    prompts.append({
                        "prompt": message,
                        "multi_modal_data": {"image": image_objects},
                    })
            elif self.q_type == 1:
                message = self.template.replace('sys_prompt', self.sys_prompt).replace('<|text|>', text)
                prompts.append({
                    "prompt": message,
                })
            else:
                if self.is_en == 0:
                    caption = self.captions[info_id]['zh'][0]
                    caption_prompt = '下面是图像的自然描述：<|caption|>, 请基于描述解决下面的问题。'
                else:
                    caption = self.captions[info_id]['en'][0]
                    caption_prompt = 'Here is the natural description of the figure: <|caption|>, please solve the following problem based on the description.'
                caption_prompt = caption_prompt.replace('<|caption|>', caption)
                text = caption_prompt + text
                message = self.template.replace('sys_prompt', self.sys_prompt).replace('<|text|>', text)
                prompts.append({
                    "prompt": message,
                })
        outputs = self.model.generate(prompts, self.gen_params)
        model_outs = []
        for output in outputs:
            cur_outs = []
            for i in range(self.n):
                cur_outs.append(output.outputs[i].text)
            model_outs.append(cur_outs)
        return model_outs