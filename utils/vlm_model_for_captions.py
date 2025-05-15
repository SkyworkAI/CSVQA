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
        self, model_path, model_name, temperature=1.0, top_p=1, max_new_tokens=8192, n=1, sys_prompt=None, template=None
    ) -> None:
        self.n = n
        self.model_name = model_name
        if not sys_prompt:
            self.sys_prompt = "You are a helpfull assistant."
        else:
            self.sys_prompt = sys_prompt
        self.template = template
        
        
        self.gen_params = SamplingParams(
            max_tokens=max_new_tokens,
            n=self.n,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.0,
        )
        world_size = torch.cuda.device_count()
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=world_size,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1}
        )

    def __call__(self, json_datas):
        prompts = []
        for text, image in json_datas:
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
        outputs = self.model.generate(prompts, self.gen_params)
        model_outs = []
        for output in outputs:
            cur_outs = []
            for i in range(self.n):
                cur_outs.append(output.outputs[i].text)
            model_outs.append(cur_outs)
        return model_outs
