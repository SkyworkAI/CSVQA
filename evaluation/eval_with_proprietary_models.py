'''
This file is used to evaluation the metric on proprietary models.
'''


import json
import base64
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import openai
import copy
import os
import argparse
from config import caption_dir, input_files_direct, input_files


client = openai.OpenAI(
    base_url="",
    api_key="",  # add your onw api-key here
)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")
    except FileNotFoundError:
        print("Error: File is not found")
    except Exception as e:
        print(f"Error: An unexpectd error occurs {e}")
    return None


def get_rs(data, model_name, q_type, is_en, temperature, top_p, max_new_tokens):
    try:
        q = data["conversations"][0]["value"]
        image_path = data["image"]
        image = image_to_base64(image_path)
        if not image:
            return None
        if q_type == 0:
            if model_name == 'o1':
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{image}'
                                }
                            },
                            {
                                'type': 'text',
                                'text': q.replace('<image>', '').strip()
                            }
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image,
                                },
                            },
                            {
                                "type": "text",
                                "text": q.replace("<image>", "").strip(),
                            },
                        ],
                    }
                ]
        elif q_type == 1:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": q.replace("<image>", "").strip(),
                        },
                    ],
                }
            ]
        else:
            caption = data['caption']
            if is_en == 0:
                caption_prompt = '下面是图像的自然描述：<|caption|>, 请基于描述解决下面的问题。'
            else:
                caption_prompt = 'Here is the natural description of the figure: <|caption|>, please solve the following problem based on the description.'

            caption_prompt = caption_prompt.replace('<|caption|>', caption)
            q = caption_prompt + q
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": q.replace("<image>", "").strip(),
                        },
                    ],
                }
            ]
        if model_name == 'o1':
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_new_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        if model_name == 'o1':
            answer = response.choices[0].message.content
        else:
            a = []
            for chunk in response:
                if not chunk.choices:
                    break
                if len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0].delta, "content"):
                        word = chunk.choices[0].delta.content
                        if word:
                            a.append(word)
            answer = "".join(a)
        if answer:
            data["conversations"][1]["value"] = answer
            return data
        else:
            return None
    except Exception as e:
        print(e)
        return None


def process_requests(datas, model_name, q_type, is_en, temperature, top_p, max_new_tokens):
    with ThreadPoolExecutor(max_workers=20) as executor:  # 可根据需求调整线程数
        futures = [executor.submit(get_rs, data, model_name, q_type, is_en, temperature, top_p, max_new_tokens) for data in datas]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"任务失败: {e}")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_name = "claude37-sonnet"
    # model_name = 'gpt-4o-mini'
    # model_name = 'gemini-2.0-flash'
    # model_name = 'o1'
    parser.add_argument('--model_name', type=str, default='o1')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument("--is_en", type=int, default=0, choices=[0, 1])
    parser.add_argument("--q_type", type=int, default=0, choices=[0, 1, 2], help='0 for v+q, 1 for q, 2 for caption + q')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=9999999)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--output_key", type=str, default="conversations")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument('--caption_model', type=str, default='qwen2_5_vl_72b_instruct')
    parser.add_argument('--caption_length', type=int, default=4096, choices=[1024, 2048, 4096, 8192])
    parser.add_argument('--is_direct', type=int, default=0, choices=[0, 1], help='0 for normally inference, 1 for directly inference')
    args = parser.parse_args()

    args.caption_file = ''
    caption = {}
    if args.is_direct == 1:
        args.output_dir = args.output_dir + '_direct'
    if args.q_type == 0:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'v_q')
    elif args.q_type == 1:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'q')
    else:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'c_q', args.caption_model, str(args.caption_length))
        args.caption_file = os.path.join(caption_dir, args.caption_model, str(args.caption_length), 'captions.json')
        with open(args.caption_file, 'r') as f:
            caption = json.load(f)

    if args.is_en == 1:
        args.output_dir = os.path.join(args.output_dir, 'en', args.model_name.replace('-', '_').replace('.', '_'))
    else:
        args.output_dir = os.path.join(args.output_dir, 'zh', args.model_name.replace('-', '_').replace('.', '_'))
    
    total_data = []
    batch_size = 10

    os.makedirs(args.output_dir, exist_ok=True)
    if args.is_direct == 0:
        args.input_file = input_files[args.is_en]
    else:
        args.input_file = input_files_direct[args.is_en]

    file_name = args.input_file.split("/")[-1]
    args.output_file = os.path.join(args.output_dir, file_name.replace('.jsonl', '_eval.jsonl'))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "a") as writer:
        with open(args.input_file, "r") as reader:
            lines = reader.readlines()
            from tqdm import tqdm
            for line in tqdm(lines):
                json_data = json.loads(line)
                if args.q_type == 2:
                    if args.is_en == 0:
                        json_data['caption'] = caption[json_data['id']]['zh'][0]
                    else:
                        json_data['caption'] = caption[json_data['id']]['en'][0]
                total_data.append(json_data)
                if len(total_data) == batch_size:
                    results = process_requests(total_data, args.model_name, args.q_type, args.is_en, args.temperature, args.top_p, args.max_new_tokens)
                    for idx, result in enumerate(results, start=1):
                        if result:
                            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                            writer.flush()
                    total_data = []
            if total_data:
                results = process_requests(total_data, args.model_name, args.q_type, args.is_en, args.temperature, args.top_p, args.max_new_tokens)
                for idx, result in enumerate(results, start=1):
                    if result:
                        writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                        writer.flush()
