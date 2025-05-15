'''
This code is used to generate the caption of each image with the local mllm models.
It use two prompts to generate the Chinese version caption and the English version caption seperately.
And you will get the caption json like this:
    'image_name': {
        'zh': [Chinese version caption],
        'en': [English version caption]
    }
'''
import json
import argparse
import os

from tqdm import tqdm


from vlm_model_for_captions import VlmModel
from utils import calculate_md5

from config import model_name_to_path, input_files, system_prompts

model_names = list(model_name_to_path.keys())


class Gen:
    def __init__(self, model, input_file, output_file, min_step, max_step, batch_size):
        self.model = model
        self.output_file = output_file
        self.load_json_datas(input_file)
        self.batch_size = batch_size
        self.json_datas = self.json_datas[min_step:max_step]
        print("cur json_datas len is: ", len(self.json_datas))

    def load_json_datas(self, input_file):
        self.json_datas = []
        md5set = set()
        if os.path.isfile(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        json_data = json.loads(line)
                        md5_value = calculate_md5(
                            json_data["conversations"][0]["value"]
                        )
                        md5set.add(md5_value)
                    except Exception:
                        continue
        print(f"Loaded {len(md5set)} samples.")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                md5_value = calculate_md5(json_data["conversations"][0]["value"])
                if md5_value not in md5set:
                    self.json_datas.append(json_data)

    def process_response(self, json_datas, responses):
        caption_json = {}
        for i, json_data in enumerate(json_datas):
            image_idx = json_data['id']
            if i % 2 == 0:
                caption_json[image_idx] = {
                    'zh': responses[i]
                }
            else:
                caption_json[image_idx]['en'] = responses[i]
        return caption_json
                

    def inner_loop(self):
        caption_jsons = {}
        for cur_json_datas in tqdm(
            [
                self.json_datas[i : i + self.batch_size]
                for i in range(0, len(self.json_datas), self.batch_size)
            ],
            total=len(self.json_datas) // self.batch_size,
            desc="BATCH",
        ):
            try:
                cur_queries = []
                cur_new_json_datas = []
                for j in cur_json_datas:
                    cur_image = j["image"]
                    zh_question = '<image>\n请使用中文描述这张图像。'
                    en_question = '<image>\nPlease describe this image in English.'
                    cur_queries.append((zh_question, cur_image))
                    cur_queries.append((en_question, cur_image))
                    cur_new_json_datas.append(j)
                    cur_new_json_datas.append(j)
                responses = self.model(json_datas=cur_queries)
                batch_captions = self.process_response(cur_new_json_datas, responses)
                caption_jsons.update(batch_captions)
            except Exception as e:
                print(e)
                cur_queries = []
                continue
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(caption_jsons, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=model_names)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=9999999)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    args = parser.parse_args()
    
    args.input_file = input_files[0]
    args.output_dir = os.path.join(os.path.dirname(args.input_file), 'captions', args.model_name, str(args.max_new_tokens))
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, 'captions.json')

    args.model_path = model_name_to_path[args.model_name]

    if os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            pass
    
    sys_prompt = system_prompts[args.model_name]
    model = VlmModel(
        args.model_path,
        args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        n=args.num_sample,
        sys_prompt=sys_prompt,
    )
    gen = Gen(
        model,
        args.input_file,
        args.output_file,
        args.min_step,
        args.max_step,
        args.batch_size,
    )
    gen.inner_loop()
