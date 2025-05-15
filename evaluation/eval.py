'''
This code is used to eval the performance of open-source mllm models on the csvqa benchmark.
The model response will be added in the `gpt value` in the `conversations` of the origin benchmark json data.
converations = [
    {
        'from': 'human',
        'value': The question,
    },
    {
        'from': 'gpt',
        'value': The model response,
    }
]
'''


import json
import argparse
import os

from tqdm import tqdm


from vlm_model import VlmModel
from utils import calculate_md5

from config import model_name_to_path, input_files, input_files_direct, system_prompts, model_name_to_template, model_name_to_hf_override, caption_dir

model_names = list(model_name_to_path.keys())


class Gen:
    def __init__(self, model, input_file, output_file, min_step, max_step, batch_size):
        self.model = model
        self.output_file = output_file
        self.load_json_datas(input_file)
        self.batch_size = batch_size
        self.json_datas = self.json_datas[min_step:max_step]
        print("cur json_datas len is: ", len(self.json_datas))
        self.fw = open(output_file, "a", encoding="utf8")

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

    def process_response(self, json_datas, reponses):
        res_index = 0
        for json_data in json_datas:
            try:
                new_conversations = []
                conversations = json_data["conversations"]
                for item in conversations:
                    if item["from"] == "human":
                        new_conversations.append({
                            "from": "human",
                            "value": item["value"],
                        })
                    else:
                        new_conversations.append({
                            "from": "gpt",
                            "value": reponses[res_index][0].strip(),
                        })
                        res_index += 1
                json_data[args.output_key] = new_conversations
                self.fw.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                self.fw.flush()
            except Exception as e:
                print(e)
                continue

    def inner_loop(self):
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
                    conversations = j["conversations"]
                    cur_image = j["image"]
                    for item in conversations:
                        if item["from"] == "human":
                            cur_queries.append((j['id'], item["value"], cur_image))
                    cur_new_json_datas.append(j)
                responses = self.model(json_datas=cur_queries)
                self.process_response(cur_new_json_datas, responses)
            except Exception as e:
                print(e)
                cur_queries = []
                continue
        self.fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=model_names)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument("--is_en", type=int, required=True, choices=[0, 1])
    parser.add_argument("--q_type", type=int, required=True, choices=[0, 1, 2], help='0 for v+q, 1 for q, 2 for caption + q')
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
    parser.add_argument('--is_direct', type=int, default=0, choices=[0, 1], help='0 for normally inference, 1 for directly infernece')
    args = parser.parse_args()

    if args.is_direct == 1:
        args.output_dir = args.output_dir + '_direct'

    args.caption_file = ''
    if args.q_type == 0:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'v_q')
    elif args.q_type == 1:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'q')
    else:
        args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), 'c_q', args.caption_model, str(args.caption_length))
        args.caption_file = os.path.join(caption_dir, args.caption_model, str(args.caption_length), 'captions.json')

    if args.is_en == 1:
        args.output_dir = os.path.join(args.output_dir, 'en', args.model_name)
    else:
        args.output_dir = os.path.join(args.output_dir, 'zh', args.model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    args.model_path = model_name_to_path[args.model_name]
    if args.is_direct == 1:
        args.input_file = input_files_direct[args.is_en]
    else:
        args.input_file = input_files[args.is_en]
    
    file_name = args.input_file.split("/")[-1]
    args.output_file = os.path.join(args.output_dir, file_name.replace('.jsonl', '_eval.jsonl'))
    if os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            pass

    sys_prompt = system_prompts[args.model_name]
    template = model_name_to_template[args.model_name]
    hf_override = model_name_to_hf_override.get(args.model_name, '')
    if args.model_name == 'glm_4v_9b' and (args.q_type == 1 or args.q_type == 2):
        hf_override = 'GLM4VForCausalLM'
    model = VlmModel(
        args.model_path,
        args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        n=args.num_sample,
        sys_prompt=sys_prompt,
        template=template,
        hf_override=hf_override,
        q_type=args.q_type,
        is_en=args.is_en,
        caption_file=args.caption_file
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
