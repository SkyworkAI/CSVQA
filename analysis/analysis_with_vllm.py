'''
This code is used to analysis the model responses. (VLLM version)
If the question is multiple-choice type, the rule-based method will be used. 
    The method is the detect the content in the \\boxed{}. If you set the `mum`, the local llm/mllm model will be
    used to check whether the model response is equal to answer when \\boxed{} can not be detected.
If the question is open type, the model-based method will be used.
    The method will use the local llm/mllm model or gpt to judge whether is model response is equal to answer.

After you run this code, you will get two files in each model subfolder.
    1. hit.jsonl: Add the 'hit` key into each evaluation data. hit=1 is correct while hit=0 is wrong.
    2. analysis.json: This is metric info of each model on specific question type.

If you want to summary the analysis results of each model on each question type, you can run the 'summary.py' code to get the merged evaluation metrics.
'''


import json
import os
import argparse
import re
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import math
from tqdm import tqdm
from config import model_name_to_path
from vllm import LLM, SamplingParams


def is_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def is_english(text):
    return bool(re.match(r"^[A-Za-z0-9\s]*$", text))

def detect_language(text):
    if is_chinese(text):
        return "Chinese"
    elif is_english(text):
        return "English"
    else:
        return "Unknown"

def extract_answer_content(text):
    if "```markdown" in text:
        text = text.split("```markdown")[1].split("```")[0]
        return text
    elif "```json" in text:
        text = text.split("```json")[1].split("```")[0]
        return text
    else:
        return text


# check prompt for Chinese response
CN_PE = """我将给你一个两个文本，你需要判断两个文本给出的最终答案或者结论是否一致，不一致返回0，一致返回1
文本1:
[q1]

文本2:
[q2]

请以json方式返回，格式如下，不允许添加额外的输出:
```json
{"check": 1}
```
"""

# check prompt for English response
EN_PE = """I will give you two texts. You need to judge whether the final answers or conclusions given by the two texts are consistent. If they are inconsistent, return 0, and if they are consistent, return 1.
Text 1:
[q1]

Text 2:
[q2]

Please return in json format. The format is as follows. No additional output is allowed:
```json
{"check": 1}
```
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='internvl3_78b')
    parser.add_argument('--q_type', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--caption_model', type=str, default='qwen2_5_vl_72b_instruct')
    parser.add_argument('--caption_length', type=int, default=4096)
    parser.add_argument('--mum', action='store_true')
    args = parser.parse_args()
    if args.mum:
        args.output_dir = args.output_dir + '_mum'
    args.model_path = model_name_to_path[args.model_name]
    # 0 for visual + question; 1 for pure question; 2 for caption + qeustion
    args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens))
    args.input_dir = os.path.join(args.input_dir, str(args.max_new_tokens))
    if args.q_type == 0:
        args.output_dir = os.path.join(args.output_dir, args.model_name, 'v_q')
        args.input_dir = os.path.join(args.input_dir, 'v_q')
    elif args.q_type == 1:
        args.output_dir = os.path.join(args.output_dir, args.model_name, 'q')
        args.input_dir = os.path.join(args.input_dir, 'q')
    else:
        args.output_dir = os.path.join(args.output_dir, args.model_name, 'c_q', args.caption_model, str(args.caption_length))
        args.input_dir = os.path.join(args.input_dir, 'c_q', args.caption_model, str(args.caption_length))

    os.makedirs(args.output_dir, exist_ok=True)
    # judge model: to check whether the open question is right, or you can use the gpt
    world_size = torch.cuda.device_count()
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=world_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1}
    )
    gen_params = SamplingParams(
        max_tokens=16000,
        n=1,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,

    )
    for lang in ['zh', 'en']:
        lang_input_dir = os.path.join(args.input_dir, lang)
        if not os.path.exists(lang_input_dir):
            continue
        lang_output_dir = os.path.join(args.output_dir, lang)
        os.makedirs(lang_output_dir, exist_ok=True)

        model_names = os.listdir(lang_input_dir)
        for model_name in model_names:
            analysis_info = {}

            total_questions = 0
            total_correct = 0

            difficulty = {
                'easy': {'total': 0, 'correct': 0, 'subject': {}, 'image_type': {}},
                'medium': {'total': 0, 'correct': 0, 'subject': {}, 'image_type': {}},
                'hard': {'total': 0, 'correct': 0, 'subject': {}, 'image_type': {}},
            }
            subject = {
                'Math': {'total': 0, 'correct': 0, 'image_type': {}, 'difficulty': {}},
                'Physics': {'total': 0, 'correct': 0, 'image_type': {}, 'difficulty': {}},
                'Chemistry': {'total': 0, 'correct': 0, 'image_type': {}, 'difficulty': {}},
                'Biology': {'total': 0, 'correct': 0, 'image_type': {}, 'difficulty': {}},
            }
            image_type = {}
            question_type = {
                'open': {'total': 0, 'correct': 0},
                'multiple-choice': {'total': 0, 'correct': 0},
            }

            lang_model_input_dir = os.path.join(lang_input_dir, model_name)
            lang_model_output_dir = os.path.join(lang_output_dir, model_name)

            try:
                file = os.listdir(lang_model_input_dir)[0]
                os.makedirs(lang_model_output_dir, exist_ok=True)
            except:
                continue

            input_file = os.path.join(lang_model_input_dir, file)
            output_jsonl_file = os.path.join(lang_model_output_dir, 'hint.jsonl')
            output_json_file = os.path.join(lang_model_output_dir, 'analysis.json')
            if os.path.exists(output_json_file):
                continue

            with open(input_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                if not lines:
                    continue
                with open(output_jsonl_file, 'a', encoding='utf-8') as writer:
                    for line in tqdm(lines):
                        info = json.loads(line)
                        hit = 0
                        if info['question_type'] == 'multiple-choice':
                            # judge the multiple-choice question response with rule.
                            total_questions += 1
                            question_type['multiple-choice']['total'] += 1
                            model_response = info['conversations'][1]['value']
                            pattern = r'\\boxed\{([^}]*)\}'
                            try:
                                match = re.findall(pattern, model_response)[-1]
                                model_answer = ''
                                for letter in ['A', 'B', 'C', 'D']:
                                    if letter in match:
                                        model_answer += letter
                                info['model_answer'] = model_answer
                                answer = info['answer']
                                if model_answer == answer:
                                    hit = 1
                                    question_type['multiple-choice']['correct'] += 1
                                info['hit'] = hit
                            except:
                                if args.mum:
                                    answer = info['answer']
                                    lang = detect_language(model_response)
                                    if lang == 'Chinese':
                                        pe = CN_PE
                                    else:
                                        pe = EN_PE
                                    pe = pe.replace("[q1]", model_response).replace("[q2]", answer)
                                    pe = [{'prompt': pe}]
                                    response = model.generate(pe, gen_params)
                                    extract_answer = extract_answer_content(response[0].outputs[0].text)
                                    try:
                                        hit = int(json.loads(extract_answer)["check"])
                                        if hit == 1:
                                            question_type['multiple-choice']['correct'] += 1
                                        else:
                                            hit = 0
                                    except:
                                        hit = 0
                                    info['hit'] = hit
                                else:
                                    info['model_answer'] = ''
                                    hit = 0
                                    info['hit'] = hit
                        else:
                            # use mllm model or llm to judge whether the open question response is right.
                            question_type['open']['total'] += 1
                            total_questions += 1
                            model_response = info['conversations'][1]['value']
                            answer = info['answer']
                            lang = detect_language(model_response)
                            if lang == 'Chinese':
                                pe = CN_PE
                            else:
                                pe = EN_PE
                            pe = pe.replace("[q1]", model_response).replace("[q2]", answer)

                            pe = [{'prompt': pe}]
                            response = model.generate(pe, gen_params)
                            extract_answer = extract_answer_content(response[0].outputs[0].text)
                            try:
                                hit = int(json.loads(extract_answer)["check"])
                                if hit == 1:
                                    question_type['open']['correct'] += 1
                                else:
                                    hit = 0
                            except:
                                hit = 0
                            info['hit'] = hit

                        if hit == 1:
                            total_correct += 1
                        if ',' in info['image_type']:
                            img_types = info['image_type'].split(',')
                        else:
                            if info['image_type'] == 'Diagrams':
                                img_types = ['Illustration']
                            else:
                                img_types = [info['image_type']]
                        
                        # process image_type
                        for img_type in img_types:
                            # difficulty
                            if img_type not in difficulty[info['difficulty']]['image_type']:
                                difficulty[info['difficulty']]['image_type'][img_type] = {'total': 1, 'correct': 0}
                            else:
                                difficulty[info['difficulty']]['image_type'][img_type]['total'] += 1
                            
                            # subject
                            if img_type not in subject[info['category']]['image_type']:
                                subject[info['category']]['image_type'][img_type] = {'total': 1, 'correct': 0}
                            else:
                                subject[info['category']]['image_type'][img_type]['total'] += 1

                            # image_type
                            if img_type not in image_type:
                                image_type[img_type] = {'total': 1, 'correct': 0, 'subject': {}, 'difficulty': {}}
                                subject[info['category']]['image_type'][img_type] = {'total': 1, 'correct': 0}
                            else:
                                image_type[img_type]['total'] += 1
                        
                        # process difficulty
                        difficulty[info['difficulty']]['total'] += 1
                        if info['difficulty'] not in subject[info['category']]['difficulty']:
                            subject[info['category']]['difficulty'][info['difficulty']] = {'total': 1, 'correct': 0}
                        else:
                            subject[info['category']]['difficulty'][info['difficulty']]['total'] += 1
                        
                        for img_type in img_types:
                            if info['difficulty'] not in image_type[img_type]['difficulty']:
                                image_type[img_type]['difficulty'][info['difficulty']] = {'total': 1, 'correct': 0}
                            else:
                                image_type[img_type]['difficulty'][info['difficulty']]['total'] += 1
                        
                        # process subject

                        subject[info['category']]['total'] += 1
                        if info['category'] not in difficulty[info['difficulty']]['subject']:
                            difficulty[info['difficulty']]['subject'][info['category']] = {'total': 1, 'correct': 0}
                        else:
                            difficulty[info['difficulty']]['subject'][info['category']]['total'] += 1
                        
                        for img_type in img_types:
                            if info['category'] not in image_type[img_type]['subject']:
                                image_type[img_type]['subject'][info['category']] = {'total': 1, 'correct': 0}
                            else:
                                image_type[img_type]['subject'][info['category']]['total'] += 1
                        
                        if hit == 1:
                            difficulty[info['difficulty']]['correct'] += 1
                            difficulty[info['difficulty']]['subject'][info['category']]['correct'] += 1
                            subject[info['category']]['correct'] += 1
                            subject[info['category']]['difficulty'][info['difficulty']]['correct'] += 1
                            for img_type in img_types:
                                image_type[img_type]['correct'] += 1
                                subject[info['category']]['image_type'][img_type]['correct'] += 1
                                difficulty[info['difficulty']]['image_type'][img_type]['correct'] += 1
                                image_type[img_type]['subject'][info['category']]['correct'] += 1
                                image_type[img_type]['difficulty'][info['difficulty']]['correct'] += 1
                        writer.write(json.dumps(info, ensure_ascii=False) + '\n')
                    
                    # to save the analysis result
                    analysis_info['overall'] = {'acc': round(total_correct / total_questions, 4), 'total': total_questions, 'correct': total_correct}
                    analysis_info['question_type'] = {}
                    analysis_info['question_type']['multiple-choice'] = {'acc': round(question_type['multiple-choice']['correct'] / question_type['multiple-choice']['total'], 4), 'total': question_type['multiple-choice']['total'], 'correct': question_type['multiple-choice']['correct']}
                    analysis_info['question_type']['open'] = {'acc': round(question_type['open']['correct'] / question_type['open']['total'], 4), 'total': question_type['open']['total'], 'correct': question_type['open']['correct']}
                    analysis_info['difficulty'] = {}       
                    for key in difficulty:
                        analysis_info['difficulty'][key] = {'acc': round(difficulty[key]['correct'] / difficulty[key]['total'], 4), 'total': difficulty[key]['total'], 'correct': difficulty[key]['correct'], 'subject': {}, 'image_type': {}}
                        
                        for key1 in difficulty[key]['subject']:
                            analysis_info['difficulty'][key]['subject'][key1] = {'acc': round(difficulty[key]['subject'][key1]['correct'] / difficulty[key]['subject'][key1]['total'], 4), 'total': difficulty[key]['subject'][key1]['total'], 'correct': difficulty[key]['subject'][key1]['correct']}
                        
                        for key2 in difficulty[key]['image_type']:
                            analysis_info['difficulty'][key]['image_type'][key2] = {'acc': round(difficulty[key]['image_type'][key2]['correct'] / difficulty[key]['image_type'][key2]['total'], 4), 'total': difficulty[key]['image_type'][key2]['total'], 'correct': difficulty[key]['image_type'][key2]['correct']}
                    
                    analysis_info['subject'] = {}
                    for key in subject:
                        analysis_info['subject'][key] = {'acc': round(subject[key]['correct'] / subject[key]['total'], 4), 'total': subject[key]['total'], 'correct': subject[key]['correct'], 'difficulty': {}, 'image_type': {}}
                        for key1 in subject[key]['difficulty']:
                            analysis_info['subject'][key]['difficulty'][key1] = {'acc': round(subject[key]['difficulty'][key1]['correct'] / subject[key]['difficulty'][key1]['total'], 4), 'total': subject[key]['difficulty'][key1]['total'], 'correct': subject[key]['difficulty'][key1]['correct']}
                        for key2 in subject[key]['image_type']:
                            analysis_info['subject'][key]['image_type'][key2] = {'acc': round(subject[key]['image_type'][key2]['correct'] / subject[key]['image_type'][key2]['total'], 4), 'total': subject[key]['image_type'][key2]['total'], 'correct': subject[key]['image_type'][key2]['correct']}
                    
                    analysis_info['image_type'] = {}
                    for key in image_type:
                        analysis_info['image_type'][key] = {'acc': round(image_type[key]['correct'] / image_type[key]['total'], 4), 'total':image_type[key]['total'], 'correct': image_type[key]['correct'], 'subject': {}, 'difficulty': {}}
                        for key1 in image_type[key]['subject']:
                            analysis_info['image_type'][key]['subject'][key1] = {'acc': round(image_type[key]['subject'][key1]['correct'] / image_type[key]['subject'][key1]['total'], 4), 'total': image_type[key]['subject'][key1]['total'], 'correct': image_type[key]['subject'][key1]['correct']}
                        for key2 in image_type[key]['difficulty']:
                            analysis_info['image_type'][key]['difficulty'][key2] = {'acc': round(image_type[key]['difficulty'][key2]['correct'] / image_type[key]['difficulty'][key2]['total'], 4), 'total': image_type[key]['difficulty'][key2]['total'], 'correct': image_type[key]['difficulty'][key2]['correct']}
                
                    with open(output_json_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_info, f, indent=4)