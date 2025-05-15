import os
import json
import pandas as pd
import argparse
from PIL import Image
import base64

category_map = {
    'Math': '数学',
    'Biology': '生物',
    'Chemistry': '化学',
    'Physics': '物理',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/datasets_vlm/jianai/Benchmark/csvqa_benchmark/data', help='path to saved tsv files')
    parser.add_argument('--image_dir', type=str, default='', help='path to saved image files')
    parser.add_argument('--is_direct', type=int, default=0)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        exit('data_dir {} does not exist'.format(args.data_dir))

    # single file includes both Chinese version question and English version questio
    file = os.path.join(args.data_dir, 'csvqa_data_new.tsv')

    if args.image_dir == '':
        args.image_dir = os.path.join(args.data_dir, 'images')
        os.makedirs(args.image_dir, exist_ok=True)

    # the jsonl files to save evaluation benchmark
    if args.is_direct == 0:
        output_file_zh = file.replace('.tsv', '_zh.jsonl')
        output_file_en = file.replace('.tsv', '_en.jsonl')
    else:
        output_file_zh = file.replace('.tsv', '_zh_direct.jsonl')
        output_file_en = file.replace('.tsv', '_en_direct.jsonl')

    # If the file already exists, clear it
    for output_file in [output_file_zh, output_file_en]:
        if os.path.exists(output_file):
            with open(output_file, 'w') as f:
                pass

    df = pd.read_csv(file, sep='\t')

    # Process Chinese version questions and create Chinese JSONL file
    with open(output_file_zh, 'a', encoding='utf-8') as writer_zh:
        for idx, info in df.iterrows():
            tmp_json = {}
            tmp_json['id'] = info['id']
            question = info['zh_question'].replace('<image>', '')
            if info['question_type'].lower() == 'open':
                question = question
            else:
                question = question + '\nA. ' + str(info['zh_A']) + '\nB. ' + str(info['zh_B']) + '\nC. ' + str(info['zh_C']) + '\nD. ' + str(info['zh_D'])
            
            category = category_map[info['category']]
            if args.is_direct == 0:
                if info['question_type'].lower() == 'open':
                    answer_text = '\\boxed{}'
                    question = (
                        f'以下是中国{category}高中习题：{question}。请依据题目的要求和所提供的信息计算得出答案。'
                        f'请使用一个单词或者短语回答这个问题。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。'
                        f'请在最后严格以“所以最终答案是：{answer_text}”输出结果。'
                    )
                else:
                    answer_text = '\\boxed{一个或多个选项字母，用英文逗号连接}'
                    question = (
                        f'以下是中国{category}高中习题：{question}。请依据题目的要求和所提供的信息计算得出答案。'
                        f'该题可能有一个或多个正确选项。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。'
                        f'请在最后严格以“所以最终答案是：{answer_text}”输出结果。'
                    )
            else:
                if info['question_type'].lower() == 'open':
                    answer_text = '\\boxed{}'
                    question = f'以下是中国{category}高中习题：{question}。请勿进行任何解释或多余回答，直接使用一个单词或者短语输出答案，仅以以下格式作答：所以最终答案是：{answer_text}'
                else:
                    answer_text = '\\boxed{一个或多个选项字母，用英文逗号连接}'
                    question = f'以下是中国{category}高中习题：{question}。请勿进行任何解释或多余回答，直接输出答案，仅以以下格式作答：所以最终答案是：{answer_text}'
            conversations = [
                {
                    'from': 'human',
                    'value': '<image>\n' + question,
                },
                {
                    'from': 'gpt',
                    'value': ''
                }
            ]
            tmp_json['conversations'] = conversations
            tmp_json['image'] = os.path.join(args.image_dir, f'{tmp_json["id"]}.jpg')
            tmp_json['answer'] = info['answer']
            tmp_json['question_type'] = info['question_type']
            tmp_json['category'] = info['category']
            tmp_json['image_type'] = info['image_type']
            tmp_json['difficulty'] = info['difficulty']
            
            # save base64 image
            image = eval(info['image'])[0]
            image = base64.b64decode(image.encode('utf-8'))
            if not os.path.exists(tmp_json['image']):
                with open(tmp_json['image'], 'wb') as f:
                    f.write(image)
                # check: some images may have the alpha channel, so we need to remove it; 
                # or you may encounter an error when encoding the multi-modality message.
                with Image.open(tmp_json['image']) as img:
                    img = img.convert('RGB')
                    img.save(tmp_json['image'])
            writer_zh.write(json.dumps(tmp_json, ensure_ascii=False) + '\n')

    # 处理英文问题并创建英文JSONL文件
    with open(output_file_en, 'a', encoding='utf-8') as writer_en:
        for idx, info in df.iterrows():
            tmp_json = {}
            tmp_json['id'] = info['id']
            question = info['en_question'].replace('<image>', '')
            if info['question_type'].lower() == 'open':
                question = question
            else:
                question = question + '\nA. ' + str(info['en_A']) + '\nB. ' + str(info['en_B']) + '\nC. ' + str(info['en_C']) + '\nD. ' + str(info['en_D'])
            
            
            if args.is_direct == 0:
                if info['question_type'].lower() == 'open':
                    answer_text = '\\boxed{}'
                    question = (
                        f'Below are Chinese {info["category"]} high school exercises: {question}. '
                        f'Answer this question with a single word or phrase according to the given requirements and the information provided. '
                        f'Use LaTeX format to represent variables and formulas in your solution. '
                        f'Please strictly end your response with "So the final answer is {answer_text}," and state the result explicitly.'
                    )
                else:
                    answer_text = '\\boxed{one or more option letters connected with commas}'
                    question = (
                        f'Below are Chinese {info["category"]} high school exercises: {question}. '
                        f'This question may have one or more correct answers. Please calculate the answer according to the given requirements and the information provided. '
                        f'Use LaTeX format to represent variables and formulas in your solution. '
                        f'Please strictly end your response with "So the final answer is {answer_text}," and state the result explicitly.'
                    )
            else:
                if info['question_type'].lower() == 'open':
                    answer_text = '\\boxed{}'
                    question = f'Below are Chinese {info["category"]} high school exercises: {question}. Please output only the final answer with a single word or phrase, without any explanation, reasoning, or additional text. Your response must be exactly:So the final answer is {answer_text}"'
                else:
                    answer_text = '\\boxed{one or more option letters connected with commas}'
                    question = f'Below are Chinese {info["category"]} high school exercises: {question}. Please output only the final answer, without any explanation, reasoning, or additional text. Your response must be exactly:So the final answer is {answer_text}"'

            conversations = [
                {
                    'from': 'human',
                    'value': '<image>\n' + question,
                },
                {
                    'from': 'gpt',
                    'value': ''
                }
            ]
            tmp_json['conversations'] = conversations
            tmp_json['image'] = os.path.join(args.image_dir, f'{tmp_json["id"]}.jpg')
            tmp_json['answer'] = info['answer']
            tmp_json['question_type'] = info['question_type']
            tmp_json['category'] = info['category']
            tmp_json['image_type'] = info['image_type']
            tmp_json['difficulty'] = info['difficulty']
            
            # save base64 image
            image = eval(info['image'])[0]
            image = base64.b64decode(image.encode('utf-8'))
            if not os.path.exists(tmp_json['image']):
                with open(tmp_json['image'], 'wb') as f:
                    f.write(image)
                # check: some images may have the alpha channel, so we need to remove it; 
                # or you may encounter an error when encoding the multi-modality message.
                with Image.open(tmp_json['image']) as img:
                    img = img.convert('RGB')
                    img.save(tmp_json['image'])
            writer_en.write(json.dumps(tmp_json, ensure_ascii=False) + '\n')
