'''
Parse the analysis results of each model and merge the evaluation metrics into the xlsx files.
The xlsx file content will be like:
model_name  overall multiple-choice open easy middle hard Math Physics Chemistry Biology *image_types
claude37 ....
InternVL2_38B ....
InternVL2_78B ....
InternVL3_38B ....
InternVl3_78B ....
.......
'''

import os
import json
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--model_name', type=str, default='internvl3_78b')
    parser.add_argument('--caption_model', type=str, default='internvl3_78b')

    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, str(args.max_new_tokens), args.model_name)
    if not os.path.exists(args.input_dir):
        exit('input dir not exist')

    q_types = os.listdir(args.input_dir)
    args.output_dir = os.path.join(args.output_dir, str(args.max_new_tokens), args.model_name)

    for q_type in q_types:
        if q_type == 'c_q':
            continue
        tmp_input_dir = os.path.join(args.input_dir, q_type)
        tmp_output_dir = os.path.join(args.output_dir, q_type)

        model_names = os.listdir(tmp_input_dir)
        for lang in ['zh', 'en']:
            tmp_lang_input_dir = os.path.join(tmp_input_dir, lang)
            tmp_lang_output_dir = os.path.join(tmp_output_dir, lang)
            os.makedirs(tmp_lang_output_dir, exist_ok=True)
            model_names = os.listdir(tmp_lang_input_dir)
            output_file = os.path.join(tmp_lang_output_dir, f'summary_{lang}.xlsx')
            model_name_to_metrics = []
            for model_name in model_names:
                model_name_to_metric = {}
                model_name_to_metric['model_name'] = model_name
                tmp_lang_model_dir = os.path.join(tmp_lang_input_dir, model_name)
                tmp_file = [f for f in os.listdir(tmp_lang_model_dir) if f.endswith('.json')][0]
                tmp_file = os.path.join(tmp_lang_model_dir, tmp_file)
                with open(tmp_file, 'r') as f:
                    model_metric = json.load(f)
                
                model_name_to_metric['overall'] = model_metric['overall']['acc']
                model_name_to_metric['multiple-choice'] = model_metric['question_type']['multiple-choice']['acc']
                model_name_to_metric['open'] = model_metric['question_type']['open']['acc']
                model_name_to_metric['easy'] = model_metric['difficulty']['easy']['acc']
                model_name_to_metric['middle'] = model_metric['difficulty']['middle']['acc']
                model_name_to_metric['hard'] = model_metric['difficulty']['hard']['acc']
                model_name_to_metric['Math'] = model_metric['subject']['Math']['acc']
                model_name_to_metric['Physics'] = model_metric['subject']['Physics']['acc']
                model_name_to_metric['Chemistry'] = model_metric['subject']['Chemistry']['acc']
                model_name_to_metric['Biology'] = model_metric['subject']['Biology']['acc']

                for image_type in model_metric['image_type']:
                    model_name_to_metric[image_type] = model_metric['image_type'][image_type]['acc']
                model_name_to_metrics.append(model_name_to_metric)
            df = pd.DataFrame(model_name_to_metrics)
            df.to_excel(output_file, index=False, engine='openpyxl')
    
    for q_type in q_types:
        if q_type != 'c_q':
            continue
        caption_models = os.listdir(os.path.join(args.input_dir, q_type))
        for caption_model in caption_models:
            tmp_caption_input_dir = os.path.join(args.input_dir, q_type, caption_model)
            caption_lengths = os.listdir(tmp_caption_input_dir)
            for caption_length in caption_lengths:
                tmp_caption_length_input_dir = os.path.join(tmp_caption_input_dir, caption_length)
                tmp_caption_length_output_dir = os.path.join(args.output_dir, q_type, caption_model, caption_length)

                for lang in ['zh', 'en']:
                    tmp_lang_input_dir = os.path.join(tmp_caption_length_input_dir, lang)
                    tmp_lang_output_dir = os.path.join(tmp_caption_length_output_dir, lang)
                    os.makedirs(tmp_lang_output_dir, exist_ok=True)
                    model_names = os.listdir(tmp_lang_input_dir)
                    output_file = os.path.join(tmp_lang_output_dir, f'summary_{lang}.xlsx')
                    model_name_to_metrics = []
                    for model_name in model_names:
                        model_name_to_metric = {}
                        model_name_to_metric['model_name'] = model_name
                        tmp_lang_model_dir = os.path.join(tmp_lang_input_dir, model_name)
                        tmp_file = [f for f in os.listdir(tmp_lang_model_dir) if f.endswith('.json')][0]
                        tmp_file = os.path.join(tmp_lang_model_dir, tmp_file)
                        with open(tmp_file, 'r') as f:
                            model_metric = json.load(f)
                        
                        model_name_to_metric['overall'] = model_metric['overall']['acc']
                        model_name_to_metric['multiple-choice'] = model_metric['question_type']['multiple-choice']['acc']
                        model_name_to_metric['open'] = model_metric['question_type']['open']['acc']
                        model_name_to_metric['easy'] = model_metric['difficulty']['easy']['acc']
                        model_name_to_metric['middle'] = model_metric['difficulty']['middle']['acc']
                        model_name_to_metric['hard'] = model_metric['difficulty']['hard']['acc']
                        model_name_to_metric['Math'] = model_metric['subject']['Math']['acc']
                        model_name_to_metric['Physics'] = model_metric['subject']['Physics']['acc']
                        model_name_to_metric['Chemistry'] = model_metric['subject']['Chemistry']['acc']
                        model_name_to_metric['Biology'] = model_metric['subject']['Biology']['acc']

                        for image_type in model_metric['image_type']:
                            model_name_to_metric[image_type] = model_metric['image_type'][image_type]['acc']
                        model_name_to_metrics.append(model_name_to_metric)
                    df = pd.DataFrame(model_name_to_metrics)
                    df.to_excel(output_file, index=False, engine='openpyxl')
            

