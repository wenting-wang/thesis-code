import json
import re
import os
import pandas as pd


def get_text(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = json.load(f)

    text = re.sub(r'dict_values', '', str(text.values()))
    text = re.sub(r'[\[\]\'\"(),]', '', text)

    return text


docket_list = []
petitioner_list = []
respondent_list = []

project_path = '/home/wenting/PycharmProjects/thesis/'
for root, dirs, files in os.walk(project_path + 'data/cleaned_data'):
    for file in files:
        if file.endswith("-result-1.txt"):
            docket_list.append(file[:-13])
            input_file = os.path.join(root, file)
            pure_text = get_text(input_file)
            petitioner_list.append(pure_text)
        if file.endswith("-result-2.txt"):
            input_file = os.path.join(root, file)
            pure_text = get_text(input_file)
            respondent_list.append(pure_text)


case_centered_docket_result = pd.read_csv(project_path + 'data/target_data/case_centered_docket_result.csv')
text_df = pd.DataFrame({'docket': docket_list,
                        'petitioner': petitioner_list,
                        'respondent': respondent_list})
case_centered_dataset = pd.merge(case_centered_docket_result, text_df, on=['docket'], how='left')
case_centered_dataset.to_csv(project_path + 'data/target_data/case_centered_dataset.csv', index=False)
