import json
from textblob import TextBlob
import re
import os
import pandas as pd


def get_sentiment_score(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = json.load(f)

    text = re.sub(r'dict_values', '', str(text.values()))
    text = re.sub(r'[\[\]\'\"(),]', '', text)

    return TextBlob(text).sentiment


docket_list = []
petitioner_polarity_list = []
petitioner_subjectivity_list = []
respondent_polarity_list = []
respondent_subjectivity_list = []

project_path = '/home/wenting/PycharmProjects/thesis/'
for root, dirs, files in os.walk(project_path + 'data/cleaned_data'):
    for file in files:
        if file.endswith("-result-1.txt"):
            docket_list.append(file[:-13])
            input_file = os.path.join(root, file)
            polarity, subjectivity = get_sentiment_score(input_file)
            petitioner_polarity_list.append(polarity)
            petitioner_subjectivity_list.append(subjectivity)
        if file.endswith("-result-2.txt"):
            input_file = os.path.join(root, file)
            polarity, subjectivity = get_sentiment_score(input_file)
            respondent_polarity_list.append(polarity)
            respondent_subjectivity_list.append(subjectivity)

sentiment_df = pd.DataFrame({'docket': docket_list,
                             'petitioner_polarity': petitioner_polarity_list,
                             'petitioner_subjectivity': petitioner_subjectivity_list,
                             'respondent_polarity': respondent_polarity_list,
                             'respondent_subjectivity': respondent_subjectivity_list})

case_centered_docket_result = pd.read_csv(project_path + 'data/target_data/case_centered_docket_result.csv')
case_centered_sentiment = pd.merge(case_centered_docket_result, sentiment_df, on=['docket'], how='left')
case_centered_sentiment.to_csv(project_path + 'data/target_data/case_centered_sentiment.csv', index=False)

