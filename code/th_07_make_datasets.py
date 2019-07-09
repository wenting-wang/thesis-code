import json
import re
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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

# rewrite dataset (by partyWinning)
# 0	no favorable disposition for petitioning party apparent (respondent win)
# 1	petitioning party received a favorable disposition (petitioner win)
# 2	favorable disposition for petitioning party unclear (unclear) -> delete

# delete docket with missing disposition
# labels[labels.isna() == True]
# case_centered_docket_result.iloc[535,:]
# 535   NaN
# 591   NaN
# 599   NaN

# labels = case_centered_docket_result['partyWinning'].astype(int)
# # pd.Series(labels).value_counts()
# # 1    2173
# # 0    1333
# # 2       1 -> delete (00-878,9.0,2.0)
# labels = np.array(labels).reshape(len(labels), -1)
# enc = OneHotEncoder(categories='auto')
# enc.fit(labels)
# targets = enc.transform(labels).toarray()
#
# new_df = case_centered_dataset.assign(partyWinning_0=pd.Series(targets[:0]).values)
# new_df = new_df.assign(partyWinning_1=pd.Series(targets[:1]).values)
# new_df = new_df.assign(partyWinning_2=pd.Series(targets[:2]).values)
#
# new_df.to_csv(project_path + 'data/target_data/case_centered_dataset_2.csv', index=False)

# UPDATE: NO NEED FOR ONE HOT ENCODING
