import pandas as pd
import os

project_path = '/home/wenting/PycharmProjects/thesis/'
output_path = project_path + 'data/prepared_data/'

categories = ['petitioner_lose', 'respondent_win', 'petitioner_win', 'respondent_lose']
for item in categories:
    os.makedirs(output_path + '%s' % item, exist_ok=True)


case_centered_dataset = pd.read_csv(project_path + 'data/target_data/case_centered_dataset.csv')

for index, row in case_centered_dataset.iterrows():
    filename = row['docket']
    if row['partyWinning'] == 0:
        with open(output_path + 'petitioner_lose/%s' % filename, 'w') as f:
            f.write('{}'.format(row['petitioner']))
        with open(output_path + 'respondent_win/%s' % filename, 'w') as f:
            f.write('{}'.format(row['respondent']))
    elif row['partyWinning'] == 1:
        with open(output_path + 'petitioner_win/%s' % filename, 'w') as f:
            f.write('{}'.format(row['petitioner']))
        with open(output_path + 'respondent_lose/%s' % filename, 'w') as f:
            f.write('{}'.format(row['respondent']))
