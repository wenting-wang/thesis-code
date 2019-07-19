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


###### prepare data for mix model
# import pandas as pd
#
# project_path = '/home/wenting/PycharmProjects/thesis/'
# output_path = project_path + 'data/mixed_data/'
#
# case_centered_dataset = pd.read_csv(project_path + 'data/target_data/case_centered_dataset.csv')
#
# for index, row in case_centered_dataset.iterrows():
#     filename = row['docket']
#     with open(output_path + "%s_petitioner" % filename, 'w') as f:
#         f.write('{}'.format(row['petitioner']))
#     with open(output_path + '%s_respondent' % filename, 'w') as f:
#         f.write('{}'.format(row['respondent']))


###### filter structured data
# import pandas as pd
#
# project_path = '/home/wenting/PycharmProjects/thesis/'
# output_path = project_path + 'data/mixed_data/'
#
# case_centered_dataset = pd.read_csv(project_path + 'data/target_data/case_centered_dataset.csv', encoding='utf-8')
# docket_list = case_centered_dataset['docket'].tolist()
#
# info = pd.read_csv('/home/wenting/PycharmProjects/thesis/data/caseinfo_org.csv', encoding='utf-8')
# info = info[info.docket.isin(docket_list)]
# info = info.sort_values(by=['docket'])
# info.to_csv('/home/wenting/PycharmProjects/thesis/data/mixed_data/caseinfo.csv', index=False)
