import pandas as pd
import os

# docket number of cases in oral argument text
project_path = '/home/wenting/PycharmProjects/thesis/'
docket_list = []
for root, dirs, files in os.walk(project_path + 'data/oral_arguments_text/'):
    for file in files:
        docket_list.append(file[:-4])

# merge existing docket number with case results
# filter docket, only docket with oral argument text, left
case_centered_result = pd.read_csv(project_path + 'data/target_data/case_centered_docket.csv')
docket_df = pd.DataFrame({'docket': docket_list})
case_centered_docket_result = pd.merge(docket_df, case_centered_result, on=['docket'], how='left')

# with open(project_path + 'data/target_data/case_centered_docket_result', 'w') as f:
#     for item in docket_list:
#         f.write("%s\n" % item)

case_centered_docket_result.to_csv(project_path + 'data/target_data/case_centered_docket_result.csv', index=False)
# print(len(docket_list))  -> 3594 (up: 3539) (up: 3510)
# len(set(docket_list)) -> 3539 (up: 3539) (up: 3510)
# print(case_centered_docket_result.shape)  -> 3594 (up: 3723) (up: 3510)


# find cases with duplicated docket number, from oral_arguments_text. delete these cases
# import collections
# du_docket = [item for item, count in collections.Counter(docket_list).items() if count > 1]

# find cases with duplicated docket number, from case_centered_docket. delete these cases
# import collections
# a = case_centered_docket_result.iloc[:, 0]
# du_docket = [item for item, count in collections.Counter().items() if count > 1]
