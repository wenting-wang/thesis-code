import os
import re

"""
split by side, and write to folder 'split_data'
result-1: justices text, speaking to petitioner
result-2: justices text, speaking to respondent
"""


def split_by_side(input_path, output_path):

    # make file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'@', '\n@', text)
        text = re.sub(r'\*', '\n*', text)

    with open(output_path[:-4] + '-result.txt', 'w') as f:
        f.write(text)

    # split by side, only justices, no lawyers
    justices = ['UNKN', 'ALIT', 'BREY', 'CONN', 'GINS', 'KENN', 'ROBE', 'SCAL', 'SOUT', 'STEV', 'THOM',
                'KAGA', 'SOTO', 'REHN', 'BLAC', 'WHIT', 'MARS', 'BREN', 'POWE', 'BURG', 'STEW']

    conversation_with_petitioner = []
    conversation_with_respondent = []
    petitioner = 0
    respondent = 0

    f = open(output_path[:-4] + '-result.txt', 'r', encoding='utf-8')
    for line in f:
        if line.startswith('@C'):
            if any(c in line for c in ('PETITIONER', 'APPELLANT', 'PLAINTIFF')):
                petitioner = 1
                respondent = 0
            elif any(c in line for c in ('RESPONDENT', 'APPELLEE', 'DEFENDANT')):
                petitioner = 0
                respondent = 1
        elif line.startswith(tuple('*' + item for item in justices)):
            if petitioner == 1 and respondent == 0:
                conversation_with_petitioner.append(line)
            elif petitioner == 0 and respondent == 1:
                conversation_with_respondent.append(line)
    f.close()

    with open(output_path[:-4] + '-result-1.txt', 'w') as f:
        for item in conversation_with_petitioner:
            f.write("%s" % item)

    with open(output_path[:-4] + '-result-2.txt', 'w') as f:
        for item in conversation_with_respondent:
            f.write("%s" % item)


project_path = '/home/wenting/PycharmProjects/thesis/'
for year in range(1979, 2015):
    os.makedirs(project_path + 'data/split_data/%s' % year, exist_ok=True)

for root, dirs, files in os.walk(project_path + 'data/oral_arguments_text'):
    for file in files:
        if file.endswith(".cha"):
            input_dir = os.path.join(root, file)
            output_dir = re.sub('oral_arguments_text', 'split_data', input_dir)
            split_by_side(input_dir, output_dir)

