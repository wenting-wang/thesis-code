import re

file_docket = '08-645'

# make file line by line
with open(file_docket + '.cha', 'r', encoding='utf-8') as f:
    text = f.read()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'@', '\n@', text)
    text = re.sub(r'\*', '\n*', text)

with open(file_docket + '-result.txt', 'w') as f:
    f.write(text)

# split by side, only justices, no lawyers
justices = ['UNKN', 'ALIT', 'BREY', 'CONN', 'GINS', 'KENN', 'ROBE', 'SCAL', 'SOUT', 'STEV', 'THOM',
            'KAGA', 'SOTO', 'REHN', 'BLAC', 'WHIT', 'MARS', 'BREN', 'POWE', 'BURG', 'STEW']

conversation_with_petitioner = []
conversation_with_respondent = []
petitioner = 0
respondent = 0

f = open(file_docket + '-result.txt')
for line in f:
    if line.startswith('@C'):
        if 'PETITIONER' or 'APPELLANT' or 'PLAINTIFF' in line:
            petitioner = 1
            respondent = 0
        elif 'RESPONDENT' or 'APPELLEE' or 'DEFENDANT' in line:
            petitioner = 0
            respondent = 1
    elif line.startswith(tuple('*' + item for item in justices)):
        if petitioner == 1 and respondent == 0:
            conversation_with_petitioner.append(line)
        elif petitioner == 0 and respondent == 1:
            conversation_with_respondent.append(line)
f.close()

with open(file_docket + '-result-1.txt', 'w') as f:
    for item in conversation_with_petitioner:
        f.write("%s" % item)

with open(file_docket + '-result-2.txt', 'w') as f:
    for item in conversation_with_respondent:
        f.write("%s" % item)
