import os
import re
import json


project_path = '/home/wenting/PycharmProjects/thesis/'
output_path = project_path + 'data/mixed_data_justice/'

justices = {'UNKN' :0, 'ALIT': 112, 'BREY': 110, 'CONN': 104, 'GINS': 109, 'KENN': 106, 'ROBE': 111,
            'SCAL': 105, 'SOUT': 107, 'STEV': 103, 'THOM': 108,'KAGA': 114, 'SOTO': 113, 'REHN': 102,
            'BLAC': 100, 'WHIT': 95, 'MARS': 98, 'BREN': 92, 'POWE': 101, 'BURG': 99, 'STEW': 94}


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            d[k].append(v)
        else:
            d[k] = v
    return d


def get_text_pe(filename, docket):
    with open(filename, 'r', encoding='utf-8') as f:
        texts = json.load(f, object_pairs_hook=dict_raise_on_duplicates)

    for justice, text in texts.items():
        justice_id = justices[justice]
        if justice_id != 0:
            text = re.sub(r'dict_values', '', str(text))
            text = re.sub(r'[\[\]\'\"(),]', '', text)
            with open(output_path + "%s_%s_petitioner" % (docket, justice_id), 'w') as f:
                f.write('{}'.format(text))
    return


def get_text_re(filename, docket):
    with open(filename, 'r', encoding='utf-8') as f:
        texts = json.load(f, object_pairs_hook=dict_raise_on_duplicates)

    for justice, text in texts.items():
        justice_id = justices[justice]
        if justice_id != 0:
            text = re.sub(r'dict_values', '', str(text))
            text = re.sub(r'[\[\]\'\"(),]', '', text)
            with open(output_path + "%s_%s_respondent" % (docket, justice_id), 'w') as f:
                f.write('{}'.format(text))
    return


for root, dirs, files in os.walk(project_path + 'data/cleaned_data'):
    for file in files:
        if file.endswith("-result-1.txt"):
            docket_num = file[:-13]
            input_file = os.path.join(root, file)
            get_text_pe(input_file, docket_num)

        if file.endswith("-result-2.txt"):
            docket_num = file[:-13]
            input_file = os.path.join(root, file)
            get_text_re(input_file, docket_num)
