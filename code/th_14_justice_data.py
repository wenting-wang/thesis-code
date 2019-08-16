import os
import re
import json
import pandas as pd


project_path = '/home/wenting/PycharmProjects/thesis/'

justices = {'UNKN' :0, 'ALIT': 112, 'BREY': 110, 'CONN': 104, 'GINS': 109, 'KENN': 106, 'ROBE': 111,
            'SCAL': 105, 'SOUT': 107, 'STEV': 103, 'THOM': 108,'KAGA': 114, 'SOTO': 113, 'REHN': 102,
            'BLAC': 100, 'WHIT': 95, 'MARS': 98, 'BREN': 92, 'POWE': 101, 'BURG': 99, 'STEW': 94}

""" code 1, get text file justice level >> text_data_justice"""


def dict_raise_on_duplicates(ordered_pairs):
    """iterate read, not overwrite"""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            d[k].append(v)
        else:
            d[k] = v
    return d


def get_text_pe(filename, docket):
    output_path = project_path + 'data/mixed_data_justice/text_data_justice/'

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
    output_path = project_path + 'data/mixed_data_justice/text_data_justice/'

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


# get text file, justice level (18305 entries, petitioner, respondent included)
def get_text():
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
# get_text()


""" code 2, get filter from audio file, drop duplicated item in info file >> case_info_justice_filtered.csv"""
""" code 2, get filter from audio file, drop duplicated item in info file >> audio_filtered.csv"""

# filter info file, based on audio file (justice results, 5209 entries)
# info file has duplicated items. drop them. now use 'case_info_justice_filtered' as filter! (5158!!!)


def get_audio_filtered():
    # get 'audio_filtered.csv'
    caseinfo_jusctice = pd.read_csv(project_path + 'data/mixed_data_justice/caseinfo_justice.csv')
    justice_results = pd.read_csv(project_path + 'data/mixed_data_justice/justice_results.csv', sep='\t')
    audio_filter = justice_results[['docket', 'justice']]  # filter

    audio_filtered = pd.merge(audio_filter, caseinfo_jusctice, on=['docket', 'justice'], how='left')
    audio_filtered = audio_filtered.drop_duplicates(['docket', 'justice'], keep=False)
    audio_filtered.to_csv(project_path + 'data/mixed_data_justice/audio_filtered.csv', index=False)


def get_info_filtered():
    caseinfo_jusctice = pd.read_csv(project_path + 'data/mixed_data_justice/caseinfo_justice.csv')
    justice_results = pd.read_csv(project_path + 'data/mixed_data_justice/audio_filtered.csv', sep='\t')
    docket_justice_filter = justice_results[['docket', 'justice', 'petitioner_vote']]  # filter

    # get 'case_info_justice_filtered.csv'
    caseinfo_jusctice_new = pd.merge(docket_justice_filter, caseinfo_jusctice, on=['docket', 'justice'], how='left')
    caseinfo_jusctice_new = caseinfo_jusctice_new.drop_duplicates(['docket', 'justice'], keep=False)
    caseinfo_jusctice_new.to_csv(project_path + 'data/mixed_data_justice/case_info_justice_filtered.csv', index=False)

# get_audio_filtered()
# get_info_filtered()


""" code 3, get text file justice level, filtered by audio and info >> text_data_justice_filtered """


# get text file, justice level, filtered by audio (justice results, 8583 entries)
# filtered result (8583 entries, petitioner and respondent included)
justice_results = pd.read_csv(project_path + 'data/mixed_data_justice/case_info_justice_filtered.csv')
docket_justice_filter = justice_results['docket'].map(str) + '_' + justice_results['justice'].map(str)
docket_justice_filter = docket_justice_filter.to_list()  # filter (5158)


def get_text_pe_new(filename, docket):
    output_path = project_path + 'data/mixed_data_justice/text_data_justice_filtered/'

    with open(filename, 'r', encoding='utf-8') as f:
        texts = json.load(f, object_pairs_hook=dict_raise_on_duplicates)

    for justice, text in texts.items():
        justice_id = justices[justice]
        id = str(docket) + '_' + str(justice_id)
        if id in docket_justice_filter:
            text = re.sub(r'dict_values', '', str(text))
            text = re.sub(r'[\[\]\'\"(),]', '', text)
            with open(output_path + "%s_%s_petitioner" % (docket, justice_id), 'w') as f:
                f.write('{}'.format(text))
    return


def get_text_re_new(filename, docket):
    output_path = project_path + 'data/mixed_data_justice/text_data_justice_filtered/'

    with open(filename, 'r', encoding='utf-8') as f:
        texts = json.load(f, object_pairs_hook=dict_raise_on_duplicates)

    for justice, text in texts.items():
        justice_id = justices[justice]
        id = str(docket) + '_' + str(justice_id)
        if id in docket_justice_filter:
            text = re.sub(r'dict_values', '', str(text))
            text = re.sub(r'[\[\]\'\"(),]', '', text)
            with open(output_path + "%s_%s_respondent" % (docket, justice_id), 'w') as f:
                f.write('{}'.format(text))
    return


def get_text_filtered():
    for root, dirs, files in os.walk(project_path + 'data/cleaned_data'):
        for file in files:
            if file.endswith("-result-1.txt"):
                docket_num = file[:-13]
                input_file = os.path.join(root, file)
                get_text_pe_new(input_file, docket_num)

            if file.endswith("-result-2.txt"):
                docket_num = file[:-13]
                input_file = os.path.join(root, file)
                get_text_re_new(input_file, docket_num)

# get_text_filtered()
# 8563






