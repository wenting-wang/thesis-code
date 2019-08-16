import os
import re
import json
import pandas as pd


project_path = '/home/wenting/PycharmProjects/thesis/'


""" code 2, get filter from audio file, drop duplicated item in info file >> caseinfo_filtered.csv"""
""" code 2, get filter from audio file, drop duplicated item in info file >> audio_filtered.csv"""

# filter info file, based on audio file (1000)
# info file has duplicated items. drop them. now use 'audio_filtered' as filter!


def group_audio_data():
    # 1002 cases
    audio = pd.read_csv(project_path + 'data/mixed_data/justice_results.csv', sep='\t')
    # get mean of vocal pitch, case level
    audio_case = audio.groupby('docket', as_index=False)['petitioner_pitch', 'respondent_pitch'].mean()
    audio_case = audio_case.sort_values(by=['docket'])  # sort, order
    audio_case.to_csv(project_path + 'data/mixed_data/audio_grouped.csv', index=False)


def get_audio_filtered():
    # get 'audio_filtered.csv', 1000 cases
    caseinfo = pd.read_csv(project_path + 'data/mixed_data/caseinfo_original.csv')
    audio = pd.read_csv(project_path + 'data/mixed_data/audio_grouped.csv', sep=',')

    audio_filtered = pd.merge(audio, caseinfo[['docket']], on=['docket'], how='left')
    audio_filtered = audio_filtered.drop_duplicates(['docket'], keep=False)
    audio_filtered.to_csv(project_path + 'data/mixed_data/audio_filtered.csv', index=False)


def get_info_filtered():
    # get 'caseinfo_filtered.csv', 1000 cases
    caseinfo = pd.read_csv(project_path + 'data/mixed_data/caseinfo_original.csv')
    audio = pd.read_csv(project_path + 'data/mixed_data/audio_filtered.csv', sep=',')
    docket_filter = audio[['docket']]  # filter, based on audio

    # get 'caseinfo_filtered.csv'
    caseinfo_new = pd.merge(docket_filter, caseinfo, on=['docket'], how='left')
    caseinfo_new = caseinfo_new.drop_duplicates(['docket'], keep=False)
    caseinfo_new.to_csv(project_path + 'data/mixed_data/caseinfo_filtered.csv', index=False)

# group_audio_data()
# get_audio_filtered()
# get_info_filtered()
# 49 cases only. very little audio data


""" code 3, get text file justice level, filtered by audio and info >> text_data_filtered """

# 3506 * 2 -> 1000 * 2
# get text file, case level, filtered by audio (1000)
# filtered result (1000 entries, petitioner and respondent included)
audio_filtered = pd.read_csv(project_path + 'data/mixed_data/caseinfo_filtered.csv')
audio_filter = audio_filtered['docket'].to_list()  # filter (1000)


def get_text_pe_new(filename, docket):
    output_path = project_path + 'data/mixed_data/text_data_filtered/'

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    if docket in audio_filter:
        text = re.sub(r'[\[\]\'\"(),]', '', text)
        with open(output_path + "%s_petitioner" % docket, 'w') as f:
            f.write('{}'.format(text))
    return


def get_text_re_new(filename, docket):
    output_path = project_path + 'data/mixed_data/text_data_filtered/'

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    if docket in audio_filter:
        text = re.sub(r'[\[\]\'\"(),]', '', text)
        with open(output_path + "%s_respondent" % docket, 'w') as f:
            f.write('{}'.format(text))
    return


def get_text_filtered():
    for root, dirs, files in os.walk(project_path + 'data/mixed_data/text_data'):
        for file in files:
            if file.endswith("petitioner"):
                docket_num = file[:-11]
                input_file = os.path.join(root, file)
                get_text_pe_new(input_file, docket_num)

            if file.endswith("respondent"):
                docket_num = file[:-11]
                input_file = os.path.join(root, file)
                get_text_re_new(input_file, docket_num)

# get_text_filtered()
# 1111






