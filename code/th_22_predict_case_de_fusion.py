import pandas as pd

# predict decision for case level, using results from justice level training

audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/audio_filtered.csv'
audio = pd.read_csv(audio_file)

docket_justice_id = audio['docket'].map(str) + '_' + audio['justice'].map(str)
audio['docket_justice_id'] = docket_justice_id
audio = audio.sort_values(by=['docket_justice_id'])  # sort, order

prediction = pd.DataFrame()
prediction['docket_justice_id'] = audio['docket_justice_id']
prediction['docket'] = audio['docket']
prediction['true_vote'] = audio['petitioner_vote']

info_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/de_pre_info.csv'
text_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/de_pre_text.csv'
audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/de_pre_audio.csv'
info_text_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/pre_info_text.csv'
info_audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/pre_info_audio.csv'
text_audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/pre_text_audio.csv'
info_text_audio = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/pre_info_text_audio.csv'

pre_info = pd.read_csv(info_file, names=['pre_info'])
pre_text = pd.read_csv(text_file, names=['pre_text'])
pre_audio = pd.read_csv(text_file, names=['pre_audio'])
pre_info_text = pd.read_csv(text_file, names=['pre_info_text'])
pre_info_audio = pd.read_csv(text_file, names=['pre_info_audio'])
pre_text_audio = pd.read_csv(text_file, names=['pre_text_audio'])
pre_info_text_audio = pd.read_csv(text_file, names=['pre_info_text_audio'])

prediction = pd.concat([prediction, pre_info], axis=1)
prediction = pd.concat([prediction, pre_text], axis=1)
prediction = pd.concat([prediction, pre_audio], axis=1)
prediction = pd.concat([prediction, pre_info_text], axis=1)
prediction = pd.concat([prediction, pre_info_audio], axis=1)
prediction = pd.concat([prediction, pre_text_audio], axis=1)
prediction = pd.concat([prediction, pre_info_text_audio], axis=1)

prediction.to_csv('/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/de_prediction.csv', index=False)


# calculate
results = prediction.groupby('docket', as_index=False).sum()
summary = results.iloc[:, 1:] >= 5  # vote for petitioner

(summary['true_vote'] == summary['pre_info']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_text']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_audio']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_info_text']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_info_audio']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_text_audio']).value_counts()[1]/997
(summary['true_vote'] == summary['pre_info_text_audio']).value_counts()[1]/997









