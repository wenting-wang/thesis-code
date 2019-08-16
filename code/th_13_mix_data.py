import pandas as pd
import numpy as np
import os
from sklearn.decomposition import TruncatedSVD
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# load text, info, audio data, in [docket] order

def load_arguments_text(info_file, text_dir, max_num_words, max_length):
    # '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/text_data_justice_filtered'
    # 3506 cases
    case_info = pd.read_csv(info_file)
    docket = case_info['docket']
    docket = docket.sort_values()  # sort, order
    docket = docket.to_list()  # filter

    arguments_petitioner = []
    arguments_respondent = []

    # all cases in order, follow csv data
    for id in docket:
        pe_path = os.path.join(text_dir, '{}_petitioner'.format(id))
        re_path = os.path.join(text_dir, '{}_respondent'.format(id))

        try:
            with open(pe_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except IOError:
            text = ''
        arguments_petitioner.append(text)
        try:
            with open(re_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except IOError:
            text = ''
        arguments_respondent.append(text)

    petitioner_len = len(arguments_petitioner)
    texts = arguments_petitioner + arguments_respondent

    # vectoring
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index  # len(word_index) unique tokens
    # padding
    texts_pad = pad_sequences(sequences, maxlen=max_length)
    # texts_pad[0] is petitioner arguments, texts_pad[1] is respondent arguments
    texts_pad = np.dstack((texts_pad[:petitioner_len], texts_pad[petitioner_len:]))  # (3506, 500, 2)

    return texts_pad, word_index

# load data
# mix_data_path = '/home/wenting/PycharmProjects/thesis/data/mixed_data/'
# info_path = mix_data_path + 'caseinfo.csv'
# text_path = mix_data_path + 'text_data'
# df = load_arguments_text(info_path, text_path)


# process data
# fill NA with 0
def load_structured_data(info_file):
    # (3506->1000, 601), 600 svd features + 1 target
    # '/home/wenting/PycharmProjects/thesis/data/mixed_data/caseinfo_filtered.csv'
    info = pd.read_csv(info_file)
    info = info.drop(['dateDecision', 'dateArgument', 'dateRearg'], axis=1)
    info = info.sort_values(by=['docket'])  # sort, order
    info = info.fillna(0)

    # one-hot-encoding
    for col in info.columns:
        if info[col].dtype == np.float64:
            info[col] = info[col].astype(np.int64)

    cat_columns = ['decisionType', 'naturalCourt', 'term',
                   'chief', 'petitioner', 'petitionerState',
                   'respondent', 'respondentState', 'jurisdiction', 'adminAction',
                   'adminActionState', 'threeJudgeFdc', 'caseOrigin', 'caseOriginState',
                   'caseSource', 'caseSourceState', 'lcDisagreement', 'certReason',
                   'declarationUncon', 'issue', 'issueArea', 'lawType', 'lawSupp']

    info_encoded = pd.get_dummies(info, columns=cat_columns, prefix_sep='_')
    # cat_dummies = [col for col in info_processed if '_' in col and col.split('_')[0] in cat_columns]
    # processed_columns = list(info_processed.columns[:])  # save the orders (1492 columns)

    # SVD
    features = info_encoded.drop(['docket', 'partyWinning'], axis=1)
    target = info_encoded['partyWinning']

    svd = TruncatedSVD(n_components=600, random_state=42)
    features_reduced = svd.fit_transform(features)
    # print(svd.explained_variance_ratio_.sum())  # 0.962033
    features_reduced = pd.DataFrame(features_reduced)

    info_processed = pd.concat([features_reduced, target], axis=1)

    return info_processed

# process test data, following same order
# df_test_processed = pd.get_dummies(df_test, prefix_sep='_', columns=cat_columns)
# Remove additional columns
# for col in df_test_processed.columns:
#     if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
#         print("Removing additional feature {}".format(col))
#         df_test_processed.drop(col, axis=1, inplace=True)
# for col in cat_dummies:
#     if col not in df_test_processed.columns:
#         print("Adding missing feature {}".format(col))
#         df_test_processed[col] = 0
# df_test_processed = df_test_processed[processed_columns]


# # try xgboost
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(features_reduced, target, test_size=0.33, random_state=42)
# # fit model no training data
# model = XGBClassifier()
# model.fit(X_train, y_train)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 63.96%


def load_audio_data(audio_file):
    # 111 cases
    # '/home/wenting/PycharmProjects/thesis/data/mixed_data/audio_filtered.csv'
    audio = pd.read_csv(audio_file)
    audio = audio.sort_values(by=['docket'])  # sort, order

    return audio[['petitioner_pitch', 'respondent_pitch']]


