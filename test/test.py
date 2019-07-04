import pandas as pd

filepath_dict = {'yelp':   '~/PycharmProjects/thesis/test/sentiment_analysis/yelp_labelled.txt',
                 'amazon': '~/PycharmProjects/thesis/test/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   '~/PycharmProjects/thesis/test/sentiment_analysis/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
