import pandas as pd
import numpy as np

# split train, dev, test (7:2:1)

# petitioner - cola
argument = pd.read_csv('/home/wenting/Documents/bert/data/argument/argument.csv')
argument['partyWinning'] = argument['partyWinning'].astype(int)
argument = argument.drop(['caseDisposition', 'respondent'], axis=1)
argument.insert(loc=2, column='alpha', value='a')
argument.to_csv('/home/wenting/Documents/bert/data/argument/argument_petitioner.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')

msk = np.random.rand(len(argument)) < 0.9
experiment = argument[msk]
test = argument[~msk]

msk2 = np.random.rand(len(experiment)) < 0.75
train = experiment[msk2]
dev = experiment[~msk2]

train.to_csv('/home/wenting/Documents/bert/data/argument/petitioner/train.tsv', header=False,
                sep='\t', index=False, encoding='utf-8')
dev.to_csv('/home/wenting/Documents/bert/data/argument/petitioner/dev.tsv', header=False,
                sep='\t', index=False, encoding='utf-8')
test = test.drop(['partyWinning', 'alpha'], axis=1)
test.to_csv('/home/wenting/Documents/bert/data/argument/petitioner/test.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')


# respondent - cola
argument = pd.read_csv('/home/wenting/Documents/bert/data/argument/argument.csv')
argument['partyWinning'] = argument['partyWinning'].astype(int)
argument = argument.drop(['caseDisposition', 'petitioner'], axis=1)
argument.insert(loc=2, column='alpha', value='a')
argument.to_csv('/home/wenting/Documents/bert/data/argument/argument_respondent.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')

msk = np.random.rand(len(argument)) < 0.9
experiment = argument[msk]
test = argument[~msk]

msk2 = np.random.rand(len(experiment)) < 0.75
train = experiment[msk2]
dev = experiment[~msk2]

train.to_csv('/home/wenting/Documents/bert/data/argument/respondent/train.tsv', header=False,
                sep='\t', index=False, encoding='utf-8')
dev.to_csv('/home/wenting/Documents/bert/data/argument/respondent/dev.tsv', header=False,
                sep='\t', index=False, encoding='utf-8')
test = test.drop(['partyWinning', 'alpha'], axis=1)
test.to_csv('/home/wenting/Documents/bert/data/argument/respondent/test.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')

# pair sentence - mrpc
argument = pd.read_csv('/home/wenting/Documents/bert/data/argument/argument.csv')
argument['partyWinning'] = argument['partyWinning'].astype(int)
argument = argument.drop(['caseDisposition'], axis=1)
argument.insert(loc=2, column='alpha', value='a')
argument.insert(loc=3, column='beta', value='b')
argument.to_csv('/home/wenting/Documents/bert/data/argument/argument_pair.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')

msk = np.random.rand(len(argument)) < 0.9
experiment = argument[msk]
test = argument[~msk]

experiment = experiment.drop('docket', axis=1)

msk2 = np.random.rand(len(experiment)) < 0.75
train = experiment[msk2]
dev = experiment[~msk2]

train.to_csv('/home/wenting/Documents/bert/data/argument/pair/train.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')
dev.to_csv('/home/wenting/Documents/bert/data/argument/pair/dev.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')
test = test.drop('partyWinning', axis=1)
test.to_csv('/home/wenting/Documents/bert/data/argument/pair/test.tsv', header=True,
                sep='\t', index=False, encoding='utf-8')

