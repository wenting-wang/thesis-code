import string
import re
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


"""
clean text, categorize text to each justice, write to folder 'cleaned_data'.

first to:
{'justiceA': [sentence1, sentence2], 'justiceB': [sentence1, sentence2], ...}

clean details:
# remove punctuation
# remove non-alphabetic
# filter out stop words
# stemmed

finally to (json file):
{'justiceA':[word1, word2, ...], 'justiceB':[word1, word2, ...], ...}

"""


def get_justice_name(filepath):
    # extract all justices name in a case
    f = open(filepath, 'r', encoding='utf-8')
    justices_set = set(re.findall(r'\*....', f.read()))
    justices_list = [re.sub(r'\*', '', i) for i in justices_set]
    f.close()
    return justices_list


def get_text_per_justice(filepath):
    justices = get_justice_name(filepath)
    # dictionary, text of each justice
    f = open(filepath, 'r', encoding='utf-8')
    dict_by_justices = {}
    for justice in justices:
        dict_by_justices['%s' % justice] = []
    for line in f:
        dict_by_justices[line[1:5]].append(line[6:])
    f.close()
    return dict_by_justices


def clean(text, stem=False, lemm=False):
    # text pre-processing
    tokens = [w.lower() for w in word_tokenize(text)]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    words = [word for word in stripped if word.isalpha()]  # remove non-alphabetic
    words = [w for w in words if w not in set(stopwords.words('english'))]  # filter out stop words
    if stem:
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]  # stemmed
    if lemm:
        lemma = WordNetLemmatizer()
        words = [lemma.lemmatize(word) for word in words]  # lemmatized

    return ' '.join(words)


def get_cleaned_text_per_justice(input_path, output_path):
    # build dictionary for cleaned text per justice
    justices = get_justice_name(input_path)
    cleaned_text_per_justice = {}
    for justice in justices:
        cleaned_text_per_justice['%s' % justice] = []
    # clean and write
    dict_argument = get_text_per_justice(input_path)
    for key, value in dict_argument.items():
        if value:
            text_by_justices = ' '.join(value)
            cleaned_text = clean(text_by_justices)
            cleaned_text_per_justice[key].append(cleaned_text)
        else:
            pass  # some justice did not speak
    with open(output_path, 'w') as f:
        json.dump(cleaned_text_per_justice, f)
        # f.write(json.dumps(cleaned_text_per_justice))


project_path = '/home/wenting/PycharmProjects/thesis/'
for year in range(1979, 2015):
    os.makedirs(project_path + 'data/cleaned_data/%s' % year, exist_ok=True)

for root, dirs, files in os.walk(project_path + 'data/split_data'):
    for file in files:
        if file.endswith(("-result-1.txt", "-result-2.txt")):
            input_file = os.path.join(root, file)
            output_file = re.sub('split_data', 'cleaned_data', input_file)
            get_cleaned_text_per_justice(input_file, output_file)
