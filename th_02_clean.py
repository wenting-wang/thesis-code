import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt

file_docket = '08-645'
filename_1 = file_docket + '-result-1.txt'
filename_2 = file_docket + '-result-2.txt'
justices = ['ALIT', 'BREY', 'CONN', 'GINS', 'KENN', 'ROBE', 'SCAL', 'SOUT', 'STEV', 'THOM', 'UNKN']

# split by justice


def load_data(name):
    dict_by_justices = {}
    for item in justices:
        dict_by_justices['%s' % item] = []
    f = open(name, 'r', encoding='utf-8')
    for line in f:
        dict_by_justices[line[1:5]].append(line[6:])
    f.close()
    return dict_by_justices


def clean(text):
    # text pre-processing
    tokens = [w.lower() for w in word_tokenize(text)]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    words = [word for word in stripped if word.isalpha()]  # remove non-alphabetic
    words = [w for w in words if w not in set(stopwords.words('english'))]  # filter out stop words

    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]  # stemmed

    return ' '.join(words)


def sentiment_score(text):
    text = TextBlob(clean(text))
    return text.sentiment.polarity, text.sentiment.subjectivity


def result_score(dict_argument, side):
    print(side)
    polarity_list = []
    subjectivity_list = []
    for key, value in dict_argument.items():
        if value:
            text_by_justices = ' '.join(value)
            polarity, subjectivity = sentiment_score(clean(text_by_justices))
            polarity_list.append(polarity)
            subjectivity_list.append(subjectivity)
            print(key + ' polarity: %.4f' % polarity + ', subjectivity: %.4f' % subjectivity)
        else:
            print(key + ' did not speak.')
    result = polarity_list, subjectivity_list
    return result


def show_plot(result_pe, result_re):
    plt.figure(figsize=(8, 5))
    plt.scatter(result_pe[0], result_pe[1], c='blue', marker='s', s=50)
    for x, y in zip(result_pe[0], result_pe[1]):
        plt.annotate('(%.2f, %.2f)' % (x, y), xy=(x, y), xytext=(0, -10),
                     textcoords='offset points', ha='center', va='top')
    plt.scatter(result_re[0], result_re[1], c='red', marker='o', s=50)
    for x, y in zip(result_re[0], result_re[1]):
        plt.annotate('(%.2f, %.2f)' % (x, y), xy=(x, y), xytext=(0, -10),
                     textcoords='offset points', ha='center', va='top')
    plt.xlim([-1, 1])
    plt.ylim([0, 1])
    plt.xlabel('polarity')
    plt.ylabel('subjectivity')
    plt.show()


result_pe = result_score(load_data(filename_1), 'Petitioner --------')
result_re = result_score(load_data(filename_2), 'Respondent --------')
show_plot(result_pe, result_re)
