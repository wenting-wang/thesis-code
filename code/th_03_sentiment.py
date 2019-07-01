from textblob import TextBlob
import matplotlib.pyplot as plt


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



