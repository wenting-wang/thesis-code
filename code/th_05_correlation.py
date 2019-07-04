import json
from textblob import TextBlob
import re

with open('/home/wenting/PycharmProjects/thesis/code/77-1546-result-1.txt', 'r') as f:
    text = json.load(f)

text = re.sub(r'dict_values', '', str(text.values()))
text = re.sub(r'[\[\]\'\"\(\)\,]', '', text)


text = TextBlob(text)
text.sentiment
