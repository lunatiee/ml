import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence


sr = pd.Series([17000,18000, 1000, 5000],index =["피자", "치킨", "콜라", "맥주"])
print('시리즈 출력 :')
print('-'*15)
print(sr)

print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

tokenizer = TreebankWordTokenizer()

text1 = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
text2 = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

print('트리뱅크 워드토크나이저1 :', tokenizer.tokenize(text1))
print('트리뱅크 워드토크나이저2 :', tokenizer.tokenize(text2))
