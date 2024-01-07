from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
  # 온점 제거 및 형태소 분석
  document = document.replace('.', '')
  tokenized_document = okt.morphs(document)

  word_to_index = {}
  bow = []



  for word in tokenized_document:  
    if word not in word_to_index.keys():
      word_to_index[word] = len(word_to_index)  
      # BoW에 전부 기본값 1을 넣는다.
      bow.insert(len(word_to_index) - 1, 1)
    else:
      # 재등장하는 단어의 인덱스
      index = word_to_index.get(word)
      # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
      bow[index] = bow[index] + 1

  return word_to_index, bow

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'
doc3 = doc1 + ' ' + doc2


doc = []
doc.insert(0, doc1)
doc.insert(1, doc2)
doc.insert(2, doc3)


for i in range(len(doc)):
  vocab, bow = build_bag_of_words(doc[i])
  print('-'*15)
  print('분석 문장:', doc[i])
  print('doc', i+1)
  print('vocabulary :', vocab)
  print('bag of words vector :', bow)
  
