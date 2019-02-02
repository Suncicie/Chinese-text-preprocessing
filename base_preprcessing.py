# @Time    : 2019/2/3 上午12:58
# @Author  : Suncicie
# @Describe    : preprocess for text
# @File    : BaseProcessForText.py
# @Software: PyCharm

encoding='utf-8'
import numpy as np
import jieba
import re
import pickle
import pandas as pd
import logging
from keras.preprocessing import text, sequence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

train_path=''
test_path=''
userDic_path=''
stopWord_path=''
synonyms_path=''


logging.info("read data -----------------")
train_set=pd.read_csv(train_path)
test_set=pd.read_csv(test_path)

def filter_df(df,
                  re_filters=r'[^\u4e00-\u9fa5_a-zA-Z0-9]{3,10}|[^\u4e00-\u9fa5a-zA-Z0-9\（\）\？\《\》\；\，\。\“\”\<\>\！,.;:\?\"\'\!\(\)\（\）\s]',
                  inplace=False):
    if inplace is False:
        df['in_filter'] = df['content'].apply(lambda string: re.sub(re_filters, "", string))
    else:
        df['in_filter'] = df['content'].apply(lambda string: re.sub(re_filters, "", string))
    return df



logging.info("filter aarbled -----------------")
train_set=filter_df(train_set)
test_set=filter_df(test_set)

# Traditional and simplified conversion, punctuation conversion, digital replacement
from DataProcess.langconv import *
train_set['in_filter'] = train_set['in_filter'].apply(lambda sentence: Converter('zh-hans').convert(sentence))
test_set['in_filter'] = test_set['in_filter'].apply(lambda sentence: Converter('zh-hans').convert(sentence))


logging.info("cut sentence -----------------")
jieba.load_userdict(userDic_path)
train_set["word"]=train_set["in_filter"].apply(lambda x:
                                                 ",".join(jieba.cut(x,cut_all=False)).split(","))
test_set["word"]=test_set["in_filter"].apply(lambda x:
                                                 ",".join(jieba.cut(x,cut_all=False)).split(","))

logging.info("stop words -------------")
stopwords=[]
for word in open(stopWord_path,'r',encoding="utf-8"):
    stopwords.append(word.strip())

# train_set["word"]=train_set["word"].apply(lambda x: [i for i in x if i not in stopwords ])
# test_set["word"]=test_set["word"].apply(lambda x: [i for i in x if i not in stopwords ])

def filter_char_map(arr):
    res = []
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0'and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return res

train_set["word"]=train_set["word"].map(lambda x: filter_char_map(x))
test_set["word"]=test_set["word"].map(lambda x: filter_char_map(x))


synon = pd.read_csv(synonyms_path)
word_map = {}
for i in range(0, len(synon)):
    word = synon.loc[i, "word"]
    syn = synon.loc[i, "syn"]
    word_map[syn] = word

train_set["word"] = train_set["word"].apply(lambda x: [word_map.get(i, i) for i in x])
test_set["word"] = test_set["word"].apply(lambda x: [word_map.get(i, i) for i in x])

train_set["word"] = train_set["word"].apply(lambda x: " ".join(x))
test_set["word"] = test_set["word"].apply(lambda x: " ".join(x))

# with open('data/train_set_v4_1.pickle', 'wb') as f:
#     pickle.dump(train_set, f)
#
# with open('data/test_set_v4_1.pickle', 'wb') as f:
#     pickle.dump(test_set, f)


# word_freq
from tqdm import tqdm_notebook as tqdm
word_dict = {}
def construct_dict(df):
    # build vorb
    for line in tqdm(df.word):
        for e in line:
            word_dict[e] = word_dict.get(e, 0) + 1
    return word_dict

word_dict = construct_dict(train_set)


word_dic_df=pd.DataFrame(word_dict,index=range(1,3))
word_dic_df=word_dic_df.T
word_dic_df.columns=["count1","count2"]
word_dic_df.sort_values(["count1"],inplace=True,ascending=False)
word_dic_df.to_csv("data/fre_dic_v2.csv")



# w2v 更新同意词表

num_words=250000
col="word"
df_train=train_set
df_test=test_set
del train_set,test_set
gc.collect()
maxlen_=450
victor_size=200

tokenizer = text.Tokenizer(num_words=num_words, lower=False, filters="")
tokenizer.fit_on_texts(list(df_train[col].values) + list(df_test[col].values))
train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=maxlen_)
test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=maxlen_)
word_index = tokenizer.word_index

count = 0
nb_words = len(word_index)
print(nb_words)
all_data = pd.concat([df_train[col], df_test[col]])

file_name = 'data/embedding/' + 'Word2Vec_v2' + col + "_" + str(victor_size) + '.model'
logging.info("start training word2vec ...")
if not os.path.exists(file_name):
    model = Word2Vec([[word for word in document] for document in all_data.values],
                         size=victor_size, window=5, iter=10, workers=11, seed=2018, min_count=2)
    model.save(file_name)
else:
    model = Word2Vec.load(file_name)
logging.info("add word2vec finished....")

# fredic
# fre_dic=pd.read_csv("data/fre_dic_v2.csv",encoding="utf-8")
# fre_dic=fre_dic.iloc[0:150,]
# fre_dic.columns=["word","count","count1"]
# fre_dic=fre_dic[["word","count"]]
# # 可以手动加要替换的词
# fre_dic["word"].to_csv("data/fre_dic_top150_v2.csv",index=None)
fre_dic = pd.read_csv("data/fre_dic_top150_v2.csv", encoding="utf-8", header=None)
fre_dic.columns = ["word"]


def get_simword(word):
    simlist = ""
    for i in model.most_similar(positive=[word]):
        simlist = simlist + i[0] + " "
    return simlist


fre_dic["sim_word"] = fre_dic["word"].apply(get_simword)
# fre_dic.to_csv("data/syn_dic.csv")

# 有syn 在word 中时，合并
