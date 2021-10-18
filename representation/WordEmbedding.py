import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def perform_removal(word):
    nword = ""
    for c in word:
        if c == '-':
            nword += " "
        else:
            nword += c
    return nword

def clear_list(name,i,n_array):
    lst = []
    lst.append(name[i])
    for a in n_array:
        for s in a:
            lst.append(s)
            if len(a)==1: lst.append(s)
    return lst

def we(in_name, out_name):
    dataset = pd.read_csv('test.csv')
    fun_group = dataset.iloc[:,6].values
    mof_name = dataset.iloc[:,0].values  
    docs = []
    for fg in fun_group:
        if isinstance(fg, str):
            doc = perform_removal(fg).split()
            docs.append(doc)
        
    model = Word2Vec(sentences=docs, vector_size=2, window = 1, min_count=20, workers=4, sg=0, epochs=5)

    sep = []
    for i in docs:
        for j in i:
            if j not in sep:
                sep.append(j)
            
    data_v = []
    for i in sep:
        data_v.append(model.wv.get_vector(i))
    vec = np.transpose(np.array(data_v))

    data = []
    for i, s in enumerate(fun_group):
        if not isinstance(s, str): 
            data.append([0,0,0,0])
            continue
        org = s.split("-")
        enc = []
        for fg in org:
            en = []
            for ele in sep:
                if fg == ele:
                    en.append(1)
                else:
                    en.append(0)
            enc.append(en)
        arr = np.transpose(np.array(enc))
        h_layer = vec.dot(arr)
        h_layer = h_layer.tolist()
        data.append(clear_list(mof_name,i,h_layer))
    
    dframe = pd.DataFrame(data, columns=["MOFName", 1, 2, 3, 4])
    dframe.to_csv('test_we.csv', index=False)
    
we('train_rm.csv', 'train_we.csv')
we('pretest.csv', 'pretest_we.csv')
we('test.csv', 'test_we.csv')
