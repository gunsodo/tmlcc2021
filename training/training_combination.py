import numpy as np
import pandas as pd

'''
This file will use not more than two representations to train throughout the selected model
and create a .csv file of the LMAE value of each combination, which will be used for creating heatmaps.
'''

model = 'xgb'   #model that you want to study representation combinations

representation = ['preprocessed','preprocessed2','binary','word_embedding','PCA_linearAP-RDF','PCA_constAP-RDF','PCA_coordinates','PCA_BOA']

LMAE_map = pd.DataFrame(data=np.zeros((8,8)),columns=representation)
LMAE_map.insert(loc=0, column='representation', value=representation)

for i in range(8):
    rep1 = representation[i]

    for j in range(i+1):
        rep2 = representation[j]
        lmae = 0.0
        
        print('Now training: ',j,rep2,' ',i,rep1)

        if i == j:
            ! python main.py -r $rep1 -m $model -d results > results.txt
        else:
            ! python main.py -r $rep2 $rep1 -m $model -d results > results.txt
        
        f = open("results.txt")
        for l in f.readlines():
            if "LMAE loss: " in l:
                lmae = float(l[11:])
        f.close()

        print(lmae)

        LMAE_map[rep1][j] = lmae

LMAE_map.to_csv('/Export/LMAE_'+str(model)+'.csv',index=False)
        
