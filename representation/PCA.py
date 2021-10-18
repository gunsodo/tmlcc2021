import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import pandas as pd

data = pd.read_csv('train_forPCA.csv') #Instead of train_forPCA.csv enter anyfile .csv

data_name = data['MOFname'] #drop anything which is unrelated to number
data.drop(columns=['MOFname'],inplace=True)

ndf = pd.DataFrame(data)

# In some case, we have to clean some data or convert data type from string to float. 
# for k in range(len(data.columns)): 
#     each_value = data[str('atom_' + str(k+1))]
#     c = each_value.values.tolist()
#     b = 0
#     d = [0]*len(c)
#     for i in range(4):
#         for j in range(len(c)):
#             x = np.fromstring(c[j].replace('[', "").replace(']', ""), dtype=float, sep=' ')
#             d[j] = x[i]
#         ndf.insert(loc = len(ndf.columns),column = str('atom_'+str(k+1)+str(i+1)), value = d)


pca = PCA(n_components=123) # enter any number below your number of column in this case 123
pca.fit(ndf)
for i in range(1, len(pca.explained_variance_) + 1):
    print(f"# PCs = {i}: {round(sum(pca.explained_variance_ratio_[:i]), 2)}") #look which one make the sum nearest to 1

pca = PCA(n_components=123) # enter that number
print(pca)
nndf = pca.fit_transform(ndf)
new = pd.DataFrame(nndf)
new.insert(loc = 0,column = 'MOFname',value = a)
new.to_csv('train_PCA.csv',index=False)
