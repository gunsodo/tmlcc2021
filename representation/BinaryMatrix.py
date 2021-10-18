import pandas as pd

def distinct(feature):
    fg_name = []
    for f in feature:
        #skip if the feature is not str
        if not isinstance(f, str): 
            continue
        org = f.split("-")
        for og in org:
            if og not in fg_name:
                fg_name.append(og)
    print(fg_name)    
    return fg_name

def df(data, feature, n_list):
    for f in n_list:
        rep = []
        for g in feature:
            n = 0  
            if not isinstance(g, str): 
                rep.append(n)
                continue
            g = str(g).split("-")
            for k in g:    
                if k == f:
                    n += 1
                    break 
            rep.append(n)
        data[f] = rep
    return data

def bm(in_name, out_name):
    dataset = pd.read_csv(in_name)
    #extract functional group and topology
    fun_group = dataset.iloc[:,6].values
    topo = dataset.iloc[:,10].values

    #create dictionary of binary matrix representation
    data = {'MOFname': dataset.iloc[:,0].values}
    #create set of distinct functional group and topology
    dfun_group = distinct(fun_group)
    dtopo = distinct(topo)

    #append binary matrix of functional group
    df(data, fun_group, dfun_group);
    #append binary matrix of topology
    df(data, topo, dtopo);

    #convert dictionary into dataframe and export to csv
    dframe = pd.DataFrame(data)
    dframe.to_csv(out_name, index=False)
    
bm('train_rm.csv', 'train_bm.csv')
bm('pretest.csv', 'pretest_bm.csv')
bm('test.csv', 'test_bm.csv')
