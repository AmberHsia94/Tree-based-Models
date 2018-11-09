import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = pd.read_csv('cs-training.csv', sep=',')
del data[data.columns[0]]
print data[:3]
print data.shape  # (150000, 11)

# y-label
data_new = data[np.isfinite(data['SeriousDlqin2yrs'])]    # check Nan/Null for y-label
print data_new.shape # (150000, 11)

# visualization
# y-label
#x = np.arange(len(data_new))
#plt.scatter(x, data_new['Cover_Type'])
#plt.show()
#plt.savefig('prcp.png')

# duplicates
data_new = data_new.drop_duplicates()
print('after drop_duplicated: ', len(data_new)) # 149391


# Nan/NuLL
print data_new.isnull().any().sum()  # 2
null_columns=data_new.columns[data_new.isnull().any()]
print('mean: ',data_new[null_columns].mean())
print('mode: ',data_new[null_columns].mode())
print('median: ',data_new[null_columns].median())
data_new = data_new.fillna(data_new.median())
data_new = data_new.dropna(axis=0, how="any")
print data_new.shape    #(150000, 11)


for item in data_new.columns:
    print item
    print len(data_new[item].unique())
print '=================================='

from sklearn.utils import shuffle
data_new = shuffle(data_new, random_state=0).reset_index(drop=True)
data_new['flag'] = 0
data_new.ix[int(0.8*len(data_new))+1:,'flag'] = 1
print len(data_new[data_new['flag']==0])

#data_new.insert(0, 'label', data_new.pop('prcp'))
#data_new.ix[:,1:] = (data_new.ix[:,1:]-data_new.ix[:,1:].min())/(data_new.ix[:,1:].max()-data_new.ix[:,1:].min())

from sklearn import preprocessing
data_new.ix[:,1:-1] = preprocessing.scale(data_new.ix[:,1:-1])
print data_new.ix[:, 1:-1].mean(axis=0)
print data_new.ix[:, 1:-1].std(axis=0)

print data_new[:3]
#category = ['Cover_Type']
#for feature in category:
#    #print feature
#    #print len(data_new[feature].unique()) # [2,2,2,2,2,7]
#    data_new_1 = pd.get_dummies(data_new[feature], prefix=feature)
#    data_new = pd.concat([data_new, data_new_1], axis=1)
#    del data_new[feature]
#print data_new.shape  #(581011, 61)

#data_new.to_csv('q4credit.csv')


