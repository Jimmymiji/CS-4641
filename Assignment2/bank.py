import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.model_selection import train_test_split


bank_data = pd.read_csv("data/bank-full.csv",delimiter=';')
features = list(bank_data.columns.values)
numeric_col = ['age','balance','day','duration','campaign','pdays','previous']
x_num_train = bank_data[numeric_col].as_matrix()

#categorical columns
bank_train = bank_data.drop(numeric_col+['y'],axis = 1)
x_bank_train = bank_train.T.to_dict().values()

#vectorize
vectorizer = DV( sparse = False )
vec_x_bank_train = vectorizer.fit_transform( x_bank_train )
feature_train = np.hstack(( x_num_train, vec_x_bank_train ))

#y
label_train = bank_data.as_matrix(columns=['y'])

# only take a half of the dataset to save time
train_X,test_X,train_y,test_y = train_test_split(feature_train,label_train,test_size=0.5,random_state=0)

train_X,test_X,train_y,test_y = train_test_split(train_X,train_y,test_size=0.3,random_state=0)

train = pd.DataFrame(train_X)
train['Y'] = train_y
train.to_csv(path_or_buf='./data/train.csv', index=False)

test = pd.DataFrame(test_X)
test['Y'] = test_y
test.to_csv(path_or_buf='./data/test.csv', index=False)


f = open("./data/train.csv",'r+')
f1 = open("./data/bank_train.csv",'w')
f.__next__()# skip header line
writer = csv.writer(f1)
for row in csv.reader(f):
    writer.writerow(row)
f.close()
f1.close()


f = open("./data/test.csv",'r+')
f1 = open("./data/bank_test.csv",'w')
f.__next__()# skip header line
writer = csv.writer(f1)
for row in csv.reader(f):
    writer.writerow(row)
f.close()
f1.close()