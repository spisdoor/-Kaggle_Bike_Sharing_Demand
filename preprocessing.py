# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', na_values=['NA'])
test = pd.read_csv('data/test.csv', na_values=['NA'])
train = train.append(test)
train.reset_index(inplace=True, drop=True)

train['year'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[0])
train['month'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[1])
train['day'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[2])
train['hour'] = train['datetime'].apply(lambda x: x.split()[1].split(':')[0]).astype(int)

train = pd.get_dummies(train, columns=['weather'])

train.iloc[0:10886,:].to_csv('train1.csv', index=False)
train.iloc[10886:,:].to_csv('test1.csv', index=False)
