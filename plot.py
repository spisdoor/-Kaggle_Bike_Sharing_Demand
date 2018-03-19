import pandas as pd
import matplotlib.pyplot as plt

target_column = 'windspeed'

df = pd.read_csv('data/train.csv', index_col=None, na_values=['NA'])
count = df.groupby(target_column).count()['count']

ax1 = count.plot(kind='bar', color='y')
ax1.set_xlabel(target_column)
ax1.set_ylabel('count')
plt.title(target_column + ' count')
plt.subplots_adjust(bottom=0.17)
plt.savefig('plot/' + target_column + '_plot.png')
