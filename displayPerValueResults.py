import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math
import os
from diffTools import doClip

pp = pprint.PrettyPrinter(indent=4)

with open('results.json', 'r') as f:
    resFull = json.load(f)

dfFull = pd.DataFrame(resFull)
pp.pprint(list(dfFull.columns))
print("Methods:")
pp.pprint(list(np.unique(dfFull['method'])))

dfMain = dfFull[(dfFull['method'] == 'manual-ml') & (dfFull['numPredCols'] > 18)]
dfMain['Framework'] = 'Non-member'

dfNonCatVal = dfMain[(dfMain['column'].notnull())]
dfNonCat = dfMain[(dfMain['accuracy'].notnull())]
dfNonCon = dfMain[(dfMain['errorPrecision'].notnull())]
dfPrior = dfFull[(dfFull['method'] == 'prior')]
dfPrior['Framework'] = 'Prior'
dfPriorCat = dfPrior[dfPrior['accuracy'].notnull()]
dfPriorCon = dfPrior[dfPrior['errorPrecision'].notnull()]
dfPriorCatVal = dfPrior[(dfPrior['method'] == 'prior') & dfFull['column'].notnull()]
df10Cat = dfFull[(dfFull['method'] == 'manual-ml-10') & dfFull['accuracy'].notnull()]
df10Con = dfFull[(dfFull['method'] == 'manual-ml-10') & dfFull['errorPrecision'].notnull()]
df50Cat = dfFull[(dfFull['method'] == 'manual-ml-50') & dfFull['accuracy'].notnull()]
df50Con = dfFull[(dfFull['method'] == 'manual-ml-50') & dfFull['errorPrecision'].notnull()]
df100Cat = dfFull[(dfFull['method'] == 'manual-ml-100') & dfFull['accuracy'].notnull()]
df100Con = dfFull[(dfFull['method'] == 'manual-ml-100') & dfFull['errorPrecision'].notnull()]

dfPriorCat['prec'] = dfPriorCat['accuracy']
dfPriorCon['prec'] = dfPriorCon['errorPrecision']
dfNonCat['prec'] = dfNonCat['accuracy']
dfNonCon['prec'] = dfNonCon['errorPrecision']
df10Cat['prec'] = df10Cat['accuracy']
df10Con['prec'] = df10Con['errorPrecision']
df50Cat['prec'] = df50Cat['accuracy']
df50Con['prec'] = df50Con['errorPrecision']
df100Cat['prec'] = df100Cat['accuracy']
df100Con['prec'] = df100Con['errorPrecision']

# sort category column data by accuracy
dfNonSorted = dfNonCat.sort_values(by='prec')
print("dfNonSorted")
colOrderAcc = list(dfNonSorted['target'])
print(colOrderAcc)

# sort continuous column data by precision
dfNonSorted = dfNonCon.sort_values(by='prec')
print("dfNonSorted")
colOrderErrPrec = list(dfNonSorted['target'])
print(colOrderErrPrec)

dfAllCatVal = pd.concat([dfNonCatVal, dfPriorCatVal], ignore_index=True)
dfAllCat = pd.concat([dfNonCat, dfPriorCat], ignore_index=True)
dfAllCon = pd.concat([dfNonCon, dfPriorCon], ignore_index=True)

# Try as line plots for accuracy
fig, ax = plt.subplots(2, 1)
sns.pointplot(x='target', y='prec', hue='Framework', data=dfAllCat, order=colOrderAcc, ax=ax[0])
sns.pointplot(x='target', y='prec', hue='Framework', data=dfAllCon, order=colOrderErrPrec, ax=ax[1])
plt.tight_layout()
ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)
#ax[0].set_ylim([0.25,1.05])
ax[0].set(xticklabels=[], xlabel='Categorical attributes (ordered by Non-member precision)', ylabel='Precision')
ax[1].set(xticklabels=[], xlabel='Continuous attributes (ordered by Non-member precision)', ylabel='Precision (prediction within 5%)')
plt.savefig("nonVsPriorAcc.png")
plt.close()

print(dfNonCatVal[['recall','precision','target']].head(50))
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(dfNonCatVal.describe(include='all'))
doLegend = False
if doLegend:
    plt.figure(figsize=(7,4))
else:
    plt.figure(figsize=(4,2.5))
scatter = sns.scatterplot(data=dfNonCatVal, x='recall', y='precision', hue='target', style='target', legend=doLegend)
if doLegend:
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#plt.xscale('log')
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)
plt.grid(True)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("perValue.png")
plt.close()

# Now let's do the replication graph

dfNon = pd.concat([dfNonCat[['target','prec']], dfNonCon[['target','prec']]], ignore_index=True, axis=0)
dfNon['replication'] = '0%'
df10 = pd.concat([df10Cat[['target','prec']], df10Con[['target','prec']]], ignore_index=True, axis=0)
df10['replication'] = '10%'
df50 = pd.concat([df50Cat[['target','prec']], df50Con[['target','prec']]], ignore_index=True, axis=0)
df50['replication'] = '50%'
df100 = pd.concat([df100Cat[['target','prec']], df100Con[['target','prec']]], ignore_index=True, axis=0)
df100['replication'] = '100%'

dfNonSorted = dfNon.sort_values(by='prec')
colOrder = list(dfNonSorted['target'])
print(colOrder)

dfAll = pd.concat([dfNon, df10, df50, df100], ignore_index=True, axis=0)
print(dfAll.to_string())

fig = plt.figure(figsize=(6, 3.5))
sns.pointplot(x='target', y='prec', hue='replication', data=dfAll, order=colOrder)
plt.tight_layout()
plt.grid(axis='y')
plt.xlabel('All attributes (ordered by 0% replication)')
plt.ylabel('Precision')
plt.xticks([])
plt.savefig("replication.png")
plt.close()