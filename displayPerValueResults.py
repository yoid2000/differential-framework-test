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
dfLow = dfFull[(dfFull['method'] == 'manual-ml') & (dfFull['numPredCols'] < 18)]
dfLow['Framework'] = 'Non-member'

dfNonCatVal = dfMain[(dfMain['column'].notnull())]
dfNonCat = dfMain[(dfMain['accuracy'].notnull())]
dfNonCon = dfMain[(dfMain['errorPrecision'].notnull())]
dfNonCatLow = dfLow[(dfLow['accuracy'].notnull())]
dfNonConLow = dfLow[(dfLow['errorPrecision'].notnull())]
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
dfNonCatLow['prec'] = dfNonCatLow['accuracy']
dfNonConLow['prec'] = dfNonConLow['errorPrecision']
df10Cat['prec'] = df10Cat['accuracy']
df10Con['prec'] = df10Con['errorPrecision']
df50Cat['prec'] = df50Cat['accuracy']
df50Con['prec'] = df50Con['errorPrecision']
df100Cat['prec'] = df100Cat['accuracy']
df100Con['prec'] = df100Con['errorPrecision']

# sort category column data by accuracy
dfNonCat = dfNonCat[['target','prec','Framework']]
dfNonCat = dfNonCat.groupby(['target','Framework'], as_index=False).mean()
dfNonCon = dfNonCon[['target','prec','Framework']]
dfNonCon = dfNonCon.groupby(['target','Framework'], as_index=False).mean()
dfNonCatLow = dfNonCatLow[['target','prec','Framework']]
dfNonCatLow = dfNonCatLow.groupby(['target','Framework'], as_index=False).mean()
dfNonConLow = dfNonConLow[['target','prec','Framework']]
dfNonConLow = dfNonConLow.groupby(['target','Framework'], as_index=False).mean()
dfPriorCat = dfPriorCat[['target','prec','Framework']]
dfPriorCat = dfPriorCat.groupby(['target','Framework'], as_index=False).mean()
dfPriorCon = dfPriorCon[['target','prec','Framework']]
dfPriorCon = dfPriorCon.groupby(['target','Framework'], as_index=False).mean()
dfNonSorted = dfNonCat.sort_values(by='prec')
print("dfNonSorted")
colOrderAcc = list(dfNonSorted['target'])
print(colOrderAcc)

# sort continuous column data by precision
dfNonSorted = dfNonCon.sort_values(by='prec')
print("dfNonSorted")
colOrderErrPrec = list(dfNonSorted['target'])
print(colOrderErrPrec)

dfAllCat = pd.concat([dfNonCat, dfPriorCat], ignore_index=True)
dfAllCon = pd.concat([dfNonCon, dfPriorCon], ignore_index=True)

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

dfNonCat['Known Attr'] = 'All'
dfNonCon['Known Attr'] = 'All'
dfNonCatLow['Known Attr'] = 'PII only'
dfNonConLow['Known Attr'] = 'PII only'
dfAllCatKnown = pd.concat([dfNonCat, dfNonCatLow], ignore_index=True)
dfAllConKnown = pd.concat([dfNonCon, dfNonConLow], ignore_index=True)

fig, ax = plt.subplots(2, 1)
sns.pointplot(x='target', y='prec', hue='Known Attr', data=dfAllCatKnown, order=colOrderAcc, ax=ax[0])
sns.pointplot(x='target', y='prec', hue='Known Attr', data=dfAllConKnown, order=colOrderErrPrec, ax=ax[1])
plt.tight_layout()
ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)
#ax[0].set_ylim([0.25,1.05])
ax[0].set(xticklabels=[], xlabel='Categorical attributes (ordered by Non-member precision)', ylabel='Precision')
ax[1].set(xticklabels=[], xlabel='Continuous attributes (ordered by Non-member precision)', ylabel='Precision (prediction within 5%)')
plt.savefig("allVsPiiAcc.png")
plt.close()

dfNonCatVal = dfNonCatVal[['target','precision','recall','label']]
dfNonCatVal = dfNonCatVal.groupby(['target', 'label'], as_index=False).mean()
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
df10Cat = df10Cat[['target','prec']]
df10Cat = df10Cat.groupby(['target'], as_index=False).mean()
df10Con = df10Con[['target','prec']]
df10Con = df10Con.groupby(['target'], as_index=False).mean()
df50Cat = df50Cat[['target','prec']]
df50Cat = df50Cat.groupby(['target'], as_index=False).mean()
df50Con = df50Con[['target','prec']]
df50Con = df50Con.groupby(['target'], as_index=False).mean()
df100Cat = df100Cat[['target','prec']]
df100Cat = df100Cat.groupby(['target'], as_index=False).mean()
df100Con = df100Con[['target','prec']]
df100Con = df100Con.groupby(['target'], as_index=False).mean()

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