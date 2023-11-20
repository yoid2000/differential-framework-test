import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
import math
import os
from diffTools import doClip

pp = pprint.PrettyPrinter(indent=4)

with open('resultsPerValue.json', 'r') as f:
    resFull = json.load(f)

resNon = resFull['manual-ml']['BankChurnersNoId_ctgan.json']
resPrior = resFull['prior']['BankChurnersNoId_ctgan.json']
res10 = resFull['manual-ml-10']['BankChurnersNoId_ctgan.json']
res50 = resFull['manual-ml-50']['BankChurnersNoId_ctgan.json']
res100 = resFull['manual-ml-100']['BankChurnersNoId_ctgan.json']

print("Gather categorical column/value statistics")
catNonData = []
catPriorData = []
for column, stuff in resNon.items():
    if 'perValue' not in stuff:
        continue
    for value, measures in stuff['perValue'].items():
        attClass = column + '_' + value
        catNonData.append({'class':attClass,
                     'column':column,
                     'value':value,
                     'Framework':'Non-Member',
                     'F1 score':measures['f1'],
                     'precision':measures['precision'],
                     'recall':measures['recall']})
for column, stuff in resPrior.items():
    if 'perValue' not in stuff:
        continue
    for value, measures in stuff['perValue'].items():
        attClass = column + '_' + value
        catPriorData.append({'class':attClass,
                     'column':column,
                     'value':value,
                     'Framework':'Prior',
                     'F1 score':measures['f1'],
                     'precision':measures['precision'],
                     'recall':measures['recall']})
dfNonCatVal = pd.DataFrame(catNonData)
print(dfNonCatVal.head(10))
dfPriorCatVal = pd.DataFrame(catPriorData)
print(dfPriorCatVal.head(10))

def addToData(res, catData, conData, framework='Non-member'):
    for column, stuff in res.items():
        if 'perValue' not in stuff:
            conData.append({'column':column,
                            'Framework':framework,
                            'avg-value':stuff['avg-value'],
                            'rmse':stuff['rmse'],
                            'precision':stuff['errorPrecision']})
        else:
            catData.append({'column':column,
                            'Framework':framework,
                            'precision':stuff['accuracy'],
                            'accuracy_freq':stuff['accuracy_freq']})
    return pd.DataFrame(catData), pd.DataFrame(conData)

print("Gather column statistics")
catNonData = []
catPriorData = []
conNonData = []
conPriorData = []
cat10Data = []
con10Data = []
cat50Data = []
con50Data = []
cat100Data = []
con100Data = []
dfNonCat, dfNonCon = addToData(resNon, catNonData, conNonData)
dfPriorCat, dfPriorCon = addToData(resPrior, catPriorData, conPriorData, framework='Prior')
df10Cat, df10Con = addToData(res10, cat10Data, con10Data)
df50Cat, df50Con = addToData(res50, cat50Data, con50Data)
df100Cat, df100Con = addToData(res100, cat100Data, con100Data)
print(dfNonCat.head(10))
print(dfPriorCat.head(10))
print(dfNonCon.head(10))
print(dfPriorCon.head(10))
print(df10Cat.head(10))
print(df10Con.head(10))

# sort both of these by f1 value of catNonData
dfNonSorted = dfNonCatVal.sort_values(by='F1 score')
print("dfNonSorted")
print(dfNonSorted[['class','F1 score']].head(15))
classOrderF1 = list(dfNonSorted['class'])
print(classOrderF1)

# sort category column data by accuracy
dfNonSorted = dfNonCat.sort_values(by='precision')
print("dfNonSorted")
colOrderAcc = list(dfNonSorted['column'])
print(colOrderAcc)

# sort continuous column data by precision
dfNonSorted = dfNonCon.sort_values(by='precision')
print("dfNonSorted")
colOrderErrPrec = list(dfNonSorted['column'])
print(colOrderErrPrec)

dfAllCatVal = pd.concat([dfNonCatVal, dfPriorCatVal], ignore_index=True)
print(dfAllCatVal[['class','Framework', 'F1 score']].head(50))

dfAllCat = pd.concat([dfNonCat, dfPriorCat], ignore_index=True)
print(dfAllCat[['column','precision', 'accuracy_freq']].head(50))

dfAllCon = pd.concat([dfNonCon, dfPriorCon], ignore_index=True)
print(dfAllCon[['column','precision', 'rmse', 'avg-value']].head(50))

# Try as line plots for accuracy
fig, ax = plt.subplots(2, 1)
sns.pointplot(x='column', y='precision', hue='Framework', data=dfAllCat, order=colOrderAcc, ax=ax[0])
sns.pointplot(x='column', y='precision', hue='Framework', data=dfAllCon, order=colOrderErrPrec, ax=ax[1])
plt.tight_layout()
ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)
#ax[0].set_ylim([0.25,1.05])
ax[0].set(xticklabels=[], xlabel='Categorical attributes (ordered by Non-member precision)', ylabel='Precision')
ax[1].set(xticklabels=[], xlabel='Continuous attributes (ordered by Non-member precision)', ylabel='Precision (prediction within 5%)')
plt.savefig("nonVsPriorAcc.png")
plt.close()

# Try as line plots for F1 Score
fig, ax = plt.subplots(2, 1)
sns.pointplot(x='class', y='F1 score', hue='Framework', data=dfAllCatVal, order=classOrderF1, ax=ax[0])
sns.pointplot(x='column', y='precision', hue='Framework', data=dfAllCon, order=colOrderErrPrec, ax=ax[1])
plt.tight_layout()
ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)
#ax[0].set_ylim([0.25,1.05])
ax[0].set(xticklabels=[], xlabel='Categorical attributes/values (ordered by Non-member F1 score)')
ax[1].set(xticklabels=[], xlabel='Continuous attributes (ordered by ML/Non-mem precision)', ylabel='Precision (prediction within 5%)')
plt.savefig("nonVsPriorF1.png")
plt.close()

doLegend = False
if doLegend:
    plt.figure(figsize=(7,4))
else:
    plt.figure(figsize=(4,2.5))
scatter = sns.scatterplot(data=dfNonCatVal, x='recall', y='precision', hue='column', style='column', legend=doLegend)
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

dfNon = pd.concat([dfNonCat[['column','precision']], dfNonCon[['column','precision']]], ignore_index=True, axis=0)
dfNon['replication'] = '0%'
df10 = pd.concat([df10Cat[['column','precision']], df10Con[['column','precision']]], ignore_index=True, axis=0)
df10['replication'] = '10%'
df50 = pd.concat([df50Cat[['column','precision']], df50Con[['column','precision']]], ignore_index=True, axis=0)
df50['replication'] = '50%'
df100 = pd.concat([df100Cat[['column','precision']], df100Con[['column','precision']]], ignore_index=True, axis=0)
df100['replication'] = '100%'

dfNonSorted = dfNon.sort_values(by='precision')
colOrder = list(dfNonSorted['column'])
print(colOrder)

dfAll = pd.concat([dfNon, df10, df50, df100], ignore_index=True, axis=0)
print(dfAll.to_string())

fig = plt.figure(figsize=(6, 3.5))
sns.pointplot(x='column', y='precision', hue='replication', data=dfAll, order=colOrder)
plt.tight_layout()
plt.grid(axis='y')
plt.xlabel('All attributes (ordered by 0% replication)')
plt.ylabel('Precision')
plt.xticks([])
plt.savefig("replication.png")
plt.close()