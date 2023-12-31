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

def doMethodTags(row):
    methodTags = {
        'manual-ml': 'Non-mem',
        'diff-anonymeter': 'Non-mem',
        'classic-anonymeter': 'Prior', 
        'manual-ml-10': 'Non-mem',
        'diff-anonymeter-10': 'Non-mem',
        'manual-ml-50': 'Non-mem',
        'diff-anonymeter-50': 'Non-mem',
        'manual-ml-100': 'Non-mem',
        'diff-anonymeter-100': 'Non-mem',
    }
    return methodTags[row['Analysis']]

def doAnalysisTags(row):
    analysisTags = {
        'manual-ml': 'ML',
        'diff-anonymeter': 'Match',
        'classic-anonymeter': 'Match', 
        'manual-ml-10': 'ML',
        'diff-anonymeter-10': 'Match',
        'manual-ml-50': 'ML',
        'diff-anonymeter-50': 'Match',
        'manual-ml-100': 'ML',
        'diff-anonymeter-100': 'Match',
    }
    return analysisTags[row['Analysis']]

def doDuplicateTags(row):
    duplicateTags = {
        'manual-ml': '0%',
        'diff-anonymeter': '0%',
        'classic-anonymeter': '0%', 
        'manual-ml-10': '10%',
        'diff-anonymeter-10': '10%',
        'manual-ml-50': '50%',
        'diff-anonymeter-50': '50%',
        'manual-ml-100': '100%',
        'diff-anonymeter-100': '100%',
    }
    return duplicateTags[row['Analysis']]

def renameMethods(df):
    replacements = {
        'manual-ml': 'ML/Non-mem',
        'diff-anonymeter': 'Match/Non-mem',
        'classic-anonymeter': 'Match/Prior', 
        'manual-ml-10': 'ML 10%',
        'diff-anonymeter-10': 'Match 10%', 
        'manual-ml-50': 'ML 50%',
        'diff-anonymeter-50': 'Match 50%', 
        'manual-ml-100': 'ML 100%',
        'diff-anonymeter-100': 'Match 100%', 
    }
    df['methodTag'] = df.apply(doMethodTags, axis=1)
    df['duplicateTag'] = df.apply(doDuplicateTags, axis=1)
    df['analysisTag'] = df.apply(doAnalysisTags, axis=1)
    return df.replace({'Analysis': replacements})

useForMl = 'manual-ml'
methods = [useForMl, 'diff-anonymeter', 'classic-anonymeter',
                     'diff-anonymeter-10', 'manual-ml-10',
                     'diff-anonymeter-50', 'manual-ml-50',
                     'diff-anonymeter-100', 'manual-ml-100']
with open('resultsMain.json', 'r') as f:
    res = json.load(f)
dataset = list(res['classic-anonymeter'].keys())[0]
print(dataset)
columns = list(res['classic-anonymeter'][dataset].keys())
pp.pprint(columns)

# The x axis is the columns
# The y axis are the accuracy / error values

# Let's separate columns into categorical (accuracy) and continuous (rmse)
catCols = []
conCols = []
for column in columns:
    if 'accuracy' in res[useForMl][dataset][column]:
        catCols.append(column)
    else:
        conCols.append(column)
print("Category columns:")
pp.pprint(catCols)
print("Continuous columns:")
pp.pprint(conCols)

# We'll put all of our plottable info in plottables
plottables = {}
for method in methods:
    plottables[method] = {}
# Ok, let's make lists for the accuracy measures:
for method in methods:
    plottables[method]['accuracy'] = []
    plottables[method]['acc-improve'] = []
    for column in catCols:
        plottables[method]['accuracy'].append(res[method][dataset][column]['accuracy'])
        plottables[method]['acc-improve'].append(res[method][dataset][column]['accuracy']/res['classic-anonymeter'][dataset][column]['accuracy'] - 1)
# And now the same for rmse
for method in methods:
    plottables[method]['rmse'] = []
    plottables[method]['avg-value'] = []
    plottables[method]['rmse_frac'] = []
    plottables[method]['rmse-improve'] = []
    plottables[method]['errorPrecision'] = []
    plottables[method]['errPrec-improve'] = []
    for column in conCols:
        plottables[method]['rmse'].append(res[method][dataset][column]['rmse'])
        plottables[method]['errorPrecision'].append(res[method][dataset][column]['errorPrecision'])
        plottables[method]['avg-value'].append(res[method][dataset][column]['avg-value'])
        plottables[method]['rmse_frac'].append(res[method][dataset][column]['rmse'] / res[method][dataset][column]['avg-value'])
        #classic = zzzz
        thisRmse = res[method][dataset][column]['rmse'] + 0.00000000001
        classicRmse = res['classic-anonymeter'][dataset][column]['rmse']
        plottables[method]['rmse-improve'].append(classicRmse/thisRmse)
        plottables[method]['errPrec-improve'].append(res[method][dataset][column]['errorPrecision']/res['classic-anonymeter'][dataset][column]['errorPrecision'] - 1)
pp.pprint(plottables)

mlIndex = methods.index(useForMl)
diffIndex = methods.index('diff-anonymeter')
classicIndex = methods.index('classic-anonymeter')
mlIndex10 = methods.index('manual-ml-10')
diffIndex10 = methods.index('diff-anonymeter-10')
mlIndex50 = methods.index('manual-ml-50')
diffIndex50 = methods.index('diff-anonymeter-50')
mlIndex100 = methods.index('manual-ml-100')
diffIndex100 = methods.index('diff-anonymeter-100')
# Plot the accuracy scores (for categorical columns)
dfAcc = pd.DataFrame({
    'Columns': doClip(catCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['accuracy'],
    methods[diffIndex]: plottables[methods[diffIndex]]['accuracy'],
    methods[classicIndex]: plottables[methods[classicIndex]]['accuracy'],
})
dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='Precision')
dfAccMelted = renameMethods(dfAccMelted)
print("dfAccMelted:")
print(dfAccMelted.head(10))

# Create the plot
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="Precision", hue='Analysis', data=dfAccMelted)
fig = snsPlot.get_figure()
fig.savefig("accuracy.png")
plt.close()

# Plot the accuracy scores (for continuous columns)
dfErrPrec = pd.DataFrame({
    'Columns': doClip(conCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['errorPrecision'],
    methods[diffIndex]: plottables[methods[diffIndex]]['errorPrecision'],
    methods[classicIndex]: plottables[methods[classicIndex]]['errorPrecision'],
})
dfErrPrecMelted = dfErrPrec.melt('Columns', var_name='Analysis', value_name='Precision')
dfErrPrecMelted = renameMethods(dfErrPrecMelted)
print("dfErrPrecMelted")
print(dfErrPrecMelted.head(15))

# Create the plot
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="Precision", hue='Analysis', data=dfErrPrecMelted)
fig = snsPlot.get_figure()
fig.savefig("errorPrecision.png")
plt.close()

catOrder = ['Education_.vel', 'Marital_St.tus', 'Income_Cat.ory', 'Attrition_.lag', 'Gender.der', 'Card_Categ.ory',]
conOrder = [
'Total_Rela.unt',
'Avg_Utiliz.tio',
'Dependent_.unt',
'Contacts_C.mon',
'Months_on_.ook',
'Customer_A.Age',
'Months_Ina.mon',
'Total_Tran._Ct',
'Total_Tran.Amt',
'Total_Amt_._Q1',
'Total_Ct_C._Q1',
'Credit_Lim.mit',
'Total_Revo.Bal',
'Avg_Open_T.Buy',
]

# Try as line plots for precision
fig, ax = plt.subplots(2, 1)
sns.pointplot(x='Columns', y='Precision', hue='Analysis', data=dfAccMelted, order=catOrder, ax=ax[0])
sns.pointplot(x='Columns', y='Precision', hue='Analysis', data=dfErrPrecMelted, order=conOrder, ax=ax[1])
plt.tight_layout()
ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)
ax[0].set_ylim([0.25,1.05])
ax[0].set(xticklabels=[], xlabel='Categorical attributes (ordered by ML/Non-mem precision)')
ax[1].set(xticklabels=[], xlabel='Continuous attributes (ordered by ML/Non-mem precision)', ylabel='Precision (prediction within 5%)')
plt.savefig("diffVsClassic.png")
plt.close()

dfAccLink = pd.DataFrame({
    'Columns': doClip(catCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['accuracy'],
    methods[mlIndex10]: plottables[methods[mlIndex10]]['accuracy'],
    methods[mlIndex50]: plottables[methods[mlIndex50]]['accuracy'],
    methods[mlIndex100]: plottables[methods[mlIndex100]]['accuracy'],
    methods[diffIndex]: plottables[methods[diffIndex]]['accuracy'],
    methods[diffIndex10]: plottables[methods[diffIndex10]]['accuracy'],
    methods[diffIndex50]: plottables[methods[diffIndex50]]['accuracy'],
    methods[diffIndex100]: plottables[methods[diffIndex100]]['accuracy'],
})
dfAccLinkMelted = dfAccLink.melt('Columns', var_name='Analysis', value_name='Precision')
dfAccLinkMelted = renameMethods(dfAccLinkMelted)
print("dfAccLinkMelted:")
print(dfAccLinkMelted.head(10))

# Try as line plots for linkage
markers = {
        'ML/Non-mem':'o',
        'Match/Non-mem':'o',
        'ML 10%':'x',
        'Match 10%':'x', 
        'ML 50%':'s',
        'Match 50%':'s', 
        'ML 100%':'^',
        'Match 100%':'^', 
}
colors = {
        'ML/Non-mem':'#4c72b0',
        'Match/Non-mem':'#ccb974',
        'ML 10%':'#4c72b0',
        'Match 10%':'#ccb974', 
        'ML 50%':'#4c72b0',
        'Match 50%':'#ccb974', 
        'ML 100%':'#4c72b0',
        'Match 100%':'#ccb974', 
}
patchHandles = [
    mpatches.Patch(color='#4c72b0', label='ML'),
    mpatches.Patch(color='#ccb974', label='Match'),
]
lineHandles = [
    mlines.Line2D([], [], color='#4c72b0', marker='o', linestyle='None', markersize=10, label='ML 0%'),
    mlines.Line2D([], [], color='#4c72b0', marker='x', linestyle='None', markersize=10, label='ML 10%'),
    mlines.Line2D([], [], color='#4c72b0', marker='s', linestyle='None', markersize=10, label='ML 50%'),
    mlines.Line2D([], [], color='#4c72b0', marker='^', linestyle='None', markersize=10, label='ML 100%'),
    mlines.Line2D([], [], color='#ccb974', marker='o', linestyle='None', markersize=10, label='Match 0%'),
    mlines.Line2D([], [], color='#ccb974', marker='x', linestyle='None', markersize=10, label='Match 10%'),
    mlines.Line2D([], [], color='#ccb974', marker='s', linestyle='None', markersize=10, label='Match 50%'),
    mlines.Line2D([], [], color='#ccb974', marker='^', linestyle='None', markersize=10, label='Match 100%'),
]
fig = plt.figure(figsize=(5, 3.5))
for method in colors.keys():
    color = colors[method]
    marker = markers[method]
    dfMethod = dfAccLinkMelted[dfAccLinkMelted['Analysis'] == method]
    sns.pointplot(x='Columns', y='Precision', color=color, markers=marker, data=dfMethod, order=catOrder)
plt.tight_layout()
plt.xticks([])
plt.xlabel("Categorical attributes (ordered by 'ML 0%' precision)")
plt.legend(handles=lineHandles)
plt.savefig("linkage.png")
plt.close()


# Plot the RMSE scores (for continuous columns)
dfAcc = pd.DataFrame({
    'Columns': doClip(conCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['rmse'],
    methods[diffIndex]: plottables[methods[diffIndex]]['rmse'],
    methods[classicIndex]: plottables[methods[classicIndex]]['rmse'],
})

dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='RMSE')

# Create the plot
plt.xticks(rotation=45)
plt.yscale('log')
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="RMSE", hue='Analysis', data=dfAccMelted)
fig = snsPlot.get_figure()
fig.savefig("rmse.png")
plt.close()

# Plot the fraction of average RMSE scores (for continuous columns)
dfAcc = pd.DataFrame({
    'Columns': doClip(conCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['rmse_frac'],
    methods[diffIndex]: plottables[methods[diffIndex]]['rmse_frac'],
    methods[classicIndex]: plottables[methods[classicIndex]]['rmse_frac'],
})

dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='Ratio of RMSE to Average Value')

# Create the plot
plt.xticks(rotation=45)
plt.yscale('log')
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="Ratio of RMSE to Average Value", hue='Analysis', data=dfAccMelted)
fig = snsPlot.get_figure()
fig.savefig("rmse-frac.png")
plt.close()

# Plot the rmse improvement over classic
dfAcc = pd.DataFrame({
    'Columns': doClip(conCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['rmse-improve'],
    methods[diffIndex]: plottables[methods[diffIndex]]['rmse-improve'],
})
dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='RMSE Improvement\nover Classic Anonymeter')
# Create the plot
plt.xticks(rotation=45)
plt.yscale('log')
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="RMSE Improvement\nover Classic Anonymeter", hue='Analysis', data=dfAccMelted)
fig = snsPlot.get_figure()
fig.savefig("rmse-improv.png")
plt.close()

# Plot the accuracy improvement over classic
dfAcc = pd.DataFrame({
    'Columns': doClip(catCols),
    methods[mlIndex]: plottables[methods[mlIndex]]['acc-improve'],
    methods[diffIndex]: plottables[methods[diffIndex]]['acc-improve'],
})
dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='Precision Improvement\nover Classic Anonymeter')
# Create the plot
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3, left=0.15)
snsPlot = sns.barplot(x="Columns", y="Precision Improvement\nover Classic Anonymeter", hue='Analysis', data=dfAccMelted)
fig = snsPlot.get_figure()
fig.savefig("acc-improv.png")
plt.close()

# Plot the accuracies for the two groups of four
dfAcc = pd.DataFrame({
    'ML 0': plottables[methods[mlIndex]]['accuracy'],
    'ML 10%': plottables[methods[mlIndex10]]['accuracy'],
    'ML 50%': plottables[methods[mlIndex50]]['accuracy'],
    'ML 100%': plottables[methods[mlIndex100]]['accuracy'],
    'Match 0': plottables[methods[diffIndex]]['accuracy'],
    'Match 10%': plottables[methods[diffIndex10]]['accuracy'],
    'Match 50%': plottables[methods[diffIndex50]]['accuracy'],
    'Match 100%': plottables[methods[diffIndex100]]['accuracy'],
})
box_plot = sns.boxplot(data=dfAcc, orient='h')
# Set the colors
colors = ['#4c72b0', '#4c72b0', '#4c72b0', '#4c72b0', '#64b5cd', '#64b5cd', '#64b5cd', '#64b5cd']
for i in range(len(colors)):
    mybox = box_plot.artists[i]
    mybox.set_facecolor(colors[i])
plt.savefig("acc-link.png")
plt.close()

'''
#4C72B0   blue
#55A868   green
#C44E52   red
#8172B2   purple
#CCB974   tan
#64B5CD
'''

# Ok, the mega plot!
dfRmseImprove = pd.DataFrame({
    'ML': plottables[methods[mlIndex]]['rmse-improve'],
    'Match': plottables[methods[diffIndex]]['rmse-improve'],
})
dfAccImprove = pd.DataFrame({
    'ML': plottables[methods[mlIndex]]['acc-improve'],
    'Match': plottables[methods[diffIndex]]['acc-improve'],
})
dfAccLink = pd.DataFrame({
    'ML 0': plottables[methods[mlIndex]]['accuracy'],
    'ML 10%': plottables[methods[mlIndex10]]['accuracy'],
    'ML 50%': plottables[methods[mlIndex50]]['accuracy'],
    'ML 100%': plottables[methods[mlIndex100]]['accuracy'],
    'Match 0': plottables[methods[diffIndex]]['accuracy'],
    'Match 10%': plottables[methods[diffIndex10]]['accuracy'],
    'Match 50%': plottables[methods[diffIndex50]]['accuracy'],
    'Match 100%': plottables[methods[diffIndex100]]['accuracy'],
})
dfRmseLink = pd.DataFrame({
    'ML 0': plottables[methods[mlIndex]]['rmse'],
    'ML 10%': plottables[methods[mlIndex10]]['rmse'],
    'ML 50%': plottables[methods[mlIndex50]]['rmse'],
    'ML 100%': plottables[methods[mlIndex100]]['rmse'],
    'Match 0': plottables[methods[diffIndex]]['rmse'],
    'Match 10%': plottables[methods[diffIndex10]]['rmse'],
    'Match 50%': plottables[methods[diffIndex50]]['rmse'],
    'Match 100%': plottables[methods[diffIndex100]]['rmse'],
})
dfErrorPrecLink = pd.DataFrame({
    'ML 0': plottables[methods[mlIndex]]['errorPrecision'],
    'ML 10%': plottables[methods[mlIndex10]]['errorPrecision'],
    'ML 50%': plottables[methods[mlIndex50]]['errorPrecision'],
    'ML 100%': plottables[methods[mlIndex100]]['errorPrecision'],
    'Match 0': plottables[methods[diffIndex]]['errorPrecision'],
    'Match 10%': plottables[methods[diffIndex10]]['errorPrecision'],
    'Match 50%': plottables[methods[diffIndex50]]['errorPrecision'],
    'Match 100%': plottables[methods[diffIndex100]]['errorPrecision'],
})

fig = plt.figure(figsize=(4, 8))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 4, 4])

# Create the seaborn graphs
ax0=fig.add_subplot(gs[0])
sns.boxplot(data=dfAccImprove, orient='h', ax=ax0)
colors = ['#4c72b0', '#ccb974']
for i in range(len(colors)):
    mybox = ax0.artists[i]
    mybox.set_facecolor(colors[i])
ax0.set_xlabel('Precision improvement over Anonymeter')

ax1=fig.add_subplot(gs[1])
sns.boxplot(data=dfRmseImprove, orient='h', ax=ax1)
for i in range(len(colors)):
    mybox = ax1.artists[i]
    mybox.set_facecolor(colors[i])
ax1.set_xlabel('RMSE improvement over Anonymeter')
ax1.set_xscale('log')

ax2=fig.add_subplot(gs[2])
sns.boxplot(data=dfAccLink, orient='h', ax=ax2)
colors = ['#4c72b0', '#4c72b0', '#4c72b0', '#4c72b0', '#ccb974', '#ccb974', '#ccb974', '#ccb974']
for i in range(len(colors)):
    mybox = ax2.artists[i]
    mybox.set_facecolor(colors[i])
ax2.set_xlabel('Precision')

ax3=fig.add_subplot(gs[3])
sns.boxplot(data=dfRmseLink, orient='h', ax=ax3)
for i in range(len(colors)):
    mybox = ax3.artists[i]
    mybox.set_facecolor(colors[i])
ax3.set_xlabel('Root Mean Squared Error (RMSE)')
ax3.set_xscale('log')

# Display the figure
#plt.subplots_adjust(bottom=0.3, left=0.15, right=0.2)
plt.tight_layout()
plt.savefig("mega.png")
plt.close()

# Now the first part of the mega plot
fig = plt.figure(figsize=(4, 3))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# Create the seaborn graphs
ax0=fig.add_subplot(gs[0])
sns.boxplot(data=dfAccImprove, orient='h', ax=ax0)
colors = ['#4c72b0', '#ccb974']
for i in range(len(colors)):
    mybox = ax0.artists[i]
    mybox.set_facecolor(colors[i])
ax0.set_xlabel('Precision improvement over Anonymeter')

ax1=fig.add_subplot(gs[1])
sns.boxplot(data=dfRmseImprove, orient='h', ax=ax1)
for i in range(len(colors)):
    mybox = ax1.artists[i]
    mybox.set_facecolor(colors[i])
ax1.set_xlabel('RMSE improvement over Anonymeter')
ax1.set_xscale('log')
plt.tight_layout()
plt.savefig("diffVsClassicImprove.png")
plt.close()

# And now the linkage part
fig = plt.figure(figsize=(4, 4.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

ax2=fig.add_subplot(gs[0])
sns.boxplot(data=dfAccLink, orient='h', ax=ax2)
colors = ['#4c72b0', '#4c72b0', '#4c72b0', '#4c72b0', '#ccb974', '#ccb974', '#ccb974', '#ccb974']
for i in range(len(colors)):
    mybox = ax2.artists[i]
    mybox.set_facecolor(colors[i])
ax2.set_xlabel('Precision (Classification)')

ax3=fig.add_subplot(gs[1])
sns.boxplot(data=dfErrorPrecLink, orient='h', ax=ax3)
for i in range(len(colors)):
    mybox = ax3.artists[i]
    mybox.set_facecolor(colors[i])
ax3.set_xlabel('Precision (Regression)')

# Display the figure
#plt.subplots_adjust(bottom=0.3, left=0.15, right=0.2)
plt.tight_layout()
plt.savefig("linkageBox.png")
plt.close()

# Let's look at three whiskers
dfRmse = pd.DataFrame({
    'ML/Non-mem': plottables[methods[mlIndex]]['rmse'],
    'Match/Non-mem': plottables[methods[diffIndex]]['rmse'],
    'Match/Prior': plottables[methods[classicIndex]]['rmse'],
})
dfAcc = pd.DataFrame({
    'ML/Non-mem': plottables[methods[mlIndex]]['accuracy'],
    'Match/Non-mem': plottables[methods[diffIndex]]['accuracy'],
    'Match/Prior': plottables[methods[classicIndex]]['accuracy'],
})
dfErrPrec = pd.DataFrame({
    'ML/Non-mem': plottables[methods[mlIndex]]['errorPrecision'],
    'Match/Non-mem': plottables[methods[diffIndex]]['errorPrecision'],
    'Match/Prior': plottables[methods[classicIndex]]['errorPrecision'],
})
print("dfRmse:")
print(dfRmse.describe())
print(dfRmse.head())
print("dfAcc:")
print(dfAcc.describe())
print(dfAcc.head())
print("dfErrPrec:")
print(dfErrPrec.describe())
print(dfErrPrec.head())

fig = plt.figure(figsize=(4, 4.5))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
ax0=fig.add_subplot(gs[0])
sns.boxplot(data=dfAcc, orient='h', ax=ax0)
colors = ['#4c72b0', '#ccb974', '#c44e52']
for i in range(len(colors)):
    mybox = ax0.artists[i]
    mybox.set_facecolor(colors[i])
ax0.set_xlabel('Precision (Classification)')

ax1=fig.add_subplot(gs[1])
sns.boxplot(data=dfRmse, orient='h', ax=ax1)
for i in range(len(colors)):
    mybox = ax1.artists[i]
    mybox.set_facecolor(colors[i])
ax1.set_xlabel('Root Mean Square Error (RMSE)')
ax1.set_xscale('log')

ax2=fig.add_subplot(gs[2])
sns.boxplot(data=dfErrPrec, orient='h', ax=ax2)
for i in range(len(colors)):
    mybox = ax2.artists[i]
    mybox.set_facecolor(colors[i])
ax2.set_xlabel('Precision (Regression)')

plt.tight_layout()
plt.savefig("diffVsClassicAll.png")
plt.close()

fig = plt.figure(figsize=(4, 3))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
ax0=fig.add_subplot(gs[0])
sns.boxplot(data=dfAcc, orient='h', ax=ax0)
colors = ['#4c72b0', '#ccb974', '#c44e52']
for i in range(len(colors)):
    mybox = ax0.artists[i]
    mybox.set_facecolor(colors[i])
ax0.set_xlabel('Precision (Classification)')

ax1=fig.add_subplot(gs[1])
sns.boxplot(data=dfErrPrec, orient='h', ax=ax1)
for i in range(len(colors)):
    mybox = ax1.artists[i]
    mybox.set_facecolor(colors[i])
ax1.set_xlabel('Precision (Regression)')

plt.tight_layout()
plt.savefig("diffVsClassicBox.png")
plt.close()