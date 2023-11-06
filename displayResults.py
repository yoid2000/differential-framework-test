import numpy as np
import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

pp = pprint.PrettyPrinter(indent=4)

def doClip(thingy,clipBegin = 10, clipEnd=3):
    clipped = []
    for thing in thingy:
        clip = thing[:clipBegin] + '.' + thing[-clipEnd:]
        while clip in clipped:
            clip += '_'
        clipped.append(clip)
    return clipped

if __name__ == "__main__":
    useForMl = 'manual-ml'
    methods = [useForMl, 'diff-anonymeter', 'classic-anonymeter']
    with open('results.json', 'r') as f:
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
        for column in conCols:
            plottables[method]['rmse'].append(res[method][dataset][column]['rmse'])
            plottables[method]['avg-value'].append(res[method][dataset][column]['avg-value'])
            plottables[method]['rmse_frac'].append(res[method][dataset][column]['rmse'] / res[method][dataset][column]['avg-value'])
            #classic = zzzz
            thisRmse = res[method][dataset][column]['rmse']
            classicRmse = res['classic-anonymeter'][dataset][column]['rmse']
            plottables[method]['rmse-improve'].append(classicRmse/thisRmse)
    pp.pprint(plottables)

    methods = ['manual-ml', 'diff-anonymeter', 'classic-anonymeter']
    mlIndex = methods.index(useForMl)
    diffIndex = methods.index('diff-anonymeter')
    classicIndex = methods.index('classic-anonymeter')
    # Plot the accuracy scores (for categorical columns)
    dfAcc = pd.DataFrame({
        'Columns': doClip(catCols),
        methods[mlIndex]: plottables[methods[mlIndex]]['accuracy'],
        methods[diffIndex]: plottables[methods[diffIndex]]['accuracy'],
        methods[classicIndex]: plottables[methods[classicIndex]]['accuracy'],
    })

    dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='Accuracy')

    # Create the plot
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    snsPlot = sns.barplot(x="Columns", y="Accuracy", hue='Analysis', data=dfAccMelted)
    fig = snsPlot.get_figure()
    fig.savefig("accuracy.png")
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
    dfAccMelted = dfAcc.melt('Columns', var_name='Analysis', value_name='Accuracy Improvement\nover Classic Anonymeter')
    # Create the plot
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    snsPlot = sns.barplot(x="Columns", y="Accuracy Improvement\nover Classic Anonymeter", hue='Analysis', data=dfAccMelted)
    fig = snsPlot.get_figure()
    fig.savefig("acc-improv.png")
    plt.close()
