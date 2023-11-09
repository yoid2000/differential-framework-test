import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

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
    with open('resultsRecall.json', 'r') as f:
        res = json.load(f)

    df = pd.DataFrame(res)
    print(df.head())

    doLegend = False
    if doLegend:
        plt.figure(figsize=(7,4))
    else:
        plt.figure(figsize=(4,2.5))
    scatter = sns.scatterplot(data=df, x='recall', y='prec', hue='target', style='target', legend=doLegend)
    if doLegend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    #plt.xscale('log')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig("prec-recall.png")
    plt.close()

