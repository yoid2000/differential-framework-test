import random
import kAnonymity1
import pandas as pd
import os

def countHomogeneous(df):
    return None

def makeRow(ncol, nval, nsen, id):
    row = {}
    key = ''
    for i in range(ncol):
        val = str(random.randint(0,nval-1))
        row[f'c{i}'] = val
        key += val + '_'
    sensitive = str(random.randint(0,nsen-1))
    row['sensitive'] = sensitive
    row['id'] = str(id)
    return row, key

def makeKAnon(df, categories, feature_columns, sensitive_column, k=5):
    ka = kAnonymity1.kanon(categories, k=k)
    full_spans = ka.get_spans(df, df.index)
    print(full_spans)
    finished_partitions = ka.partition_dataset(df, feature_columns, sensitive_column, full_spans, ka.is_k_anonymous)
    dfn = ka.build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)
    return dfn

distinct = {}
table = []
test_ncol = 17
test_nval = 2
test_nsen = 3
test_ndistinct = 60000
k = 5
id = 0
origFileName = f"df.c{test_ncol}.v{test_nval}.pkl"
anonFileName = f"dfAnon.c{test_ncol}.v{test_nval}.k{k}.pkl"
csvFileName = f"df.c{test_ncol}.v{test_nval}.csv"
if os.path.exists(origFileName):
    df = pd.read_pickle(origFileName)
else:
    while len(distinct) < test_ndistinct:
        row, key = makeRow(test_ncol, test_nval, test_nsen, id)
        distinct_key = '_'.join(row)
        distinct[key] = 1
        table.append(row)
        id += 1
    print(f"distinct len {len(distinct)}, table len {len(table)}")
    df = pd.DataFrame(table)
    df.to_pickle(origFileName)
print(df.head())
dfAnon = df.drop('id', axis=1)
dfAnon.to_csv(csvFileName, index=False)
feature_columns = list(dfAnon.columns)
feature_columns.remove('sensitive')
sensitive_column = 'sensitive'
print(f"feature_columns {feature_columns}")
if os.path.exists(anonFileName):
    dfn = pd.read_pickle(anonFileName)
else:
    dfn = makeKAnon(dfAnon, list(dfAnon.columns), feature_columns, sensitive_column, k=k)
    dfn.to_pickle(anonFileName)

print(dfn.head(50))
dfSorted = dfn.sort_values(feature_columns+[sensitive_column])
print(dfSorted.head(50))


numHomogeneous = countHomogeneous(dfn)
