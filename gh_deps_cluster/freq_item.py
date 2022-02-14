import data
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from tabulate import tabulate


py_reqs: pd.DataFrame = pd.read_csv(data.get('python', 'py_reqs.csv'))
reqs_sets = py_reqs['requirements'].str.split('|')
te = TransactionEncoder()
te_ary = te.fit_transform(reqs_sets)
df = pd.DataFrame(te_ary, columns=te.columns_)
res = apriori(df, min_support=0.02, use_colnames=True)
res['itemsets'] = res['itemsets'].apply(lambda s: ', '.join(sorted(s)))
print(tabulate(res))
