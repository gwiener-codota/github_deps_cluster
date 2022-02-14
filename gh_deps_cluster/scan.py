import data
import pandas as pd
import typing
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


def hist(x: pd.Series):
    n = len(x)
    cnt = x.explode().value_counts() / n
    return cnt.to_dict()


py_reqs: pd.DataFrame = pd.read_csv(data.get('python', 'py_reqs.csv'))
reqs = py_reqs['requirements'].str.split('|')
vectorize = TfidfVectorizer(
    lowercase=False,
    tokenizer=lambda x: x,
    token_pattern=r'[A-Za-z][A-Za-z0-9_\-]*',
    min_df=10,
    max_df=0.85,
    binary=True
)
X = vectorize.fit_transform(reqs)
print(X.shape)
scan = DBSCAN(metric='cosine', eps=0.3, n_jobs=8)
y = scan.fit_predict(X)
labels = pd.Series(y, name='label')
results = pd.DataFrame(dict(label=labels, reqs=reqs))
clusters = results.groupby('label')
print(clusters.size())
histograms = clusters['reqs'].agg(hist)
print(histograms)
