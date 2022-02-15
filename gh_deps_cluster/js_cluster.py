import data
import itertools
import numpy as np
import pandas as pd
from gh_deps_cluster.pair_share import pair_share
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df = pd.read_csv(
    data.get('javascript', 'package-json.csv'),
    index_col='repo_name',
    usecols=['repo_name', 'dependencies']
).fillna('')
vectorize = TfidfVectorizer(
    tokenizer=lambda x: x.split('|'),
    token_pattern=r'^@?(\w+)',
    min_df=0.01,
    max_df=0.8,
    binary=True,
    use_idf=False,
    norm='l1'
)
X = vectorize.fit_transform(df.dependencies)
rev_idx = vectorize.get_feature_names_out()
print(X.shape)

# scan = DBSCAN(metric='cosine', eps=0.33, n_jobs=32)
scan = KMeans(n_clusters=32)
scan.fit(X)
threshold = np.quantile(scan.cluster_centers_, .9)
indices = np.argwhere(scan.cluster_centers_ >= threshold)
for k, g in itertools.groupby(indices, lambda x: x[0]):
    names = [rev_idx[x[1]] for x in g]
    print(k, names)
