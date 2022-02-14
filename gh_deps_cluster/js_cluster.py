import data
import itertools
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
print(X.shape)

# scan = DBSCAN(metric='cosine', eps=0.33, n_jobs=32)
scan = KMeans(n_clusters=16)
Y = scan.fit_predict(X)
nz_0, nz_1 = scan.cluster_centers_.nonzero()
rev_idx = vectorize.get_feature_names_out()
for k, g in itertools.groupby(zip(nz_0, nz_1), key=lambda x: x[0]):
    names = map(lambda t: rev_idx[t[1]], g)
    print(k, *names)
