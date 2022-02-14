import itertools
import re
import typing
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import cm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import data

topic_pat = re.compile(r'@?([\w_\-]+)')


def deps_str_to_topics_set(s: str) -> typing.Set[str]:
    arr = s.split('|')
    ret = set()
    for dep in arr:
        if m := topic_pat.match(dep):
            ret.add(m.group(1))
    return ret


dep_strs: pd.Series = pd.read_csv(
    data.get('javascript', 'package-json.csv'),
    index_col='repo_name',
    usecols=['repo_name', 'dependencies'],
    squeeze=True
).fillna('')
num_repos = len(dep_strs)
topic_sets: pd.Series = dep_strs.apply(deps_str_to_topics_set)
topics: pd.Series = topic_sets.explode().rename('topic')
topic_counts: pd.Series = topics.value_counts()
topic_ratios: pd.Series = topic_counts / num_repos
topic_stats = pd.DataFrame(dict(num_repos=topic_counts, ratio=topic_ratios))
major_topics_idx = topic_stats.ratio >= 0.015
major_topics = topic_stats[major_topics_idx]
filtered_topics: pd.Series = pd.merge(topics, major_topics, left_on='topic', right_index=True)['topic']
repo_topics = filtered_topics.groupby(filtered_topics.index)
pair_count = Counter()
for repo, my_topics in repo_topics:
    pair_count.update([pair for pair in itertools.product(my_topics, repeat=2) if pair[0] < pair[1]])
pairs, num_pairs = zip(*pair_count.most_common())
a, b = zip(*pairs)
graph_df = pd.DataFrame(dict(a=a, b=b, num_pairs=num_pairs))
graph_df['weight'] = graph_df['num_pairs'] / num_repos
print(graph_df.weight.describe())
graph_inv = graph_df.rename(columns={'a': 'b', 'b': 'a'})
graph_sim = pd.concat((graph_df, graph_inv))
X = graph_sim.pivot(index='a', columns='b', values='weight').fillna(0.0)
spectra = SpectralClustering(n_clusters=8, affinity='precomputed')
labels = spectra.fit_predict(X)
results = pd.DataFrame(dict(dep=X.index, label=labels))
clusters: pd.Series = results.groupby('label')['dep'].apply(list)
for i, c in clusters.iteritems():
    print(i, *c, sep=', ')
graph_df_sm = graph_df[graph_df.weight >= 0.012]
graph = nx.from_pandas_edgelist(graph_df_sm, source='a', target='b', edge_attr='weight').to_undirected()
print(graph)

weight_scale = MinMaxScaler((1, 4))
edge_width = weight_scale.fit_transform(graph_df_sm['weight'].to_numpy().reshape(-1, 1)).reshape(-1).tolist()
plt.figure(num=None, figsize=(40, 40))
nx.draw_spring(graph, with_labels=True, edge_color='lightgray', width=edge_width)
plt.show()
