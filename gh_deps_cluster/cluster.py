import os
import pandas as pd
from sklearn.cluster import SpectralClustering
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', None)

my_dir = os.path.dirname(__file__)
data_dir = os.path.join(my_dir, '..', 'data')
py_dir = os.path.join(data_dir, 'python')
py_deps_graph_path = os.path.join(py_dir, 'py_deps_graph_sm.csv')
py_deps_graph: pd.DataFrame = pd.read_csv(py_deps_graph_path)
py_deps_graph_inv = pd.DataFrame(dict(
    a=py_deps_graph['b'],
    b=py_deps_graph['a'],
    num_pairs=py_deps_graph['num_pairs'],
    ratio=py_deps_graph['ratio']
))
py_deps_graph_sim = pd.concat((py_deps_graph, py_deps_graph_inv))
x = py_deps_graph_sim.pivot(index='a', columns='b', values='ratio').fillna(0.0)

spectra = SpectralClustering(n_clusters=48, affinity='precomputed')
labels = spectra.fit_predict(x)
results = pd.DataFrame(dict(dep=x.index, label=labels))
clusters: pd.Series = results.groupby('label')['dep'].apply(list)
for i, c in clusters.iteritems():
    print(i, *c, sep=', ')
