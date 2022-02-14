import data
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

graph_df: pd.DataFrame = pd.read_csv(data.get('python', 'py_deps_graph_xs.csv')).rename(columns={'ratio': 'weight'})
graph = nx.from_pandas_edgelist(graph_df, source='a', target='b', edge_attr='weight').to_undirected()
print(graph)

plt.figure(num=None, figsize=(40, 40))
nx.draw_spring(graph)
plt.show()
