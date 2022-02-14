import pandas as pd
import os

my_dir = os.path.dirname(__file__)
data_dir = os.path.join(my_dir, '..', 'data')
py_dir = os.path.join(data_dir, 'python')
num_repos_path = os.path.join(py_dir, 'num_repos_with_reqs.txt')
py_deps_counts_path = os.path.join(py_dir, 'py_deps_counts.csv')
py_deps_graph_path = os.path.join(py_dir, 'py_deps_graph.csv')
with open(num_repos_path) as f:
    num_repos = int(f.read().strip())
py_deps_counts: pd.DataFrame = pd.read_csv(py_deps_counts_path)
deps_with_min_count: pd.Series = py_deps_counts['dep'][py_deps_counts.num_repos >= 200]
py_deps_graph: pd.DataFrame = pd.read_csv(py_deps_graph_path)
py_deps_graph['ratio'] = py_deps_graph['num_pairs'] / num_repos
print(len(py_deps_graph))
a_has_min_count_idx = py_deps_graph['a'].isin(deps_with_min_count)
b_has_min_count_idx = py_deps_graph['b'].isin(deps_with_min_count)
edge_has_min_count_idx = py_deps_graph['num_pairs'] >= 300
both_has_min_count_idx = a_has_min_count_idx & b_has_min_count_idx
has_min_count_idx = both_has_min_count_idx & edge_has_min_count_idx
print(sum(has_min_count_idx))
py_deps_graph_filtered = py_deps_graph[has_min_count_idx]
out_path = os.path.join(py_dir, 'py_deps_graph_xs.csv')
py_deps_graph_filtered.to_csv(out_path, index=False)
