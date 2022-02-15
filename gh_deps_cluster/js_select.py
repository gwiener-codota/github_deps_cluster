import numpy as np
import pandas as pd
import data

wanted = ['react', 'redux']
pattern = '|'.join(map(lambda s: f'.*{s}.*', wanted))


repos: pd.DataFrame = pd.read_csv(
    data.get('javascript', 'github_repositories.csv'),
    header=None,
    index_col='repo_name',
    names=['repo_name', 'stars', 'language', 'license']
)
dep_str: pd.Series = pd.read_csv(
    data.get('javascript', 'package-json.csv'),
    index_col='repo_name',
    usecols=['repo_name', 'dependencies'],
    squeeze=True
).fillna('')
dep_arr: pd.Series = dep_str.apply(lambda x: x.split('|'))
deps = dep_arr.explode()
selected = deps[deps.str.contains(pattern, case=False)]
repo_num_deps = selected.groupby(selected.index).count().sort_values(ascending=False)
boost = np.log10(repo_num_deps).rename('boost')
candidates = pd.merge(repos, boost, left_index=True, right_index=True)
candidates['score'] = candidates['stars'] * candidates['boost']
candidates.sort_values(by='score', inplace=True, ascending=False)
out = candidates.drop(columns=['boost', 'score']).head(7000)
out.to_csv(data.get('out', 'react.csv'), index_label='repo_name')
