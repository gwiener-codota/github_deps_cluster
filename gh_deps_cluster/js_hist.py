import argparse
import dataclasses
import logging
import re

import pandas as pd

import data


logging.basicConfig(level=logging.DEBUG)


@dataclasses.dataclass
class Args:
    num_topics: int
    min_stars: int
    sample_size: int


topic_pat = re.compile(r'@?(\w+)/.*')


def dep_to_topic(s: str) -> str:
    if m := topic_pat.fullmatch(s):
        return m.group(1)
    return s


def dist_diff(u, v):
    return (u - v).abs().sum() / len(u)


def main():
    logger = logging.getLogger(__name__)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num-topics', type=int, default=100)
    arg_parser.add_argument('--min-stars', type=int, default=25)
    arg_parser.add_argument('--sample-size', type=int, default=7000)
    args = Args(**vars(arg_parser.parse_args()))
    logger.debug(args)
    repo_deps = pd.read_csv(data.get('javascript', 'package-json.csv'))
    logger.debug(f'{len(repo_deps)} JS repositories with dependencies')
    repo_info = pd.read_csv(data.get('javascript', 'js_repos.csv'))
    logger.debug(f'{len(repo_info)} JS repositories with metadata')
    repo_deps_arr = repo_deps.set_index('repo_name').dependencies.str.split('|')
    topics_dup = repo_deps_arr.explode().fillna('').apply(dep_to_topic)
    topics = topics_dup.reset_index().drop_duplicates().set_index('repo_name').dependencies
    topic_counts = topics.value_counts()
    top_topics_counts = topic_counts[:args.num_topics]
    top_topic_names = top_topics_counts.index.values
    top_topics = topics[topics.isin(top_topic_names)]
    orig_dist = top_topics_counts / len(repo_deps)
    sample_repos = repo_deps.sample(n=args.sample_size).repo_name
    logger.debug(f'sampled {len(sample_repos)} repos')
    samp_topics = top_topics[top_topics.index.isin(sample_repos)]
    samp_dist = samp_topics.value_counts().reindex(top_topic_names) / len(sample_repos)
    logger.info(f'orig --> sample {dist_diff(orig_dist, samp_dist)}')
    repos_with_min_stars = repo_info[repo_info.stars >= args.min_stars].repo_name
    logger.debug(f'{len(repos_with_min_stars)} repos with min. {args.min_stars} stars')
    topics_with_min_stars = top_topics[top_topics.index.isin(repos_with_min_stars)]
    min_start_dist = topics_with_min_stars.value_counts().reindex(top_topic_names) / len(repos_with_min_stars)
    logger.info(f'orig --> min. stars {dist_diff(orig_dist, min_start_dist)}')
    sample_repos_min_stars = repos_with_min_stars.sample(n=args.sample_size)
    logger.debug(f'sampled {len(sample_repos_min_stars)} repos with min. {args.min_stars} stars')
    samp_min_stars_topics = top_topics[top_topics.index.isin(sample_repos_min_stars)]
    samp_min_stars_dist = samp_min_stars_topics.value_counts().reindex(top_topic_names) / len(sample_repos_min_stars)
    logger.info(f'min. stars --> sample min. stars {dist_diff(min_start_dist, samp_min_stars_dist)}')
    logger.info(f'orig --> sample min. stars {dist_diff(orig_dist, samp_min_stars_dist)}')
    # Not doing stratified sampling -
    # scikit-learn StratifiedShuffleSplit does multiclass sampling by treating every combination of labels
    # from the data as a single pseudo-class, which is not practical for 50+ labels


if __name__ == '__main__':
    main()
