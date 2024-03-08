import typing
from functools import partial

import numpy as np

import library


def get_normalized_total_orderings(baseline: typing.List, other: typing.List) -> typing.List:
    merged_list = [[i, baseline[i], other[i], np.random.uniform()] for i in range(len(baseline))]
    normalized_baseline = [x[0] for x in sorted(merged_list, reverse=True, key=lambda x: (x[1], x[2], x[3]))]
    normalized_other = [x[0] for x in sorted(merged_list, reverse=True, key=lambda x: (x[2], x[1], x[3]))]
    return [normalized_baseline, normalized_other]


def compare_centrality(baseline: typing.List, ranking_other: typing.List) -> typing.Dict:
    norm_baseline, norm_other = get_normalized_total_orderings(baseline, ranking_other)
    overlap_list = []
    for i in range(0, len(baseline)):
        overlap_list.append(len(list(set(norm_other[:i + 1]) & set(norm_baseline[:i + 1]))))

    return {'overlap': overlap_list}


def plot_overlap(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, zoom: bool = True,
                 note: typing.AnyStr = None) -> None:
    library.plot_general(results, baseline=baseline, centrality=centrality, header="", zoom=zoom,
                         parabola=True, note=note)
    return


def get_ranking(centrality: typing.AnyStr):
    lookup = {
        "betweenness": library.get_ranking_betweenness,
        "closeness": library.get_ranking_closeness,
        "harmonic": library.get_ranking_harmonic,
        "pagerank": library.get_ranking_pagerank,
        "degree": library.get_ranking_degree,
        "load": library.get_ranking_load,
        "katz": library.get_ranking_katz,
        "eigenvector": library.get_ranking_eigenvector
    }
    for damping_factor in range(0,100):
        lookup.update({f"pagerank-{damping_factor}": partial(library.get_ranking_pagerank, alpha=damping_factor/100)})

    return lookup[centrality]


def run_comparing_sim(runs: int, centrality1: typing.AnyStr, centrality2: typing.AnyStr, configuration: typing.NamedTuple,
                      note: typing.AnyStr = None, parallel: bool = True) -> None:
    def simulate_one_scenario(in_centrality1: typing.AnyStr, in_centrality2: typing.AnyStr, in_configuration: typing.NamedTuple):
        graph = in_configuration.generator()
        return compare_centrality(get_ranking(in_centrality1)(graph), get_ranking(in_centrality2)(graph))

    print(f'Running: {centrality1} vs. {centrality2}, conffile: {configuration.name}')
    if not parallel:
        sim_results = []
        for i in range(runs):
            sim_results.append(simulate_one_scenario(centrality1, centrality2, configuration))
    else:
        sim_results = library.run_in_parallel(runs, partial(simulate_one_scenario, centrality1, centrality2, configuration))
    results = {'overlap': [res['overlap'] for res in sim_results]}
    plot_overlap(results['overlap'], baseline=centrality1, centrality=centrality2, zoom=False, note=note)


def main():

    all_conf_files = ['example_config.ini']
    for conf_file in all_conf_files:
        configuration = library.get_config_from_ini(conf_file)
        for centrality_pair in configuration.centralities:
            run_comparing_sim(configuration.runs, centrality_pair[0], centrality_pair[1], configuration, note=configuration.note, parallel=False)
            print(f"Completed simulation for {centrality_pair[0]} vs. {centrality_pair[1]}, configuration: {conf_file}")


if __name__ == "__main__":
    main()
