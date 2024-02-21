import datetime
import multiprocessing
import typing
from collections import namedtuple
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

matplotlib.use("TkAgg")


def get_config_from_ini(path_to_ini: str) -> typing.NamedTuple:
    # returns a namedtuple defined below
    Conf = namedtuple('Conf', 'generator note name centralities suffix runs')

    # Get the configparser object
    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(path_to_ini)

    # Get graph properties
    topology = config_object["TOPOLOGY"]
    vertex_count = int(topology["N"])
    graph_model = topology["M"]
    gamma = None
    note = graph_model
    if graph_model == "configuration-model":
        gamma = float(config_object["CONFIGURATIONMODEL"]["gamma"])
        note = f'{graph_model}-"power_law_exp":{gamma}'

    # Get properties of simulation
    simulation = config_object["SIMULATION"]
    centralities = [x.strip() for x in simulation["centralities"].split(',')]
    all_scenarios = list(combinations_with_replacement(centralities, 2))
    runs = int(simulation["runs"])
    try:
        suffix = simulation["suffix"]
    except KeyError:
        suffix = ""

    # Finalize objects
    graph_generator = partial(generate_network, vertex_count, graph_model, gamma)
    full_note = f'{note}{("" if suffix == "" else "-")}{suffix}'

    return Conf(graph_generator, full_note, path_to_ini, all_scenarios, suffix, runs)


def generate_network(n, net_type, gamma=None):
    # Input:
    #   N - number of nodes
    #   cnType  - network type (i.e. scale-free, small-world, Erdos-Renyi random graph)
    # Output:
    #   graph - a network with random generated topology according to the input parameters

    match net_type:
        case "scale-free":
            graph = nx.powerlaw_cluster_graph(n, 5, 0.3)
            while not nx.is_connected(graph):
                graph = nx.powerlaw_cluster_graph(n, 5, 0.3)
        case "small-world":
            graph = nx.newman_watts_strogatz_graph(n, 6, 0.6)
            while not nx.is_connected(graph):
                graph = nx.newman_watts_strogatz_graph(n, 6, 0.6)
        case "Erdos-Renyi":
            graph = nx.fast_gnp_random_graph(n, 0.01)
            while not nx.is_connected(graph):
                graph = nx.fast_gnp_random_graph(n, 0.01)
        case "barabasi-albert":
            graph = nx.barabasi_albert_graph(n, 1)
        case "hep-th":
            graph = nx.read_edgelist('Cit-HepTh.txt')
        case "coll-grqc-LCC":
            tmp_graph = nx.read_edgelist('CA-GrQc.txt')
            tmp_graph = tmp_graph.subgraph(max(nx.connected_components(tmp_graph), key=len))
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "coll-grqc":
            tmp_graph = nx.read_edgelist('CA-GrQc.txt')
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "configuration-model":
            while True:
                degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
                if nx.is_graphical(degree_sequence):
                    break
            graph = nx.configuration_model(degree_sequence)
            graph = nx.Graph(graph)  # remove parallel edges
            graph.remove_edges_from(nx.selfloop_edges(graph))  # remove self-loops
        case _:
            raise Exception("Missing network type")

    return graph


def get_ranking_pagerank(graph: nx.Graph) -> typing.Dict:
    # pagerank_ranking = [i[0] for i in
    #                     sorted(nx.algorithms.pagerank(graph).items(), key=lambda x: x[1],
    #                            reverse=True)]
    return nx.algorithms.pagerank(graph)


def get_ranking_harmonic(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.harmonic_centrality(graph)


def get_ranking_eigenvector(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.eigenvector_centrality(graph)


def get_ranking_betweenness(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.betweenness_centrality(graph)


def get_ranking_load(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.load_centrality(graph)


def get_ranking_degree(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.degree_centrality(graph)


def get_ranking_closeness(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.closeness_centrality(graph)


def get_ranking_katz(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.katz_centrality_numpy(graph)


def run_in_parallel(runs: int, fn: typing.Callable) -> typing.List:
    with multiprocessing.Manager():
        pool = multiprocessing.Pool()
        async_results = []
        for i in range(runs):
            async_results.append(pool.apply_async(fn))
        pool.close()
        pool.join()
        results = [result.get() for result in async_results]

    return results


def plot_general(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, header: typing.AnyStr = None,
                 parabola: bool = False, zoom: bool = True, note: typing.AnyStr = None) -> None:
    if isinstance(results[0], list):
        mean = np.mean(np.array(results), axis=0)
        error = np.std(np.array(results), axis=0)
    else:
        mean = results
        error = 0

    plt.errorbar(range(1, 1 + len(mean)), mean, yerr=(error if len(results) > 1 else None), ecolor='r', capsize=5,
                 linestyle='none', marker='x',
                 markersize=2)
    if zoom:
        plt.title(
            f"{header} {baseline} vs. {centrality} \n Runs:{len(results)}, graph size: {len(mean)}, top 10% of vertices \n {note}")
        plt.xlim(1, 1 + (len(mean) / 10))
        plt.savefig(f"{datetime.datetime.now()}-{baseline}-{centrality}-top10p.pdf")

    plt.title(
        f"{header} {baseline} vs. {centrality} \n Runs:{len(results)}, graph size: {len(mean)}, all vertices  \n {note}")
    plt.xlim(0, len(mean))
    plt.ylim(0, len(mean))
    ax = plt.gca()
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    if parabola:
        points = np.linspace(0, len(mean), 100000)
        y_points = (points ** 2) / len(mean)
        ax.plot(points, y_points)

    Path("./results").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./results/{baseline}-{centrality}-{datetime.datetime.now():%Y-%m-%d-%X}.pdf")

    plt.clf()
