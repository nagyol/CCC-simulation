import datetime
import glob
import multiprocessing
import os
import pickle
import uuid
import typing
from collections import namedtuple
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
from itertools import combinations
from pathlib import Path
from os import path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np

try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

Conf = namedtuple('Conf', 'generator note name centralities suffix runs cache save load cache_dir N M')

def nx_to_igraph(graph: typing.Union[nx.Graph, nx.DiGraph]) -> "ig.Graph":
    if not HAS_IGRAPH:
        raise ImportError("igraph not installed")
    return ig.Graph.from_networkx(graph)

def _igraph_centrality_wrapper(graph: nx.Graph, func_name: str, **kwargs) -> typing.Dict:
    if HAS_IGRAPH:
        try:
            ig_graph = nx_to_igraph(graph)
            if func_name == "pagerank":
                scores = ig_graph.pagerank(**kwargs)
            elif func_name == "betweenness":
                scores = ig_graph.betweenness(**kwargs)
            elif func_name == "closeness":
                scores = ig_graph.closeness(normalized=True, **kwargs)
            elif func_name == "eigenvector_centrality":
                scores = ig_graph.eigenvector_centrality(**kwargs)
            elif func_name == "degree":
                scores = ig_graph.degree(**kwargs)
            elif func_name == "indegree":
                 scores = ig_graph.degree(mode="in", **kwargs)
            elif func_name == "outdegree":
                 scores = ig_graph.degree(mode="out", **kwargs)
            elif func_name == "harmonic_centrality":
                 if hasattr(ig_graph, "harmonic_centrality"):
                     scores = ig_graph.harmonic_centrality(**kwargs)
                 else:
                     return None
            else:
                return None

            if "_nx_name" in ig_graph.vs.attributes():
                return {v["_nx_name"]: s for v, s in zip(ig_graph.vs, scores)}
            else:
                return {i: s for i, s in enumerate(scores)}
        except Exception as e:
            print(f"igraph failed for {func_name}: {e}. Falling back to NetworkX.")
            pass
    return None

def get_config_from_ini(path_to_ini: str) -> typing.NamedTuple:
    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(path_to_ini)

    topology = config_object["TOPOLOGY"]
    vertex_count = int(topology["N"])
    graph_model = topology["M"]
    gamma = None
    out_gamma = None
    note = graph_model
    if graph_model == "configuration-model":
        gamma = float(config_object["CONFIGURATIONMODEL"]["gamma"])
        note = f'{graph_model}-"PL_exp":{gamma}'
    if graph_model == "directed-CM":
        gamma = float(config_object["CONFIGURATIONMODEL"]["gamma"])
        out_gamma = float(config_object["CONFIGURATIONMODEL"]["out_gamma"])
        note = f'{graph_model}-"inPL_exp":{gamma}-"outPL_exp":{out_gamma}'

    simulation = config_object["SIMULATION"]
    if simulation["centralities"] == "pagerankDF":
        centralities = []
        if "DFstep" in simulation.keys():
            df_step = int(simulation["DFstep"])
        else:
            df_step = 10
        for i in range(0, 100, df_step):
            centralities.append(f"pagerank-{i}")
            all_scenarios = list(combinations(centralities, 2))
    else:
        centralities = [x.strip() for x in simulation["centralities"].split(',')]
        all_scenarios = list(combinations(centralities, 2))
    runs = int(simulation["runs"])

    suffix = simulation.get("suffix", "")
    cache = simulation.getboolean("cache", fallback=False)
    save = simulation.getboolean("save", fallback=False)
    load = simulation.getboolean("load", fallback=False)
    cache_dir = simulation.get("cache_dir", "cache")

    graph_generator = partial(generate_network, vertex_count, graph_model, gamma, out_gamma)
    full_note = f'{note}{("" if suffix == "" else "-")}{suffix}'

    return Conf(graph_generator, full_note, path_to_ini, all_scenarios, suffix, runs, cache, save, load, cache_dir, vertex_count, graph_model)

def generate_network(n, net_type, gamma=None, out_gamma=None):
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
        case "cit-hepph":
            tmp_graph = nx.read_edgelist('Cit-HepPh.txt', create_using=nx.DiGraph)
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "configuration-model":
            while True:
                degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
                if nx.is_graphical(degree_sequence):
                    break
            graph = nx.configuration_model(degree_sequence)
        case "configuration-model-simple":
            while True:
                degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
                if nx.is_graphical(degree_sequence):
                    break
            graph = nx.configuration_model(degree_sequence)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        case "directed-CM":
            in_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
            while True:
                out_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not out_gamma else out_gamma)]
                if sum(in_degree_sequence) == sum(out_degree_sequence):
                    break
            graph = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
        case "directed-CM-simple":
            in_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
            while True:
                out_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not out_gamma else out_gamma)]
                if sum(in_degree_sequence) == sum(out_degree_sequence):
                    break
            graph = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
            graph = nx.DiGraph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        case _:
            raise Exception("Missing network type")
    return graph

def get_ranking_pagerank(graph: nx.Graph, alpha: float = 0.85) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "pagerank", damping=alpha)
    if res is not None: return res
    return nx.algorithms.pagerank(graph, alpha=alpha)

def get_ranking_harmonic(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "harmonic_centrality")
    if res is not None: return res
    return nx.algorithms.harmonic_centrality(graph)

def get_ranking_eigenvector(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "eigenvector_centrality", scale=False)
    if res is not None: return res
    return nx.algorithms.eigenvector_centrality_numpy(graph)

def get_ranking_betweenness(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "betweenness")
    if res is not None: return res
    return nx.algorithms.betweenness_centrality(graph)

def get_ranking_load(graph: nx.Graph) -> typing.Dict:
    return nx.algorithms.load_centrality(graph)

def get_ranking_degree(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "degree")
    if res is not None: return res
    return nx.algorithms.degree_centrality(graph)

def get_ranking_indegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "indegree")
    if res is not None: return res
    return nx.algorithms.in_degree_centrality(graph)

def get_ranking_outdegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "outdegree")
    if res is not None: return res
    return nx.algorithms.out_degree_centrality(graph)

def get_ranking_closeness(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "closeness")
    if res is not None: return res
    return nx.algorithms.closeness_centrality(graph)

def get_ranking_katz(graph: nx.Graph) -> typing.Dict:
    if graph.is_multigraph():
        graph = nx.DiGraph(graph)
    alpha = 1./(2*max(graph.degree())[0])
    return nx.algorithms.katz_centrality_numpy(graph, alpha=alpha)

def get_ranking(centrality: typing.AnyStr):
    lookup = {
        "betweenness": get_ranking_betweenness,
        "closeness": get_ranking_closeness,
        "harmonic": get_ranking_harmonic,
        "pagerank": get_ranking_pagerank,
        "degree": get_ranking_degree,
        "load": get_ranking_load,
        "katz": get_ranking_katz,
        "eigenvector": get_ranking_eigenvector,
        "indegree": get_ranking_indegree,
        "outdegree": get_ranking_outdegree
    }
    for damping_factor in range(0,100):
        lookup.update({f"pagerank-{damping_factor}": partial(get_ranking_pagerank, alpha=damping_factor/100)})
    return lookup[centrality]

def run_in_parallel(runs: int, fn: typing.Callable) -> typing.List:
    with multiprocessing.Pool() as pool:
        async_results = [pool.apply_async(fn) for _ in range(runs)]
        results = [result.get() for result in async_results]
    return results

def get_cached_files(config: Conf) -> typing.List[str]:
    # Ensure cache directory exists
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # Construct search pattern
    # Pattern: graph_{M}_{N}_cent_{suffix}_{index/uuid}.gpickle
    # We use glob to find files starting with the prefix.
    # Note: suffix can be empty.

    prefix = f"graph_{config.M}_{config.N}_cent_{config.suffix}_"
    pattern = path.join(config.cache_dir, f"{prefix}*.gpickle")

    files = glob.glob(pattern)
    files.sort() # Ensure deterministic order
    return files

def process_graph(index: int, config: Conf, cached_files: typing.List[str], all_centralities: typing.List[str]) -> str:
    # This function is designed to be run in parallel (or serial)
    # It ensures the graph at 'index' is ready (loaded/generated, centralities computed, saved).
    # Returns the filename of the graph topology.

    file_path = None
    graph = None
    modified_graph = False

    # Ensure cache dir exists
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # Check if we can load an existing file
    if config.load and index < len(cached_files):
        file_path = cached_files[index]
        try:
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}. Generating new graph.")
            graph = None

    if graph is None:
        # Generate new graph
        graph = config.generator()
        modified_graph = True
        # If we generated a new graph, we need a new filename.
        # We assign a UUID to ensure uniqueness.
        unique_id = uuid.uuid4().hex
        filename = f"graph_{config.M}_{config.N}_cent_{config.suffix}_{unique_id}.gpickle"
        file_path = path.join(config.cache_dir, filename)

    # Base name for centrality files: matches graph filename but with .npy extension and suffix
    # file_path is like .../graph_..._uuid.gpickle
    # We want .../graph_..._uuid_centrality_{name}.npy
    base_name = path.splitext(path.basename(file_path))[0]

    # Ensure centralities
    for cent_name in all_centralities:
        npy_filename = f"{base_name}_centrality_{cent_name}.npy"
        npy_path = path.join(config.cache_dir, npy_filename)

        if not path.exists(npy_path):
            # Compute
            # print(f"Computing {cent_name} for graph {file_path}...")
            ranking_func = get_ranking(cent_name)
            scores_dict = ranking_func(graph)

            # Convert to array assuming nodes are 0..N-1
            # We assume the graph has N nodes labeled 0 to N-1
            # If not, we might have issues, but generator ensures integer labels.
            # We sort by node ID to ensure consistency.

            # Fast conversion
            # If nodes are exactly range(N):
            if graph.number_of_nodes() > 0:
                scores_arr = np.array([scores_dict[i] for i in range(graph.number_of_nodes())])
            else:
                scores_arr = np.array([])

            np.save(npy_path, scores_arr)

            # NOTE: We do NOT add attributes to the graph object anymore to keep it lightweight.
            # If the user wants the graph saved with centralities, we deviate here for performance.
            # Requirement: "graph itself should remain saved ... so that new centralities can be added"
            # We accomplish this by keeping the gpickle (topology) and adding new .npy files as needed.

    # Save graph topology if needed (only if new or modified)
    # If we just computed centralities but didn't change graph topology, we don't need to save graph unless it's new.
    if config.save and modified_graph:
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)

    return file_path

def get_normalized_total_orderings(baseline: typing.Union[typing.List, typing.Dict], other: typing.Union[typing.List, typing.Dict]) -> typing.List:
    n = len(baseline)
    if isinstance(baseline, dict):
        baseline_arr = np.array([baseline[i] for i in range(n)])
    else:
        baseline_arr = np.array(baseline)

    if isinstance(other, dict):
        other_arr = np.array([other[i] for i in range(n)])
    else:
        other_arr = np.array(other)

    random_arr = np.random.uniform(size=n)
    indices = np.arange(n)

    keys_baseline = (-random_arr, -other_arr, -baseline_arr)
    normalized_baseline = indices[np.lexsort(keys_baseline)]

    keys_other = (-random_arr, -baseline_arr, -other_arr)
    normalized_other = indices[np.lexsort(keys_other)]

    return [normalized_baseline.tolist(), normalized_other.tolist()]

def compare_centrality(baseline: typing.List, ranking_other: typing.List) -> typing.Dict:
    norm_baseline, norm_other = get_normalized_total_orderings(baseline, ranking_other)
    n = len(baseline)

    # Vectorized approach
    # norm_baseline[i] = node_id at rank i in baseline
    # We want rank_in_baseline[node_id]

    inv_nb = np.zeros(n, dtype=int)
    inv_nb[norm_baseline] = np.arange(n)

    inv_no = np.zeros(n, dtype=int)
    inv_no[norm_other] = np.arange(n)

    # For a node to be in the overlap set at step k (0-indexed),
    # it must have rank <= k in BOTH rankings.
    # So max(rank_baseline, rank_other) <= k.

    max_ranks = np.maximum(inv_nb, inv_no)

    # We want to count how many nodes have max_rank <= k for each k.
    # We can use bincount to get counts of nodes with max_rank == k.

    counts = np.bincount(max_ranks, minlength=n)
    overlap_list = np.cumsum(counts)

    return {'overlap': overlap_list.tolist()}

def plot_general(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, header: typing.AnyStr = None,
                 parabola: bool = False, zoom: bool = True, rescale: bool = False, note: typing.AnyStr = None) -> None:
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

    if rescale:
        ticks = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / len(mean)))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)

    Path("./results").mkdir(parents=True, exist_ok=True)
    filename = fr"{baseline}-{centrality}-{datetime.datetime.now():%Y-%m-%d-%H%M%S}.pdf"
    destination = path.join(path.abspath("results"), filename)
    plt.savefig(destination)

    plt.clf()
