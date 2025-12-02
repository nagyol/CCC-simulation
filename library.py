import datetime
import multiprocessing
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

try:
    import grape
    HAS_GRAPE = True
except ImportError:
    HAS_GRAPE = False

import pandas as pd
import tempfile
import os

Conf = namedtuple('Conf', 'generator note name centralities suffix runs')

def nx_to_grape(graph: typing.Union[nx.Graph, nx.DiGraph]) -> "grape.Graph":
    if not HAS_GRAPE:
        raise ImportError("grape not installed")

    # GRAPE loads efficiently from edge lists.
    # To avoid disk I/O we might want a direct bridge, but standard way is often via file or pandas.
    # converting to pandas is fast.
    # We'll use a temporary file to be safe and robust.

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        tmp_filename = tmp.name
        # Write edges
        # We can use nx.to_pandas_edgelist but that creates a DF.
        # nx.write_edgelist might be faster?
        # nx.write_edgelist(graph, tmp_filename, delimiter=',', data=False)
        # But we need headers for GRAPE usually or index logic.

        # Let's use pandas for simplicity and robustness with column names
        df = nx.to_pandas_edgelist(graph)
        df.to_csv(tmp_filename, index=False)

    try:
        g = grape.Graph.from_csv(
            edge_path=tmp_filename,
            edge_list_separator=",",
            edge_list_header=True,
            sources_column="source",
            destinations_column="target",
            directed=graph.is_directed(),
            name="graph",
            verbose=False
        )
        return g
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

def nx_to_igraph(graph: typing.Union[nx.Graph, nx.DiGraph]) -> "ig.Graph":
    # Efficient conversion from NetworkX to igraph
    # Assumes nodes are integers 0..N-1
    if not HAS_IGRAPH:
        raise ImportError("igraph not installed")

    directed = graph.is_directed()
    # If the graph is a MultiGraph, igraph handles parallel edges, but we need to verify expectations.
    # The config models in generate_network seem to be mostly simple graphs or handle multigraphs by conversion.

    # igraph from_networkx is quite fast, but we can potentially speed it up
    # if we know the structure.
    # However, from_networkx is generally well-optimized in recent versions.
    # One caveat: from_networkx might preserve node names as 'name' attribute,
    # and vertex indices might not match node IDs if node IDs are not contiguous integers.
    # But our generate_network ensures conversion to integers.

    # Let's verify if node IDs align.
    # If nodes are 0..N-1, igraph will use them as indices implicitly if we just pass edges?
    # No, from_networkx maps them.

    return ig.Graph.from_networkx(graph)

def _grape_centrality_wrapper(graph: nx.Graph, func_name: str, **kwargs) -> typing.Dict:
    # Prefer GRAPE for path-based metrics (Betweenness, Closeness)
    if HAS_GRAPE:
        try:
            # Mapping
            # Closeness -> get_closeness_centrality
            # Betweenness -> get_betweenness_centrality (approximated?)
            # GRAPE uses 'approximated' names sometimes.

            use_grape = False
            if func_name in ["closeness", "betweenness", "harmonic_centrality", "degree"]:
                use_grape = True

            if not use_grape:
                return None

            g_grape = nx_to_grape(graph)

            scores = None
            if func_name == "closeness":
                 scores = g_grape.get_closeness_centrality()
            elif func_name == "betweenness":
                 # Use exact if available, else approximated?
                 # Based on benchmark, get_betweenness_centrality might be missing or named differently in some versions.
                 # But in the snippet I saw 'get_weighted_approximated...'.
                 # If exact is missing, we might skip.
                 # Actually, let's try standard name.
                 if hasattr(g_grape, "get_betweenness_centrality"):
                     scores = g_grape.get_betweenness_centrality()
                 elif hasattr(g_grape, "get_approximated_betweenness_centrality"):
                     scores = g_grape.get_approximated_betweenness_centrality()
            elif func_name == "harmonic_centrality":
                 if hasattr(g_grape, "get_harmonic_centrality"):
                     scores = g_grape.get_harmonic_centrality()
            elif func_name == "degree":
                 scores = g_grape.get_degree_centrality()

            if scores is not None:
                # scores is likely a list or numpy array in node ID order?
                # GRAPE node mapping matches input order if we used CSV carefully.
                # Note: 'source', 'target' columns in CSV.
                # GRAPE might re-index. We need to map back.
                # GRAPE has `get_node_name_from_node_id`.

                # We need to construct the dict.
                # Assuming node names in CSV were string representations of NX ints?
                # to_pandas_edgelist preserves values.
                # If NX nodes are ints, CSV has ints.
                # GRAPE reads them as strings usually? Or inferred.

                # Safer: iterate nodes.
                result = {}
                # get_node_names returns list of names.
                # scores is aligned with node IDs.
                node_names = g_grape.get_node_names()
                # node_names should be the original IDs (as strings or ints).

                for i, score in enumerate(scores):
                    name = node_names[i]
                    # Convert back to int if NX used ints
                    try:
                         name_int = int(name)
                         result[name_int] = score
                    except:
                         result[name] = score
                return result

        except Exception as e:
            # print(f"GRAPE failed for {func_name}: {e}")
            pass

    return None

def _igraph_centrality_wrapper(graph: nx.Graph, func_name: str, **kwargs) -> typing.Dict:
    # Try GRAPE first for heavy path metrics
    if func_name in ["closeness", "betweenness", "harmonic_centrality"]:
        res = _grape_centrality_wrapper(graph, func_name, **kwargs)
        if res is not None:
            return res

    if HAS_IGRAPH:
        try:
            # We need to cache the conversion if possible, but the current architecture
            # passes a fresh graph or existing graph to each function.
            # Conversion overhead is small compared to speedup for heavy metrics (betweenness),
            # but might be comparable for fast ones (degree).

            ig_graph = nx_to_igraph(graph)

            # Map method names
            if func_name == "pagerank":
                # igraph pagerank returns a list of scores
                scores = ig_graph.pagerank(**kwargs)
            elif func_name == "betweenness":
                scores = ig_graph.betweenness(**kwargs)
            elif func_name == "closeness":
                # igraph closeness: normalized=True by default? No.
                # NX closeness is normalized.
                # igraph: closeness(vertices=None, mode=ALL, cutoff=None, weights=None, normalized=True)
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
                 # igraph < 0.10 might not have harmonic. 1.0.0 has it?
                 # It's often called harmonic_centrality or similar.
                 # Let's check attribute.
                 if hasattr(ig_graph, "harmonic_centrality"):
                     scores = ig_graph.harmonic_centrality(**kwargs)
                 else:
                     # Fallback to NX if not implemented (though it should be in recent igraph)
                     return None
            else:
                return None

            # Create dictionary mapping node ID -> score
            # If using from_networkx, the vertex attribute '_nx_name' stores original ID.
            # But if original IDs were integers 0..N-1, they might be implicit.

            if "_nx_name" in ig_graph.vs.attributes():
                return {v["_nx_name"]: s for v, s in zip(ig_graph.vs, scores)}
            else:
                # Assume contiguous 0..N-1
                return {i: s for i, s in enumerate(scores)}

        except Exception as e:
            # Fallback to NX on any error (e.g. implementation mismatch)
            print(f"igraph failed for {func_name}: {e}. Falling back to NetworkX.")
            pass

    return None

def get_config_from_ini(path_to_ini: str) -> typing.NamedTuple:
    # returns a namedtuple defined below


    # Get the configparser object
    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(path_to_ini)

    # Get graph properties
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

    # Get properties of simulation
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
    try:
        suffix = simulation["suffix"]
    except KeyError:
        suffix = ""

    # Finalize objects
    graph_generator = partial(generate_network, vertex_count, graph_model, gamma, out_gamma)
    full_note = f'{note}{("" if suffix == "" else "-")}{suffix}'

    return Conf(graph_generator, full_note, path_to_ini, all_scenarios, suffix, runs)


def generate_network(n, net_type, gamma=None, out_gamma=None):
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
            graph = nx.Graph(graph)  # remove parallel edges
            graph.remove_edges_from(nx.selfloop_edges(graph))  # remove self-loops
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
    # Try igraph first
    # igraph uses 'damping' instead of 'alpha'
    res = _igraph_centrality_wrapper(graph, "pagerank", damping=alpha)
    if res is not None:
        return res
    return nx.algorithms.pagerank(graph, alpha=alpha)


def get_ranking_harmonic(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "harmonic_centrality")
    if res is not None:
        return res
    return nx.algorithms.harmonic_centrality(graph)


def get_ranking_eigenvector(graph: nx.Graph) -> typing.Dict:
    # igraph eigenvector_centrality defaults to scale=True
    res = _igraph_centrality_wrapper(graph, "eigenvector_centrality", scale=False)
    if res is not None:
        return res
    return nx.algorithms.eigenvector_centrality_numpy(graph)


def get_ranking_betweenness(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "betweenness")
    if res is not None:
        return res
    return nx.algorithms.betweenness_centrality(graph)


def get_ranking_load(graph: nx.Graph) -> typing.Dict:
    # Load centrality is not standard in igraph (it's similar to betweenness but different).
    # Keep NX for now.
    return nx.algorithms.load_centrality(graph)


def get_ranking_degree(graph: nx.Graph) -> typing.Dict:
    # igraph degree is raw degree. NX degree_centrality is normalized (degree / (N-1)).
    # We must normalize igraph output to match NX.
    # But wait, my wrapper returns raw scores for degree?
    # Let's adjust wrapper or adjust here.
    # Adjusting here is safer.

    # Actually, for the purpose of ranking (ordering), normalization doesn't matter!
    # The 'compare_centrality' function only cares about the order of nodes.
    # However, if we ever compare raw values, it matters.
    # The function name is 'get_ranking_*' but it returns a dict of values.
    # The consuming code in main.py: compare_centrality sorts them.
    # So relative order is all that matters.
    # Raw degree and normalized degree have same order.

    res = _igraph_centrality_wrapper(graph, "degree")
    if res is not None:
        return res
    return nx.algorithms.degree_centrality(graph)

def get_ranking_indegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "indegree")
    if res is not None:
        return res
    return nx.algorithms.in_degree_centrality(graph)

def get_ranking_outdegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "outdegree")
    if res is not None:
        return res
    return nx.algorithms.out_degree_centrality(graph)


def get_ranking_closeness(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "closeness")
    if res is not None:
        return res
    return nx.algorithms.closeness_centrality(graph)


def get_ranking_katz(graph: nx.Graph) -> typing.Dict:
    if graph.is_multigraph():
        graph = nx.DiGraph(graph)
    alpha = 1./(2*max(graph.degree())[0])
    return nx.algorithms.katz_centrality_numpy(graph, alpha=alpha)


def run_in_parallel(runs: int, fn: typing.Callable) -> typing.List:
    with multiprocessing.Pool() as pool:
        # Using map or imap might be cleaner, but starmap/map expects iterables.
        # Since fn takes no arguments (it is a partial), we can just execute it `runs` times.
        # However, pool.map needs an iterable.
        # We can use apply_async in a list comp or loop, but context manager handles cleanup.
        async_results = [pool.apply_async(fn) for _ in range(runs)]
        results = [result.get() for result in async_results]
    return results


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
