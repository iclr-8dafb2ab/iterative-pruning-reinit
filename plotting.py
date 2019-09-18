import cycler
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math
import numpy as np
import torch


def norm(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)


def plot_linear_weights(tensor, minimum=None, maximum=None):

    # for image normalization purposes in [0, 1]
    minimum = minimum if minimum is not None else float(tensor.min())
    maximum = maximum if maximum is not None else float(tensor.max())

    plt.imshow(norm(tensor.numpy(), minimum, maximum))
    plt.show()


def plot_fc_param(fc_param, minimum=None, maximum=None):

    # for image normalization purposes in [0, 1]
    minimum = minimum if minimum is not None else float(tensor.min())
    maximum = maximum if maximum is not None else float(tensor.max())

    plt.figure(figsize=(7, 7))

    plt.imshow(
        fc_param.data.numpy(), cmap=plt.cm.RdBu_r, vmin=minimum, vmax=maximum
    )
    plt.axis("off")
    plt.show()


def plot_conv_weights(tensor, minimum=None, maximum=None):
    """Generate a canvas with multiple subplots organized in a square, each
    representing the weights of a single filter in the convolutional kernel.
    """
    num_filters, height, width = tensor.shape

    # needed for square display
    num_rows = math.ceil(math.sqrt(num_filters))

    # generate array of subplots to fill in
    fig, axs = plt.subplots(num_rows, num_rows, figsize=(10, 10))

    # for image normalization purposes in [0, 1]
    minimum = minimum if minimum is not None else float(tensor.min())
    maximum = maximum if maximum is not None else float(tensor.max())

    # plot each filter in its own subplot
    for filtr, ax in zip(tensor, axs.ravel()):
        ax.imshow(norm(filtr.numpy(), minimum, maximum))

    # remove axis ticks and show
    plt.setp(axs, xticks=[], yticks=[])
    plt.show()


def plot_conv_param(conv_param, minimum=None, maximum=None):

    # for image normalization purposes in [0, 1]
    minimum = minimum if minimum is not None else float(tensor.min())
    maximum = maximum if maximum is not None else float(tensor.max())

    n_channels_out = conv_param.shape[0]
    n_channels_in = conv_param.shape[1]

    f, axarr = plt.subplots(n_channels_in, n_channels_out, figsize=(25, 10))

    for o, channel_o in enumerate(conv_param):
        for i, channel in enumerate(channel_o):
            if n_channels_in > 1:
                axis = axarr[i][o]
            else:
                axis = axarr[o]
            axis.imshow(
                channel.data.numpy(),
                cmap=plt.cm.RdBu_r,
                vmin=minimum,
                vmax=maximum,
            )
            axis.axis("off")
    plt.show()


def plot_bias(tensor, minimum=None, maximum=None):

    # for image normalization purposes in [0, 1]
    minimum = minimum if minimum is not None else float(tensor.min())
    maximum = maximum if maximum is not None else float(tensor.max())

    # split long bias tensor into small chunks of max length 150 for better
    # plotting
    num_columns = 150

    # plot each chunk in a different figure
    # TODO: use subfigures instead
    for i in range(int(np.ceil(tensor.shape[0] / float(num_columns)))):
        partial_tensor = tensor.numpy()[
            i * num_columns: (i + 1) * num_columns
        ]
        plt.figure(figsize=(10, 5))
        plt.imshow(norm(np.column_stack(partial_tensor), minimum, maximum))
        plt.show()


def plot_weight_evolution_pruning(weights):
    weights = np.array(weights)
    weights[weights == 0] = np.nan
    matplotlib.rcParams["axes.prop_cycle"] = cycler.cycler(
        "color", plt.cm.magma(np.linspace(0, 1, weights.shape[1]))
    )

    plt.figure(figsize=(7, 7))
    plt.plot(
        weights[:, np.argsort(abs(weights[0]))], linestyle="-", alpha=0.7
    )
    plt.hlines(0, 0, weights.shape[0] - 1)
    plt.ylabel("Weight")
    plt.xlabel("Pruning Iteration")
    plt.show()


def plot_experiment_tree(nodes):
    import graphviz
    import dask
    from dask import dot
    from dask.base import collections_to_dsk

    dsk = dict(collections_to_dsk(list(nodes.values())))

    node_attr = None
    edge_attr = None
    data_attributes = {}
    function_attributes = {}

    graph_attr = {}
    graph_attr["rankdir"] = "BT"
    #     graph_attr.update(kwargs)
    g = graphviz.Digraph(
        graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
    )

    seen = set()

    states_uuids = {v.key: k for k, v in nodes.items()}

    for k, v in dsk.items():
        k_name = dask.dot.name(k)
        if k_name not in seen:
            seen.add(k_name)
            attrs = data_attributes.get(k, {})
            attrs.setdefault(
                "label", dot.box_label((k, states_uuids[str(k)]))
            )
            attrs.setdefault("shape", "box")
            g.node(k_name, **attrs)

        if dask.dot.istask(v):
            func_name = dask.dot.name((k, "function"))
            if func_name not in seen:
                seen.add(func_name)
                attrs = function_attributes.get(k, {})
                attrs.setdefault(
                    "label",
                    ",\n".join(
                        [
                            k
                            for k in v[0].keywords
                            if v[0].keywords[k] is not None
                        ]
                    ),
                )  # dask.dot.key_split(k))
                attrs.setdefault("shape", "circle")
                g.node(func_name, **attrs)
            g.edge(func_name, k_name)

            for dep in dask.dot.get_dependencies(dsk, k):
                dep_name = dask.dot.name(dep)
                if dep_name not in seen:
                    seen.add(dep_name)
                    attrs = data_attributes.get(dep, {})
                    attrs.setdefault(
                        "label", dot.box_label((dep, states_uuids[str(dep)]))
                    )
                    attrs.setdefault("shape", "box")
                    g.node(dep_name, **attrs)
                g.edge(dep_name, func_name)
        elif ishashable(v) and v in dsk:
            g.edge(name(v), k_name)
    return g
