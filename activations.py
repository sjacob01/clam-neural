import pathlib

import matplotlib.cm
from matplotlib import pyplot
import numpy
import umap

DATA_ROOT = pathlib.Path(__file__).parent.joinpath('data')
CROSS_ROOT = pathlib.Path(__file__).parent.joinpath('data_cross')
PLOTS_ROOT = pathlib.Path(__file__).parent.joinpath('plots')


def umap_reductions(**kwargs) -> tuple[list[numpy.ndarray], numpy.ndarray]:
    reductions = list()
    for i in range(1, 7):
        path = CROSS_ROOT.joinpath(f'reduced_layer_{i}.npy')

        if path.exists():
            embedding = numpy.load(str(path))
        else:
            layer = numpy.concatenate([
                numpy.load(str(CROSS_ROOT.joinpath(f'mnist_activations_{i}.npy'))),
                numpy.load(str(CROSS_ROOT.joinpath(f'fashion_activations_{i}.npy'))),
            ])

            reducer = umap.UMAP(**kwargs)
            embedding = reducer.fit_transform(layer)
            numpy.save(str(path), numpy.asarray(embedding, dtype=numpy.float32), fix_imports=False, allow_pickle=False)

        reductions.append(embedding)

    labels_path = CROSS_ROOT.joinpath(f'reduced_labels.npy')
    if labels_path.exists():
        labels = numpy.load(str(labels_path))
    else:
        mnist = numpy.load(str(CROSS_ROOT.joinpath(f'mnist_labels.npy')))
        fashion_mnist = numpy.load(str(CROSS_ROOT.joinpath(f'fashion_labels.npy'))) + 10
        labels = numpy.asarray(numpy.concatenate([mnist, fashion_mnist]), dtype=numpy.uint8)
        numpy.save(
            str(labels_path),
            labels,
            fix_imports=False,
            allow_pickle=False,
        )

    return reductions, labels


def draw_plots(
        reductions: list[numpy.ndarray],
        labels: numpy.ndarray,
        binarize: bool
):

    if binarize:
        labels = numpy.asarray(labels > 9, dtype=numpy.uint8)

    unique_labels = numpy.unique(labels)

    for i, layer in enumerate(reductions, start=1):

        pyplot.figure(figsize=(16, 10))
        c_map = matplotlib.cm.get_cmap('tab20')

        for l in unique_labels:
            subset = layer[numpy.argwhere(labels == l)[:, 0]]
            x, y = subset[:, 0], subset[:, 1]

            if binarize:
                c = [c_map.colors[l * 10]] * subset.shape[0]
                label = 'mnist' if l == 0 else 'fashion'
                pyplot.scatter(x, y, c=c, s=0.1, alpha=0.5, label=label)
            else:
                c = [c_map.colors[l]] * subset.shape[0]
                label = f'mnist_{l}' if l < 10 else f'fashion_{l - 10}'
                pyplot.scatter(x, y, c=c, s=0.1, alpha=0.5, label=label)

        legend = pyplot.legend()
        for l, _ in enumerate(unique_labels):
            # noinspection PyProtectedMember
            legend.legendHandles[l]._sizes = [30]

        pyplot.title(f'layer {i} binarized' if binarize else f'layer {i}')
        path = PLOTS_ROOT.joinpath(f'layer_{i}_binary.png' if binarize else f'layer_{i}.png')
        pyplot.savefig(str(path), dpi=300)
        pyplot.close('all')

    return


def main():
    reductions, labels = umap_reductions()
    draw_plots(reductions, labels, False)
    draw_plots(reductions, labels, True)
    return


if __name__ == '__main__':
    assert DATA_ROOT.exists(), f'Path not found: {DATA_ROOT}'
    assert CROSS_ROOT.exists(), f'Path not found: {DATA_ROOT}'
    assert PLOTS_ROOT.exists(), f'Path not found: {PLOTS_ROOT}'
    main()
