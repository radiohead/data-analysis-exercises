import math
import numpy
import seaborn
import utils

from csv import DictReader

from matplotlib import pyplot

from pandas import DataFrame
from pandas.tools import plotting

from sklearn import manifold, decomposition
from sklearn.preprocessing import StandardScaler

def sturges(data, attribute):
    return math.ceil(math.log(len(data), 2) + 1)

def scotts(data, attribute):
    attribute_data = map(lambda row: row[attribute], data)
    deviation = numpy.std(attribute_data)
    height = 3.5 * deviation / math.pow(len(attribute_data), 1/3)

    return math.ceil((max(attribute_data) - min(attribute_data)) / height)

def square_root(data, attribute):
    return int(math.sqrt(len(data)))

BIN_FUNCTIONS = {
    'sturges': sturges,
    'scotts': scotts,
    'sqrt': square_root
}

def plot_histogram(data, attribute, bin_function, bin_name):
    bin_number = bin_function(data, attribute)
    histogram_data = map(lambda row: row[attribute], data)

    n, bins, _ = pyplot.hist(histogram_data, bin_number, [min(histogram_data), max(histogram_data)], facecolor='green', alpha=0.75)
    pyplot.axis([min(bins), max(bins), min(n), max(n)])
    pyplot.grid(True)

    pyplot.xlabel(attribute)
    pyplot.ylabel('Amount')
    pyplot.title('Histogram of ' + attribute + ' using ' + bin_name + ' for bin number')
    pyplot.savefig('build/histograms/' + str.replace(attribute, ' ', '_') + '_' + bin_name + '.png')
    pyplot.clf()

def plot_scatter_matrix(data_frame):
    plotting.scatter_matrix(data_frame, alpha=0.2, figsize=(6, 6), diagonal='kde')
    figure = pyplot.gcf()
    figure.set_size_inches(20, 20)
    pyplot.savefig('build/scatter_matrix.png', dpi=300)
    pyplot.clf()

def plot_parallel_coordinates(data_frame):
    plotting.parallel_coordinates(data_frame, 'quality', colors = seaborn.color_palette('pastel', n_colors=6).as_hex())

    figure = pyplot.gcf()
    figure.set_size_inches(15, 15)
    pyplot.savefig('build/parallel_coordinates.png', dpi=300)
    pyplot.clf()

def plot_pca_projection(data, normalized = False):
    samples = numpy.array([[value for key, value in row.items() if key not in ['quality']] for row in data])
    classes = numpy.array([row['quality'] for row in data])
    class_names = numpy.unique(classes)
    colors = seaborn.color_palette('pastel', n_colors = len(class_names))

    if normalized:
        samples = StandardScaler().fit(samples).fit_transform(samples)

    pca = decomposition.PCA(n_components = 2)
    transformed = pca.fit_transform(samples)

    pyplot.figure()
    for c, i, class_name in zip(colors, range(0, len(class_names)), class_names):
        pyplot.scatter(transformed[classes == i, 0], transformed[classes == i, 1], c=c, label=class_name)
    pyplot.legend()

    pyplot.axhline(0, color='black')
    pyplot.axvline(0, color='black')
    pyplot.grid(True)
    pyplot.savefig('build/pca_projection' + ('_normalized' if normalized else '') + '.png')

def plot_mds(data):
    samples = numpy.array([[value for key, value in row.items() if key not in ['quality']] for row in data])
    classes = numpy.array([row['quality'] for row in data])
    class_names = numpy.unique(classes)
    colors = seaborn.color_palette('pastel', n_colors = len(class_names))

    mds = manifold.MDS(n_components = 2)
    transformed = mds.fit(samples).fit_transform(samples)

    pyplot.figure()
    for c, i, class_name in zip(colors, range(0, len(class_names)), class_names):
        pyplot.scatter(transformed[classes == i, 0], transformed[classes == i, 1], c=c, label=class_name)
    pyplot.legend()

    pyplot.axhline(0, color='black')
    pyplot.axvline(0, color='black')
    pyplot.grid(True)
    pyplot.savefig('build/2d_mds.png')

def main():
    data = utils.read_data_from_csv('data/winequality-red.csv')

    for attribute in data[0].keys():
        for name, func in BIN_FUNCTIONS.iteritems():
            plot_histogram(data, attribute, func, name)

    data_frame = DataFrame(data)
    plot_scatter_matrix(data_frame)
    plot_parallel_coordinates(data_frame)

    plot_pca_projection(data)
    plot_pca_projection(data, normalized = True)

    plot_mds(data)

    data_frame.corr(method='pearson').to_csv('build/pearson.csv')
    data_frame.corr(method='kendall').to_csv('build/kendall.csv')

if __name__ == '__main__':
    main()
