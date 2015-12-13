import math
import numpy
import seaborn
import utils

from matplotlib import pyplot
from sklearn import neighbors, linear_model

def calculate_linear_errors(model, predicted, real):
    errors = []

    for index, _ in enumerate(predicted):
        errors.append(math.fabs(predicted[index] - real[index]))

    return errors

def plot_errors(errors, plot_name):
    n, bins = numpy.histogram(errors, utils.square_root(errors))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    pyplot.plot(bin_centers, n)
    pyplot.grid(True)

    pyplot.xlabel(plot_name)
    pyplot.ylabel('Amount')

def regression(model, X_train, X_test, Y_train, Y_test, type = 'linear'):
    model.fit(X_train, Y_train)

    if type == 'linear':
        for row in zip(model.coef_, attributes):
            print("[%0.3f, %s]" % row)
        print "%0.3f" % model.intercept_

    in_sample_errors = calculate_linear_errors(model, model.predict(X_train), Y_train)
    plot_errors(in_sample_errors, 'Absolute error (in-sample)')
    utils.save_plot(pyplot, name = "build/%s_in_sample.png" % type)

    print "In-sample variance: %f" % numpy.var(in_sample_errors)
    print "In-sample mean: %f" % numpy.mean(in_sample_errors)

    out_sample_errors = calculate_linear_errors(model, model.predict(X_test), Y_test)
    plot_errors(out_sample_errors, 'Absolute error (out-sample)')
    utils.save_plot(pyplot, name = "build/%s_out_sample.png" % type)

    print "Out-of-sample variance: %0.3f" % numpy.var(out_sample_errors)
    print "Out-of-sample mean: %0.3f" % numpy.mean(out_sample_errors)

    return (numpy.mean(out_sample_errors) + numpy.mean(in_sample_errors)) / 2

if __name__ == '__main__':
    dataset = utils.dict_to_numpy(
        utils.read_data_from_csv('data/winequality-red.csv'),
        columns_to_exclude = ['fixed acidity', 'chlorides', 'free sulfur dioxide'])

    data = dataset['data']
    target = dataset['target']
    attributes = dataset['attributes']

    X_train = data[:-100]
    X_test = data[-100:]
    Y_train = target[:-100]
    Y_test = target[-100:]

    print 'Linear regression'
    regression_model = linear_model.LinearRegression()
    regression(regression_model, X_train, X_test, Y_train, Y_test, 'linear')
    print

    for i in range(1, 9):
        print 'kNn regression for %s neighbors' % i
        regression_model = neighbors.KNeighborsRegressor(i)
        print 'Avg error %0.4f' % regression(regression_model, X_train, X_test, Y_train, Y_test, 'knn_%s' % i)
        print
