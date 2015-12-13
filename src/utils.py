import math
from numpy import array as narray
from csv import DictReader

def read_data_from_csv(path):
    data = []

    with open(path, 'r') as csv_file:
      for row in DictReader(csv_file, delimiter=';'):
          for k in row:
              row[k] = float(row[k])
          data.append(row)

    return data

def dict_to_numpy(csv_data, columns_to_exclude = []):
    target = []
    data = []
    attributes = None

    for row in csv_data:
        target.append(row.pop('quality'))
        [row.pop(col) for col in columns_to_exclude]

        if attributes is None:
          attributes = row.keys() 

        data.append(row.values())

    target = narray(target)
    data = narray(data)

    return {
      'target': target,
      'data': data,
      'attributes': attributes
    }

def square_root(data, attribute = None):
    return int(math.sqrt(len(data)))

def save_plot(pyplot, name = 'plot.png'):
    pyplot.savefig(name, dpi=300)
    pyplot.clf()
