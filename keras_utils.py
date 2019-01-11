import keras.backend as K
import collections
import json
from keras.models import model_from_json
from loader import Loader

#### 機械学習用#####
# アンダーサンプリング
def under_sampling(X, Y, split):
    split_pos = int(len(X) * split)
    x, y = Creator.classify(X[0:split_pos], Y[0:split_pos])
    x_test, y_test = Creator.classify(X[split_pos:], Y[split_pos:])
    return x, y, x_test, y_test

# データ数の差を埋める形で重みを変える
def with_class_weight(y):
    count_dict = collections.Counter(numpy.asarray(y))
    max_value = max(count_dict.values())
    class_weight = dict()
    for k, v in count_dict.items():
        class_weight[k] = max_value / v
    print("#--class_weight--#")
    print(count_dict)
    print(class_weight)
    print("#--/class_weight--#")
    return class_weight

def validate_split(X, Y, split):
    split_pos = int(len(X) * split)
    x = X[0:split_pos]
    y = Y[0:split_pos]
    x_test = X[split_pos:]
    y_test = Y[split_pos:]
    return x, y, x_test, y_test

def histogram(data, begin, end):
    count_dict = collections.Counter(numpy.asarray(data))
    results = [count_dict[x] if x in count_dict else 0 for x in range(begin, end)]
    return results

def output_pgm(data, name, max_value, width):
    width = width
    height = len(data) / width
    with open(name, 'w') as f:
        f.write("P2\n")
        f.write("%s %s\n" % (int(width), int(height)))
        f.write("%s\n" % max_value)
        f.write("%s\n" % " ".join(map(lambda x: str(int(x)),data)))

def output_csv(data, name, sep=None, timestamp=None):
  detail = pandas.DataFrame(data)
  with_index = (timestamp is not None)
  if with_index:
    detail.index = pandas.to_datetime(timestamp)
  detail.to_csv(name, sep=sep, index=with_index, header=False)

def precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

# 正解のうち、どの程度が検索にヒットするか(precisionとトレードオフ)
def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def chunked(iterable, n):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]

