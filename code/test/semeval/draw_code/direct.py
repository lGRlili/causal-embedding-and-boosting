import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.font_manager import fontManager
import codecs
import numpy as np
import random
import os
import re
import json


def calculate_sim(pairs, vec1, vec2):
    _res, _wp = [], []
    for w1, w2, label in pairs:
        score = -1.0 if w1 not in vec1 or w2 not in vec2 else np.dot(vec1[w1], vec2[w2])
        _res.append((score, float(label)))
    return _res


def get_pr_points(sorted_score, relevant_label):
    """
    :param sorted_score: item in it is like: (score, label)
    :param relevant_label:
    :return:
    """
    numPair = len(sorted_score)
    assert numPair > 0
    a= [1 for s in sorted_score if s[1] == relevant_label]
    numRelevant = sum([1 for s in sorted_score if s[1] == relevant_label])
    curvePoints = []
    scores = sorted(list(set([s[0] for s in sorted_score])), reverse=True)
    groupedByScore = [(s, [item for item in sorted_score if item[0] == s]) for s in scores]
    for i in range(len(groupedByScore)):
        score, items = groupedByScore[i]
        numRelevantInGroup = sum([1 for item in items if item[1] == relevant_label])
        if numRelevantInGroup > 0:
            sliceGroup = groupedByScore[:i + 1]
            items_slice = [x for y in sliceGroup for x in y[1]]
            numRelevantInSlice = sum([1 for s in items_slice if s[1] == relevant_label])
            sliceRecall = numRelevantInSlice / float(numRelevant)
            slicePrecision = numRelevantInSlice / float(len(items_slice))
            curvePoints.append((sliceRecall, slicePrecision))
    return curvePoints


def save_points(points, saved_path):
    fout = codecs.open(saved_path, 'w', 'utf-8')
    for recall, precision in points:
        fout.write('{} {}\n'.format(recall, precision))


if __name__ == '__main__':
    aa = 'points.{}'.format('max_41')
    print(aa)
    max_scores = [[3, 1], [4, 0], [6, 1], [8, 0]]
    result = sorted(max_scores, key=lambda s: s[0], reverse=True)
    max_points = get_pr_points(result, relevant_label=1)
    save_points(max_points, aa)


