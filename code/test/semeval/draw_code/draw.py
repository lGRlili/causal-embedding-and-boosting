import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.font_manager import fontManager
import codecs
import numpy as np
# from causality.tools import project_source_path
import os
import random
from sklearn.metrics import auc


def draw_curve(data, params, name):
    plt.cla()
    plt.figure(figsize=(14, 9))

    # axes.set_title('p-r curve')
    for i in range(len(data)):
        points_list = np.array(data[i])

        recall, precision = [float(d[0]) for d in points_list], [float(d[1]) for d in points_list]
        plt.plot(
            recall, precision, label=params[i][0], color=params[i][1], linestyle=params[i][2],
            marker=params[i][3], ms=10, markevery=params[i][4]
        )

    plt.legend(loc='upper right', prop={'size': 17})
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.xlim(0, None)
    plt.ylim(0.4, None)
    # , bbox_inches = 'tight'
    plt.rcParams['savefig.dpi'] = 1000
    # plt.show()
    plt.savefig('{}.png'.format(name), bbox_inches='tight')


if __name__ == '__main__':
    params = {
        'en_sota': ('BMM', 'c', '-', '^', 0.1),
        'fl_max_11': ('Max-Matching', 'steelblue', '-.', 's', 0.1),
        'att-new': ('Att-Matching', 'coral', '--', 'X', (0.2, 0.05)),
        'Vanilla': ('vEmbed', 'green', '-.', 'o', 0.1),
        'Causal': ('cEmbed', 'teal', '-.', '>', (0.0, 0.05)),
        'Causal_bidir': ('cEmbedBi', 'goldenrod', '-.', 'd', (0.45, 0.1)),
        'Causal_bidir_pmi': ('cEmbedBiNoise', 'dimgrey', '-.', '*', 0.1),
    }

    points_data, parameters = [], []
    # path = 'points.{}'.format('max_41')
    path = 'PRCurve_icw_ourmethods.txt'
    # files = os.listdir(path)
    # names = [f.strip().split('.')[1] for f in files]
    # ordered_file = [
    #     # 'max20latest',
    #     'en_sota',
    #     'fl_max_11',
    #
    #     'att-new',
    #     'Causal_bidir_pmi',
    #
    #     'Causal_bidir',
    #     'Causal',
    #
    #     # 'pairwise-matching',
    #     # 'Lookup_baseline',
    #     'Vanilla',
    #
    # ]
    ordered_file = [
        # 'max20latest',
        'en_sota']
    for f in ordered_file:
        if f not in params:
            continue
        print(f)
        lines = codecs.open(path, 'r', 'utf-8').readlines()
        points = [line.strip().split(' ') for line in lines]
        points_data.append(points)
        parameters.append(params[str(f)])

    draw_curve(points_data, parameters, 'PRCurve')

