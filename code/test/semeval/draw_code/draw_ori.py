import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.font_manager import fontManager
import codecs
import numpy as np
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
        # 'fl_att_11': ('fl_att_11', 'blue', '--'),
        # 'fl_att_21': ('fl_att_21', 'dimgrey', '-'),
        # 'fl_att_31': ('fl_att_31', 'dimgrey', '-.'),
        # 'fl_max_21': ('fl_max_21', 'olive', '-.'),
        # 'fl_max_31': ('fl_max_31', 'green', '--'),
        # 'extract_max': ('en_max_11', 'dimgrey', '-.'),
        # 'extract_att': ('att_new_11', 'olive', '-'),
        # 'attentive-matching': ('Attentive-Matching', 'blue', '--'),
        # 'combine2max': ('combine2max', 'green', '-'),
        # 'combine3max': ('combine3max', 'olive', '-.'),
        # 'combine4max': ('combine4max', 'red', '--'),
        # 'weighted6max': ('weighted6max', 'blue', '--'),
        # 'weighted6max2': ('weighted6max2', 'red', '-'),
        # 'weighted8max3': ('weighted8max3', 'magenta', '-.'),
        # 'boosted_7': ('boosted_7', 'green', '-'),
        # 'max20latest': ('BMM', 'red', '-'),
        'PRCurve_copa_dev_PMI': ('PRCurve_copa_dev_PMI', 'c', '-', '^', 0.1),
        'PRCurve_copa_test_PMI': ('PRCurve_copa_test_PMI', 'steelblue', '-.', 's', 0.1),
        'PRCurve_copa_dev_cause_effect_embedding': ('PRCurve_copa_dev_embedding', 'coral', '--', 'X', (0.2, 0.05)),
        'PRCurve_copa_test_cause_effect_embedding': ('PRCurve_copa_test_embedding', 'green', '-.', 'o', 0.1),
        'PRCurve_concepnet_PMI': ('PRCurve_concepnet_PMI', 'teal', '-.', '>', (0.0, 0.05)),
        'PRCurve_semeval_PMI': ('PRCurve_semeval_PMI', 'goldenrod', '-.', 'd', (0.45, 0.1)),
        #
        # 'en_sota': ('BMM', 'black', '--', '^', 0.1),
        # 'att-new': ('Att-Matching', 'black', ':', 'x', (0.45, 0.05)),
        # 'max-matching': ('Max', 'red', '-'),
        # 'pairwise-matching': ('Pairwise-Matching', 'dimgrey', '-.'),
        # 'Lookup_baseline': ('Look-up', 'olive', '-.'),
        # # 'pairwise-matching': ('Pairwise-Matching', 'dimgrey', '-.'),
        # # 'Lookup_baseline': ('Look-up', 'olive', '-.'),
        # 'Vanilla': ('vEmbed', 'black', '--', '*', 0.1),
        'PRCurve_copa_dev_PMI_dir': ('PRCurve_copa_dev_PMI_dir', 'coral', '--', 'X', (0.2, 0.05)),
        'PRCurve_copa_test_PMI_dir': ('PRCurve_copa_test_PMI_dir', 'green', '-.', 'o', 0.1),
        'PRCurve_copa_dev_PMI_dir1': ('PRCurve_copa_dev_PMI_dir1', 'teal', '-.', '>', (0.0, 0.05)),
        'PRCurve_copa_test_PMI_dir1': ('PRCurve_copa_test_PMI_dir1', 'goldenrod', '-.', 'd', (0.45, 0.1)),
        # 'Causal_bidir_pmi': ('cEmbedBiNoise', 'black', '--', 'o', 0.1),

        # 'extract_max': ('extract_max', 'blue', '-'),
        # 'extract_att': ('extract_att', 'dimgrey', '-.'),

    }

    points_data, parameters = [], []
    project_source_path = ['PRCurve_copa_dev_PMI.txt', 'PRCurve_copa_test_PMI.txt',
                           'PRCurve_copa_dev_cause_effect_embedding.txt',
                           'PRCurve_copa_test_cause_effect_embedding.txt',
                           'PRCurve_concepnet_PMI.txt', 'PRCurve_semeval_PMI.txt']
    project_source_path_dir = ['PRCurve_copa_dev_PMI.txt', 'PRCurve_copa_test_PMI.txt',
                               'PRCurve_copa_dev_PMI_dir.txt', 'PRCurve_copa_test_PMI_dir.txt',
                               'PRCurve_copa_dev_PMI_dir1.txt', 'PRCurve_copa_test_PMI_dir1.txt']
    # path = os.path.join(project_source_path, 'prcurve/')
    # files = os.listdir(path)
    # names = [f.strip().split('.')[1] for f in files]
    ordered_file = [
        'PRCurve_icw_ourmethods', 'PRCurve_icw_causalNET', 'PRCurve_icw_PMI',
        'PRCurve_bok_ourmethods', 'PRCurve_bok_causalNET', 'PRCurve_bok_PMI',
        'PRCurve_gut_ourmethods', 'PRCurve_gut_causalNET', 'PRCurve_gut_PMI']

    # for f in ordered_file:
    #     if f not in params:
    #         continue
    #     print(f)
    for path in project_source_path_dir[:]:
        f = path[:-4]
        lines = codecs.open(path, 'r', 'utf-8').readlines()
        points = [line.strip().split(' ') for line in lines]
        points_data.append(points)
        parameters.append(params[str(f)])

    draw_curve(points_data, parameters, 'PRCurve_dir1')
    # """
    # pr图重画 OK
    # auc计算 ok
    # example(en, cn分析)
    # architecture重画
    # 激发性模板 对照优缺点
    # """
    # for f in ordered_file:
    #     if f not in params:
    #         continue
    #     lines = codecs.open(os.path.join(path, 'points.{}'.format(f)), 'r', 'utf-8').readlines()
    #     points = [list(map(float, line.strip().split('\t'))) for line in lines]
    #     recall, precision = zip(*points)
    #     auc_val = auc(recall, precision)
    #     print(f, auc_val)
