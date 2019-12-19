import sys

sys.path.extend(['../'])
from reference.max import MaxData
from reference.max import MaxModel
from glob import glob
import os
from collections import defaultdict
from sklearn.utils import shuffle

if __name__ == '__main__':

    data_path = '../../data/'
    data_path = '../../../../data_disk/ray/data/'
    pos_path = 'data_cause_effect'
    input_path = data_path + pos_path
    file_list = []
    data_choose_list = list(sorted(glob(os.path.join(input_path, '*'))))
    for data_choose in data_choose_list:
        file_choose = list(sorted(glob(os.path.join(data_choose, '*.npy'))))
        for file in file_choose:
            file_list.append(file)
    print(len(file_list))
    data_list = defaultdict(int)
    file_list = shuffle(file_list)

    data_list['pos_path'] = file_list[:1]
    # data_list['pos_path'] = 'sharp_data.txt'
    data_list['pos_path'] = 'advcl_for_embedding.txt'
    # data_list['pos_path'] = 'test.txt'
    data_list['labeled_path'] = []
    embedding_size = 100
    batch_size = 256
    num_epochs = 100
    num_samples = 5
    learning_rate = 0.05
    min_count = 10
    # embedding_size = 300
    # batch_size = 256
    # num_epochs = 10
    # num_samples = 5
    # learning_rate = 1e-4
    # min_count = 10
    print('embedding_size:', embedding_size, 'batch_size:', batch_size, 'num_epochs:', num_epochs,
          "num_samples:", num_samples, "learning_rate:", learning_rate, 'min_count:', min_count)
    data_loader = MaxData(sample_neg_randomly=True, num_samples=num_samples)
    model = MaxModel(embedding_size=embedding_size, batch_size=batch_size, num_epochs=num_epochs,
                     num_samples=num_samples, learning_rate=learning_rate, data_loader=data_loader)
    model.load_data(data_list, min_count=min_count)
    choose = 'max_match'
    # choose = 'top_K_match'
    # choose = 'pair_wise_match'
    choose = 'attentive_match'
    print('choose:', choose)
    if choose == 'max_match':
        cause_output_path = data_path + 'embedding' + '/cause_embedding_max_match_output_path'
        effect_output_path = data_path + 'embedding' + '/effect_embedding_max_match_output_path'

        model.construct_graph_max()
        model.train_stage(cause_output_path, effect_output_path)
    elif choose == 'top_K_match':
        cause_output_path = data_path + 'embedding' + '/cause_embedding_top_k_match_output_path'
        effect_output_path = data_path + 'embedding' + '/effect_embedding_top_k_match_output_path'

        model.construct_graph_top_k()
        model.train_stage(cause_output_path, effect_output_path)
    elif choose == 'pair_wise_match':
        cause_output_path = data_path + 'embedding' + '/cause_embedding_pair_wise_match_output_path'
        effect_output_path = data_path + 'embedding' + '/effect_embedding_pair_wise_match_output_path'

        model.construct_graph_pair_wise_match()
        model.train_stage(cause_output_path, effect_output_path)
    elif choose == 'attentive_match':
        cause_output_path = data_path + 'embedding' + '/cause_embedding_attentive_match_output_path'
        effect_output_path = data_path + 'embedding' + '/effect_embedding_attentive_match_output_path'

        model.construct_graph_attentive_match()
        model.train_stage(cause_output_path, effect_output_path)

