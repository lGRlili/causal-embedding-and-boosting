import sys

from sota.att import AttData, AttModel
from sota.max import MaxData, MaxModel
from sota.pw import PairWise, PWData
import os


def train_model(model, data_path, cause_path, effect_path, batch_size, num_samples,
                num_epochs, embedding_size, min_count, learning_rate, sample_neg_randomly):
    if cause_path == '' or effect_path == '':
        print('both cause and effect model path can not be null')
        exit(1)

    if model == 'pw':
        causalVec = PairWise(
            embedding_size=embedding_size, batch_size=batch_size, num_epochs=num_epochs,
            num_samples=num_samples, learning_rate=learning_rate, data_loader=PWData(sample_neg_randomly, num_samples)
        )
    elif model == 'max':
        causalVec = MaxModel(
            embedding_size=embedding_size, batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate, num_samples=num_samples, data_loader=MaxData(sample_neg_randomly, num_samples)
        )
    else:
        causalVec = AttModel(
            embedding_size=embedding_size, batch_size=batch_size, num_epochs=num_epochs,
            num_samples=num_samples, learning_rate=learning_rate, data_loader=AttData(sample_neg_randomly, num_samples)
        )

    causalVec.load_data(data_path=data_path, min_count=min_count)
    causalVec.construct_graph()
    causalVec.train_stage(cause_output_path=cause_path, effect_output_path=effect_path)
    print('train stage is over!')


if __name__ == '__main__':
    project_source_path = ''
    path = os.path.join(project_source_path, '')
    params = {
        'data_path': {
            # 'pos_path': project_source_path + 'cross/bk_verb_positives.txt',
            # 'neg_path': project_source_path + 'cross/bk_negatives.txt',
            # 'labeled_path': project_source_path + 'cross/sg_eva.txt',
            'pos_path': path + '../../data/embedding_data/sharp_data.txt',
            # 'neg_path': project_source_path + 'bk_negatives.txt',
            'labeled_path': path + 'sharp_annotate.txt'
        },
        # 'model': 'max',
        'model': 'attentive',
        'batch_size': 256,
        'num_epochs': 100,
        'embedding_size': 100,
        'learning_rate': 0.005,
        'cause_path': project_source_path + 'temp/bk2sg/bk2sg_max_cause',
        'effect_path': project_source_path + 'temp/bk2sg/bk2sg_max_effect',
        'min_count': 1,
        'num_samples': 10,
        'sample_neg_randomly': True,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    train_model(
        model=params['model'], data_path=params['data_path'], cause_path=params['cause_path'],
        effect_path=params['effect_path'], batch_size=params['batch_size'], num_samples=params['num_samples'],
        num_epochs=params['num_epochs'], embedding_size=params['embedding_size'], min_count=params['min_count'],
        learning_rate=params['learning_rate'], sample_neg_randomly=params['sample_neg_randomly']
    )
