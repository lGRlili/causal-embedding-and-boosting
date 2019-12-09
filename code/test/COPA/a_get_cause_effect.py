import json
import spacy
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords


def filter_isalpha(word):
    return word.encode('UTF-8').isalpha()


def filter_stop_word(word):
    return word not in eng_stop_word


def get_data(file_name):
    with open(file_name, 'r') as file:
        while True:
            str_data = file.readline()
            if str_data == '':
                break
            train_data = json.loads(str_data)
            # train_data = json.dumps(train_data)
            print(train_data)
            # print(type(train_data))
            phrase_fir = train_data['premise']
            phrase_choose1 = train_data['choice1']
            phrase_choice2 = train_data['choice2']
            phrase_fir = phrase_fir.lower()
            phrase_choose1 = phrase_choose1.lower()
            phrase_choice2 = phrase_choice2.lower()

            # 对数据进行分词
            # 分词
            phrase_fir = WordPunctTokenizer().tokenize(phrase_fir)
            phrase_choose1 = WordPunctTokenizer().tokenize(phrase_choose1)
            phrase_choice2 = WordPunctTokenizer().tokenize(phrase_choice2)
            if filter_stop_word_flag:
                phrase_fir = filter(filter_stop_word, phrase_fir)
                phrase_choose1 = filter(filter_stop_word, phrase_choose1)
                phrase_choice2 = filter(filter_stop_word, phrase_choice2)
                phrase_fir, phrase_choose1, phrase_choice2 = list(phrase_fir), list(phrase_choose1), list(phrase_choice2)
            if filter_isalpha_flag:
                phrase_fir = filter(filter_isalpha, phrase_fir)
                phrase_choose1 = filter(filter_isalpha, phrase_choose1)
                phrase_choice2 = filter(filter_isalpha, phrase_choice2)
                phrase_fir, phrase_choose1, phrase_choice2 = list(phrase_fir), list(phrase_choose1), list(phrase_choice2)

            for word1 in phrase_fir:
                for word2 in phrase_choose1:
                    cause_effect_pair = word1 + '_' + word2
                    cause_effect_pair_list.add(cause_effect_pair)
            for word1 in phrase_fir:
                for word2 in phrase_choice2:
                    cause_effect_pair = word1 + '_' + word2
                    cause_effect_pair_list.add(cause_effect_pair)
            for word1 in phrase_choose1:
                for word2 in phrase_fir:
                    cause_effect_pair = word1 + '_' + word2
                    cause_effect_pair_list.add(cause_effect_pair)
            for word1 in phrase_choice2:
                for word2 in phrase_fir:
                    cause_effect_pair = word1 + '_' + word2
                    cause_effect_pair_list.add(cause_effect_pair)
            print(len(cause_effect_pair_list))


if __name__ == '__main__':
    filter_stop_word_flag = 1
    filter_isalpha_flag = 1
    data_path = '../../../data/'
    output_cause_effect_pair_count = data_path + 'data_pair_count/cause_effect_pair_count.txt'
    train_file_name = data_path + 'test/COPA/COPA/train.jsonl'
    test_file_name = data_path + 'test/COPA/COPA/test.jsonl'

    eng_stop_word = stopwords.words('english')
    eng_stop_word = set(eng_stop_word)
    print(eng_stop_word)
    print(len(eng_stop_word))

    file_list = []
    count = 0
    nlp = spacy.load('en_core_web_sm')
    nlp.remove_pipe('ner')
    cause_effect_pair_list = set()
    get_data(train_file_name)
    get_data(test_file_name)

    print(len(cause_effect_pair_list))
    output_path = 'cause_effect_pair.npy'
    np.save(output_path, list(cause_effect_pair_list))








