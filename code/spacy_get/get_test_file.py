from glob import glob
import os
import spacy
import pandas as pd
from tqdm import tqdm
from API.text_preprocessing import *
from nltk.tokenize import sent_tokenize



def read(input_dir, output_dir):
    count = 0
    doc_list = []
    file_list = list(sorted(glob(os.path.join(input_dir, '*'))))
    # file_list = file_list[:1]
    print(file_list)
    for file_name in file_list:
        file_id = file_name.split('/')[-1]
        output_id = output_dir + '/' + file_id
        isExists = os.path.exists(output_id)
        if isExists:
            print('路径', output_id, '已经存在')
        else:
            os.mkdir(output_id)
            print('路径', output_id, '创建成功')
        txt_list = list(sorted(glob(os.path.join(file_name, '*txt'))))
        print(len(txt_list))

        for i in tqdm(range(len(txt_list))):
            txt_path = txt_list[i]
        # for i, txt_path in enumerate(txt_list):
            book_name = txt_path.split('/')
            book_name = book_name[-1]
            book_name = book_name[:len(book_name) - 4]
            # 获得全部的句子
            original_sents = open(txt_path).readlines()
            sents, n_sent = convert_into_sentences(original_sents)

            paper = ' '.join(sents)
            paper = fileter_content(paper)
            print(paper)

            sentences = sent_tokenize(paper)
            print(sentences)
            print(len(sentences))
            print('---')
            for i in range(1, len(sentences)-1):
                count += 1
                id1, id2 = str(count), str(count)
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (str(1), id1, id2, sentences[i], sentences[i+1]))
                if count > 10000:
                    return


if __name__ == '__main__':
    train_fh = open('train.tsv', 'w')
    count = 0
    header = "Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n"
    train_fh.write(header)

    choose = 1
    data_path = '../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    input_list = ['data_txt/icw', 'data_txt/bok', 'data_txt/gut', 'data_txt/wiki']
    output_list = ['data_repickle/icw', 'data_repickle/bok', 'data_repickle/gut', 'data_repickle/wiki']

    input_path = data_path + input_list[choose]
    output_path = data_path + output_list[choose]
    read(input_path, output_path)
    print('finish')