from glob import glob
import os
import spacy
import pandas as pd
from tqdm import tqdm
import sys
sys.path.extend(['../'])
from datetime import datetime
from API.text_preprocessing import *

starts = datetime.now()


def print_tiem():
    ends = datetime.now()
    detla_time = ends - starts
    seconds = detla_time.seconds
    second = seconds % 60
    minitus = int(seconds / 60)
    minitus = minitus % 60
    hours = int(seconds / 3600)
    print("已经经过:", hours, "H ", minitus, "M ", second, "S")


class parse(object):
    def __init__(self, text, norm, lemma_, pos_, tag_, dep_, head, id, child, left, right, ancestor):
        self.text = text
        self.norm = norm
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.head = head
        self.id = id
        self.child = child
        self.left = left
        self.right = right
        self.ancestor = ancestor


def get_spacy(paper, nlp, doc_list):

    max_len = 90000
    str_list = []
    # print(len(paper))
    while len(paper) > max_len:
        temp = paper[max_len:]
        # 找到100000之后最近的一个断句符
        c = temp.find('. ')
        # print(c)
        doc_input = paper[:max_len + c + 2]
        # 将当前的元素存入路径下
        str_list.append(doc_input)
        paper = paper[max_len + c + 2:]
        # print(doc_input[len(doc_input)-100:])
    str_list.append(paper)

    print('句子长度:', len(str_list), '*100000')
    try:
        for sig_str in str_list:
            ddoc = nlp(sig_str)
            # doc_list.append(ddoc)
            for sent in ddoc.sents:
                document = [item.text for item in sent.noun_chunks]
                # print(document)
                sentences = []
                # print('word'.rjust(11), 'word lemma'.rjust(11), 'word pos'.rjust(11), 'relationship'.rjust(11),
                #       'father_word'.rjust(11), 'fa_word pos'.rjust(11), 'id'.rjust(3), '子节点')
                for token in sent:
                    # print(token.text.rjust(11), token.lemma_.rjust(11), token.pos_.rjust(11), token.dep_.rjust(11),
                    #       token.head.text.rjust(11), token.head.pos_.rjust(11), str(token.i).rjust(3),
                    #       [child.i for child in token.children],[child.i for child in token.lefts],
                    # [child.i for child in token.rights],[child.i for child in token.ancestors])
                    # pass
                    # ent_type = token.ent_type_
                    # ent_id = token.ent_id_
                    # ent_iob = token.ent_iob_
                    # ent_kb_id = token.ent_kb_id_
                    #
                    # lower = token.lower_
                    # shape = token.shape_
                    # prefix = token.prefix_
                    # suffix = token.suffix_
                    # lang = token.lang_
                    # cluster = token.cluster
                    text = token.text
                    norm = token.norm_
                    lemma_ = token.lemma_
                    pos_ = token.pos_
                    tag_ = token.tag_
                    dep_ = token.dep_
                    head = token.head.i
                    iid = token.i
                    child = [child.i for child in token.children]
                    left = [child.i for child in token.lefts]
                    right = [child.i for child in token.rights]
                    ancestor = [child.i for child in token.ancestors]
                    temp_parse = parse(text, norm,  lemma_, pos_, tag_, dep_, head, iid, child, left, right, ancestor)
                    sentences.append(temp_parse)
                # print("***"*45)
                doc_list.append([sentences, document])
    except:
        print('error')

    return doc_list


def read(input_dir, output_dir):
    count = 0
    doc_list = []
    nlp = spacy.load('en_core_web_sm')
    nlp.remove_pipe('ner')
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
            if 'gut' in input_dir:
                original_sents = gut_remove_heda_tail(original_sents)
            sents, n_sent = convert_into_sentences(original_sents)

            paper = ' '.join(sents)
            paper = fileter_content(paper)
            paper += '\n \n just for seplit . \n \n'
            doc_list = get_spacy(paper, nlp, doc_list)
            print(len(doc_list))
            if len(doc_list) >= 10000:

                df_dict = {"doc": doc_list}
                df = pd.DataFrame(df_dict, columns=["doc"])
                doc_path = output_id + '/' + str(count) + ".pkl"
                count += 1
                print("doc_path:", doc_path)
                print("句子数据:", len(df))
                df.to_pickle(doc_path)
                print("save_safely")
                print("---" * 45)
                del doc_list[:]
                del df

            sys.stderr.write(
                '{}/{}\t{}\t{}\n'.format(i, len(txt_list), n_sent, txt_path))

        df_dict = {"doc": doc_list}
        df = pd.DataFrame(df_dict, columns=["doc"])
        doc_path = output_id + '/' + str(count) + ".pkl"
        count += 1
        print("doc_path:", doc_path)
        print("句子数据:", len(df))
        df.to_pickle(doc_path)
        print("save_safely")
        print("---" * 45)
        del doc_list[:]
        del df


def read_wiki(input_dir, output_dir):
    count = 0
    doc_list = []
    nlp = spacy.load('en_core_web_sm')
    nlp.remove_pipe('ner')
    file_list = list(sorted(glob(os.path.join(input_dir, '*'))))
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

        for i, txt_path in enumerate(txt_list):
            book_name = txt_path.split('/')
            book_name = book_name[-1]
            book_name = book_name[:len(book_name) - 4]
            print(book_name)
            # 获得全部的句子
            count_time = 0
            paper_list = []
            with open(txt_path, 'r') as file:
                while True:
                    line = file.readline()
                    count_time += 1
                    print(len(paper_list))
                    if not line:
                        break
                    # print(paper)
                    paper_list.append(line)
                    if len(paper_list) >= 10:
                        sents, n_sent = convert_into_sentences(paper_list)
                        paper = ' '.join(sents)
                        paper = fileter_content(paper)
                        doc_list = get_spacy(paper, nlp, doc_list)
                        del paper_list[:]
                        print(len(doc_list))
                        if len(doc_list) >= 10000:
                            df_dict = {"doc": doc_list}
                            df = pd.DataFrame(df_dict, columns=["doc"])
                            doc_path = output_id + '/' + str(count) + ".pkl"
                            count += 1
                            print("doc_path:", doc_path)
                            print("句子数据:", len(df))
                            df.to_pickle(doc_path)
                            print("save_safely")
                            print("---" * 45)
                            del doc_list[:]
                            del df


def split_sentence(doc_name):
    road = doc_name + '/1/en_wiki_1.txt'
    # book_name = road.split('/')
    # book_name = book_name[-1]
    book_name = road[:len(road) - 4]
    paper_list = []

    count = 0

    with open(road, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            paper_list.append(line)
            print(line)
            if len(paper_list) >= 100:
                save_path = book_name + "_" + str(count) + '.txt'
                print(save_path)
                count += 1
                with open(save_path, 'w') as save_file:
                    for data_line in paper_list:
                        save_file.write(data_line)

                del paper_list[:]


if __name__ == '__main__':

    choose = 0
    data_path = '../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    input_list = ['data_txt/icw', 'data_txt/bok', 'data_txt/gut', 'data_txt/wiki']
    output_list = ['data_repickle/icw', 'data_repickle/bok', 'data_repickle/gut', 'data_repickle/wiki']

    input_path = data_path + input_list[choose]
    output_path = data_path + output_list[choose]
    read(input_path, output_path)

    # read_wiki(data_path + input_list[3], data_path + output_list[3])
    # split_sentence(data_path + input_list[3])