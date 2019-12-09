import pickle
import math
import copy
from datetime import datetime
from xml.dom.minidom import parse
import xml.dom.minidom
import json

data_path = '../../../data/'
input_text = data_path + 'test/COPA/COPA-resources/datasets/copa-dev.xml'
# input_label = data_path + 'test/COPA/COPA-resources/results/gold.dev'
# output = 'train.jsonl'
input_text = data_path + 'test/COPA/COPA-resources/datasets/copa-test.xml'
input_label = data_path + 'test/COPA/COPA-resources/results/gold.test'
output = 'test.jsonl'
DOMTree = xml.dom.minidom.parse(input_text)
collection = DOMTree.documentElement
# 在集合中获取数据
patterns = collection.getElementsByTagName("item")
count = 0


with open(output, 'w') as output_file:
    with open(input_label, 'r') as file:
        while file:

            label = file.readline()
            pattern = patterns[count]
            flag = pattern.getAttribute("asks-for")
            id_number = pattern.getAttribute("id")
            type_1 = pattern.getElementsByTagName('p')[0]
            premise = type_1.childNodes[0].data
            format_1 = pattern.getElementsByTagName('a1')[0]
            choice1 = format_1.childNodes[0].data
            rating = pattern.getElementsByTagName('a2')[0]
            choice2 = rating.childNodes[0].data
            print(type(label))
            label = label.replace('\n', '')
            label = label.split(' ')
            print(label)
            print(flag, id_number, premise, choice1, choice2)
            count += 1
            word_dict = {'premise': premise, 'choice1': choice1, 'choice2': choice2, 'question': flag, 'label': int(label[2]), 'idx': int(id_number)-1}
            # train_data = json.loads(str_data)
            word_dict = json.dumps(word_dict)
            output_file.write(word_dict + '\n')