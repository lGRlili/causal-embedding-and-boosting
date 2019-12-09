from collections import defaultdict
import numpy as np
import functools
import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)
# a = defaultdict(int)
#
# c = {'fucking', 'pd', 'nk', 'michael', 'austyn', 'emi', 'jpg', 'ap', 'quot', 'pg', 'yuy', 'ahs', 'god', 'idk', 'fff',
#      'ti', 'da', 'p', 'uw', 'yep', 'fic', 'gf', 'aa', 'pw', 'aiba', 'lem', 'hello', 'wa', 'wax', 'ra', 'yea', 'leh',
#      'ooh', 'sure', 'yum', 'lt', 'mhm', 'yay', 'wokay', 'wentz', 'yes', 'gosh', 'whew', 'ehm', 'hi', 'bookshelf',
#      'khayeed', 'cj', 'bk', 'waslike', 'dunno', 'hey', 'pc', 'shakin', 'shit', 'eww', 'sw', 'ay', 'hmmmm', 'wanan',
#      'nah', 'tbh', 'sorry', 'pillsy', 'lol', 'dj', 'oi', 'homey', 'um', 'ur', 'daniel', 'dp', 'bye', 'wc', 'yosh', 'jp',
#      'ugh', 'ot', 'heck', 'goodness', 'bo', 'hii', 'hb', 'uk', 'ok', 'please', 'yeah', 'spinelli', 'jus', 'wew', 'wow',
#      'melizza', 'gee', 'aha', 'ari', 'tachi', 'farewell', 'wth', 'boriiinnngg', 'la', 'ds', 'hmm', 'weaponlike', 'rp',
#      'hehehe', 'hm', 'ass', 'tch', 'bc', 'mulder', 'tashi', 'ohkay', 'ag', 'mmm', 'adn', 'kay', 'que', 'yuuri', 'wya',
#      'jj', 'imo', 'heather', 'fa', 'goodbye', 'darn', 'anyways', 'like', 'ah', 'er', 'aw', 'michelle', 'fuck', 'ms',
#      'w', 'sarah', 'bah', 'lena', 'eva', 'oh', 'see', 'tada', 'yeh', 'qb', 'hcg', 'sc', 'mi', 'ru', 'stroke', 'nn',
#      'mel', 'say', 'ba', 'man', 'nope', 'yup', 'ak', 'oj', 'ze', 'bla', 'sg', 'boy', 'dr', 'jeez', 'canst', 'fl',
#      'quack', 'uh', 'htm', 'okay', 'whoa', 'otaku', 'toshi', 'jc', 'hahaha', 'whatcha', 'obvio', 'sh', 'atsushi', 'agh',
#      'waiwai', 'tht', 'dear', 'saporta', 'hehe', 'gm', 'dg', 'va', 'yo', 'pj', 'nvr', 'mello', 'duh', 'hve', 'plz',
#      'right', 'welll', 'ho', 'ahh', 'uc', 'ht', 'amen', 'jk', 'oops', 'oe', 'alright', 'ike', 'duc', 'un', 'blah',
#      'trillian', 'saa', 'nut', 'td', 'well', 'ooooh', 'pls', 'alas', 'ls', 'dang', 'gl', 'wna', 'scouts', 'leonie',
#      'spike', 'yaa', 'kh', 'dun', 'fr', 'ha', 'im', 'hoorah', 'esl', 'firefly', 'au', 'hp', 'hah', 'yer', 'btr', 'mike',
#      'soo', 'asap', 'huh', 'eh', 'pto', 'anyway', 'yeeeah', 'davila', 'tb', 'quote', 'rachel', 'mug', 'damn', 'haha',
#      'uo', 'ob', 'iw', 'dh', 'bita', 'ss', 'bi'}
#
# d = {'eu', 'kim', 'joe', 'pelvis', 'bastien', 'ang', 'skali', 'gj', 'cl', 'el', 'herald', 'c', 'kudryavtsev', 'hawaii',
#      'jj', 'bc', 'luigi', 'cola', 'awwwww', 'kotetsu', 'meola', 'jim', 'naruto', 'rosi', 'eg', 'stepdad', 'ryan',
#      'meriah', 'mp', 'maureen', 'hyung', 'yay', 'ry', 'bix', 'yogurt', 'sonia', 'san', 'mart', 'dan', 'burien', 'kz',
#      'whedon', 'jon', 'konbanwa', 'gung', 'wt', 'wen', 'helena', 'nyc', 'microsoft', 'mei', 'u', 'meiguoren', 'woohoo',
#      'jerald', 'minimum', 'nawab', 'ru', 'fit', 'friday', 'mcdonald', 'uruguay', 'ju', 'gd', 'philippe', 'hsiaowei',
#      'zhoumi', 'nut', 'sober', 'ysb', 'jh', 'nichols', 'ethan', 'mac', 'ag', 'tobien', 'te', 'lo', 'engg', 'rikkai',
#      'amazon', 'hebrew', 'dr', 'dl', 'ba', 'yo', 'walvis', 'backboard', 'comm', 'july', 'gielgud', 'de', 'heaven',
#      'zelgadiss', 'god', 'nathan', 'sa', 'prac', 'dong', 'arbor', 'bloomington', 'domyoji', 'pu', 'nordstrom', 'ddd',
#      'pennsylvania', 'christian', 'earl', 'villas', 'org', 'speck', 'seg', 'fa', 'grrr', 'gavin', 'sob', 'heh', 'st',
#      'dungarvan', 'fus', 'mexican', 'feh', 'jb', 'kyo', 'caitlin', 'metallica', 'al', 'dee', 'jack', 'noah', 'xmas',
#      'panky', 'jyoti', 'kun', 'ish', 'adriana', 'duncan', 'yoghurt', 'hae', 'jsl', 'chinese', 'vox', 'yilin', 'xara',
#      'eww', 'nic', 'alisha', 'dw', 'ew', 'mh', 'cacie', 'devil', 'jonas', 'zig', 'vongola', 'tina', 'lewis', 'thursday',
#      'hailie', 'mono', 'lim', 'chicago', 'tis', 'jc', 'mature', 'jo', 'mc', 'curtis', 'bobbie', 'tsukushi', 'right',
#      'uf', 'tzu', 'monday', 'j', 'borg', 'ny', 'gw', 'mx', 'ruckus', 'wang', 'vay', 'joshua', 'lt', 'jarang', 'vb',
#      'california', 'decaf', 'zelos', 'sk', 'il', 'ehm', 'ri', 'jekyll', 'charlie', 'itm', 'wil', 'angelo', 'urahara',
#      'haku', 'say', 'yunsheng', 'py', 'hk', 'rowdy', 'sd', 'adora', 'jay', 'alaska', 'kl', 'urgggg', 'katamari',
#      'cleveland', 'nh', 'la', 'sai', 'r', 'clinton', 'cheressa', 'vcs', 'jovi', 'ebay', 'yso', 'toshiro', 'peru',
#      'pangpang', 'nessie', 'jimmy', 'xenia', 'yos', 'hrh', 'afghanistan', 'book', 'wm', 'po', 'wv', 'pp', 'sw', 'bob',
#      'walmart', 'da', 'johan', 'yang', 'zedekiah', 'zoriy', 'widescreen', 'yaz', 'sr', 'prime', 'chih', 'democrat',
#      'yugoslavia', 'yuki', 'epo', 'bee', 'kaj', 'onimusha', 'white', 'david', 'kou', 'zhang', 'english', 'dc', 'v',
#      'int', 'pov', 'tegoshi', 'lu', 'hun', 'jillian', 'jazira', 'fyi', 'sasuke', 'jm', 'damon', 'ann', 'asia', 'aang',
#      'con', 'ou', 'kubica', 'winky', 'chan', 'du', 'sarah', 'abby', 'snerk', 'mount', 'bonjour', 'cathy', 'cas',
#      'rachel', 'evan', 'chris', 'wal', 'rh', 'kat', 'rm', 'europeans', 'yah', 'soutien', 'mao', 'ho', 'ls', 'jazeera',
#      'eos', 'huang', 'saturday', 'vt', 'wa', 'teddy', 'ivan', 'jews', 'sux', 'swansboro', 'chang', 'nat', 'jae', 'gk',
#      'kojima', 'merihell', 'xd', 'secret', 'hillary', 'gm', 'frank', 'tuesday', 'bush', 'wes', 'ys', 'stockholm',
#      'babylon', 'draco', 'japanese', 'yugo', 'pw', 'pt', 'iki', 'lena', 'ching', 'becca', 'shen', 'haha', 'herzl',
#      'wakashi', 'jen', 'abbie', 'smali', 'uaz', 'chen', 'atan', 'doh', 'katakana', 'ting', 'oclc', 'qt', 'joanne', 'jv',
#      'troy', 'sac', 'jlpt', 'je', 'b', 'mo', 'mariana', 'ollie', 'yaya', 'wednesday', 'colombia', 'paycheck', 'pinyin',
#      'dxa', 'caedmon', 'bible', 'zodiac', 'chanoch', 'jor', 'uk', 'darien', 'msnbc', 'mon', 'trowa', 'ld', 'vd', 'ii',
#      'cnas', 'eac', 'september', 'dk', 'amber', 'libby', 'spawn', 'yanks', 'kae', 'hss', 'ka', 'sniffle', 'tj',
#      'yishun', 'hgh', 'hu', 'jordan', 'jot', 'interior', 'yin', 'ko', 'biglang', 'pv', 'w', 'ie', 'seunghyun', 'boeng',
#      'tn', 'ibm', 'bi', 'kru', 'hui', 'galen', 'jy', 'br', 'nc', 'yy', 'annie', 'skala', 'alas', 'jantsch', 'sunday',
#      'vandenberg', 'rosie', 'urbana', 'january', 'sir', 'bas', 'tseng', 'reina', 'har', 'ut', 'p', 'fm', 'aa', 'kikoku',
#      'sophie', 'ari', 'ubc', 'kc', 'keiki', 'strongsville', 'chien', 'redang', 'k', 'q', 'monterey', 'yvonne', 'glen',
#      'ephesus', 'israel', 'planescape', 'li', 'juno', 'yichao', 'ik', 'ali', 'buddha', 'mm', 'oc', 'utica', 'shelbie',
#      'des', 'zonia', 'jenna', 'et', 'zippos', 'grr', 'stanford', 'warren', 'etienne', 'ye', 'kida', 'paradise', 'sara',
#      'div', 'verac', 'ec', 'isp', 'sp', 'karen', 'serena', 'kamina', 'brolin', 'yeh', 'ey', 'leo', 'ra', 'bin', 'dt',
#      'tourette', 'paulien', 'zn', 'fi', 'esl', 'jonah', 'kj', 'ruby', 'ch', 'dm', 'ebola', 'yargh', 'ab', 'chiado', 'z',
#      'obama', 'paintallica', 'seigaku', 'au', 'yeap', 'wasp', 'pb', 'lv', 'tcf', 'misogyny', 'rodin', 'nox', 'mr', 'jd',
#      'ja', 'elizabeth', 'kongkong', 'tv', 'ianto', 'kyuhyun', 'christmas', 'kinnda', 'american', 'mccain', 'hud',
#      'zyrtec', 'lol', 'ema', 'elvis', 'un', 'aesc', 'rodolphus', 'vegetarian', 'montana', 'bmw', 'hell', 'michelle',
#      'kyria', 'yu', 'mg', 'kang', 'shave', 'g', 'google', 'gio', 'savannah', 'zeke', 'cnbc', 'bupkus', 'lyn', 'taiwan',
#      'vp', 'ette', 'msi', 'azn', 'uncletang', 'e', 'janne', 'sept', 'devon', 'john', 'angelica', 'iu', 'danshi', 'pdf',
#      'south', 'enough', 'anne', 'vm', 'lj', 'gh', 'harley', 'indians', 'sheldon', 'columbia', 'von', 'ap', 'oy', 'bev',
#      'windows', 'phien', 'lau', 'nms', 'hiv', 'georgiana', 'boooo', 'tang', 'amanda', 'manhattan', 'pe', 'owen',
#      'kelso', 'est', 'rt', 'kianna', 'mt', 'befoe', 'wonky', 'mazda', 'janie', 'kelly', 'cy', 'mell', 'aj', 'rhan',
#      'va', 'ea', 'soi', 'rien', 'tamara', 'heba', 'f', 'jw', 'ueshima', 'l', 'african', 'acne', 'stevi', 'mb', 'gasp',
#      'ft', 'jpg', 'zondervan', 'hugo', 'zia', 'daniel', 'huntington', 'homophobia', 'jq', 'hallelujah', 'pa', 'bp',
#      'wbc', 'ambien', 'mug', 'monica', 'siya', 'kfc', 'kandy', 'ick', 'utopia', 'lukus', 'jesus', 'jp'}
# def cmp(x, y):
#     if x[3] > y[3]:
#         return 1
#     if x[3] < y[3]:
#         return -1
#     else:
#         return 0
# # for word in c:
# #     a[word] = word
# # print(a)
#
# count_1 = 0
# count_0 = 0
# count_fu1 = 0
#
# path = '../../data/to_cause_effect_with_label/icw/to_cue_4.npy'
# now_data_look = np.load(path, allow_pickle=True)
# print(len(now_data_look))
# for word in now_data_look:
#     if word[2] == 1:
#         count_1 += 1
#     elif word[2] == -1:
#         count_fu1 += 1
#         word[0], word[1] = word[1], word[0]
#     else:
#         count_0 += 1
#         continue
#     first_sentence = sorted(word[0], key=functools.cmp_to_key(cmp))
#     second_sentence = sorted(word[1], key=functools.cmp_to_key(cmp))
#     print(' '.join([words[0] for words in first_sentence]))
#     print(' '.join([words[0] for words in second_sentence]))
#     print(word[2])
# print(count_1, count_fu1, count_0)


import tensorflow as tf
import numpy as np
import torch
print(torch.__version__)

# 选出每一行的最大的前两个数字
# 返回的是最大的k个数字，同时返回的是最大的k个数字在最后的一个维度的下标
import tensorflow as tf

dropout = tf.placeholder(tf.float32)
x = tf.Variable(tf.ones([10, 10]))
y = tf.nn.dropout(x, dropout)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(sess.run(x, feed_dict={dropout: 0.4}))
print(sess.run(y, feed_dict={dropout: 0.4}))
a = [('which_nsubj', 10635), ('that_nsubj', 7373), ('to_aux', 175507), ('when_advmod', 154204), ('as_mark', 120624), ('if_mark', 89488), ('because_mark', 82301), ('while_mark', 41121), ('so_mark', 33849), ('since_mark', 29909), ('before_mark', 26739), ('like_mark', 24392), ('for_mark', 20176), ('until_mark', 19282), ('even_advmod', 17094), ('after_mark', 17008), ('than_mark', 16202), ('just_advmod', 12725), ('where_advmod', 10033), ('although_mark', 9922), ('though_mark', 9434), ('once_mark', 8137), ('so_advmod', 7446), ('only_advmod', 3986), ('unless_mark', 3526), ('whenever_advmod', 2902), ('that_mark', 2118), ('then_advmod', 2117), ('cause_mark', 1857), ('till_mark', 1763), ('still_advmod', 1216), ('whether_mark', 1121), ('why_advmod', 1063), ('once_advmod', 981), ('ever_advmod', 940), ('cuz_mark', 933), ('with_mark', 911), ('how_advmod', 895), ('now_advmod', 833), ('except_mark', 699), ("'cause_mark", 654), ('whilst_mark', 616), ('almost_advmod', 584), ('whereas_mark', 582), ('wherever_advmod', 567), ('right_advmod', 518), ('soon_advmod', 508), ('later_advmod', 493), ('thus_advmod', 490), ('long_advmod', 475), ('also_advmod', 471), ('always_advmod', 457), ('however_advmod', 408), ('already_advmod', 297), ('of_advmod', 282), ('sometimes_advmod', 277), ('again_advmod', 243), ('instead_advmod', 241), ('often_advmod', 217), ('usually_advmod', 206), ('much_advmod', 204), ('lest_mark', 203), ('perhaps_advmod', 199), ('in_mark', 198), ('well_advmod', 190), ('more_advmod', 180), ('first_advmod', 176), ('til_mark', 172), ('as_advmod', 170), ('thereby_advmod', 167), ('ago_advmod', 143), ('far_advmod', 141), ('somehow_advmod', 127), ('otherwise_advmod', 126), ('anywhere_advmod', 117), ('anytime_mark', 107), ('at_mark', 100)]

fonction_word_list = ['after', 'often', 'at', 'sometimes', 'still', 'whether', 'as', 'because', 'of', 'instead', 'cuz', 'soon', 'where', 'whilst', 'wherever', 'which', 'although', 'unless', 'far', 'while', 'right', 'to', 'long', 'always', 'than', 'for', 'much', 'whenever', 'however', 'whereas', 'that', 'well', 'also', 'since', 'otherwise', 'again', 'ago', 'lest', 'cause', 'when', 'only', 'ever', 'except', 'somehow', "'cause", 'then', 'already', 'though', 'thus', 'perhaps', 'until', 'anywhere', 'so', 'with', 'later', 'usually', 'first', 'in', 'more', 'like', 'til', 'why', 'just', 'before', 'thereby', 'anytime', 'even', 'if', 'now', 'how', 'till', 'almost', 'once']
print(len(fonction_word_list))


