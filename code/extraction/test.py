# from collections import defaultdict
# import numpy as np
# import functools
# import sys
# print(sys.executable)
# import tensorflow as tf
# print(tf.__version__)
# # a = defaultdict(int)
# #
# # c = {'fucking', 'pd', 'nk', 'michael', 'austyn', 'emi', 'jpg', 'ap', 'quot', 'pg', 'yuy', 'ahs', 'god', 'idk', 'fff',
# #      'ti', 'da', 'p', 'uw', 'yep', 'fic', 'gf', 'aa', 'pw', 'aiba', 'lem', 'hello', 'wa', 'wax', 'ra', 'yea', 'leh',
# #      'ooh', 'sure', 'yum', 'lt', 'mhm', 'yay', 'wokay', 'wentz', 'yes', 'gosh', 'whew', 'ehm', 'hi', 'bookshelf',
# #      'khayeed', 'cj', 'bk', 'waslike', 'dunno', 'hey', 'pc', 'shakin', 'shit', 'eww', 'sw', 'ay', 'hmmmm', 'wanan',
# #      'nah', 'tbh', 'sorry', 'pillsy', 'lol', 'dj', 'oi', 'homey', 'um', 'ur', 'daniel', 'dp', 'bye', 'wc', 'yosh', 'jp',
# #      'ugh', 'ot', 'heck', 'goodness', 'bo', 'hii', 'hb', 'uk', 'ok', 'please', 'yeah', 'spinelli', 'jus', 'wew', 'wow',
# #      'melizza', 'gee', 'aha', 'ari', 'tachi', 'farewell', 'wth', 'boriiinnngg', 'la', 'ds', 'hmm', 'weaponlike', 'rp',
# #      'hehehe', 'hm', 'ass', 'tch', 'bc', 'mulder', 'tashi', 'ohkay', 'ag', 'mmm', 'adn', 'kay', 'que', 'yuuri', 'wya',
# #      'jj', 'imo', 'heather', 'fa', 'goodbye', 'darn', 'anyways', 'like', 'ah', 'er', 'aw', 'michelle', 'fuck', 'ms',
# #      'w', 'sarah', 'bah', 'lena', 'eva', 'oh', 'see', 'tada', 'yeh', 'qb', 'hcg', 'sc', 'mi', 'ru', 'stroke', 'nn',
# #      'mel', 'say', 'ba', 'man', 'nope', 'yup', 'ak', 'oj', 'ze', 'bla', 'sg', 'boy', 'dr', 'jeez', 'canst', 'fl',
# #      'quack', 'uh', 'htm', 'okay', 'whoa', 'otaku', 'toshi', 'jc', 'hahaha', 'whatcha', 'obvio', 'sh', 'atsushi', 'agh',
# #      'waiwai', 'tht', 'dear', 'saporta', 'hehe', 'gm', 'dg', 'va', 'yo', 'pj', 'nvr', 'mello', 'duh', 'hve', 'plz',
# #      'right', 'welll', 'ho', 'ahh', 'uc', 'ht', 'amen', 'jk', 'oops', 'oe', 'alright', 'ike', 'duc', 'un', 'blah',
# #      'trillian', 'saa', 'nut', 'td', 'well', 'ooooh', 'pls', 'alas', 'ls', 'dang', 'gl', 'wna', 'scouts', 'leonie',
# #      'spike', 'yaa', 'kh', 'dun', 'fr', 'ha', 'im', 'hoorah', 'esl', 'firefly', 'au', 'hp', 'hah', 'yer', 'btr', 'mike',
# #      'soo', 'asap', 'huh', 'eh', 'pto', 'anyway', 'yeeeah', 'davila', 'tb', 'quote', 'rachel', 'mug', 'damn', 'haha',
# #      'uo', 'ob', 'iw', 'dh', 'bita', 'ss', 'bi'}
# #
# # d = {'eu', 'kim', 'joe', 'pelvis', 'bastien', 'ang', 'skali', 'gj', 'cl', 'el', 'herald', 'c', 'kudryavtsev', 'hawaii',
# #      'jj', 'bc', 'luigi', 'cola', 'awwwww', 'kotetsu', 'meola', 'jim', 'naruto', 'rosi', 'eg', 'stepdad', 'ryan',
# #      'meriah', 'mp', 'maureen', 'hyung', 'yay', 'ry', 'bix', 'yogurt', 'sonia', 'san', 'mart', 'dan', 'burien', 'kz',
# #      'whedon', 'jon', 'konbanwa', 'gung', 'wt', 'wen', 'helena', 'nyc', 'microsoft', 'mei', 'u', 'meiguoren', 'woohoo',
# #      'jerald', 'minimum', 'nawab', 'ru', 'fit', 'friday', 'mcdonald', 'uruguay', 'ju', 'gd', 'philippe', 'hsiaowei',
# #      'zhoumi', 'nut', 'sober', 'ysb', 'jh', 'nichols', 'ethan', 'mac', 'ag', 'tobien', 'te', 'lo', 'engg', 'rikkai',
# #      'amazon', 'hebrew', 'dr', 'dl', 'ba', 'yo', 'walvis', 'backboard', 'comm', 'july', 'gielgud', 'de', 'heaven',
# #      'zelgadiss', 'god', 'nathan', 'sa', 'prac', 'dong', 'arbor', 'bloomington', 'domyoji', 'pu', 'nordstrom', 'ddd',
# #      'pennsylvania', 'christian', 'earl', 'villas', 'org', 'speck', 'seg', 'fa', 'grrr', 'gavin', 'sob', 'heh', 'st',
# #      'dungarvan', 'fus', 'mexican', 'feh', 'jb', 'kyo', 'caitlin', 'metallica', 'al', 'dee', 'jack', 'noah', 'xmas',
# #      'panky', 'jyoti', 'kun', 'ish', 'adriana', 'duncan', 'yoghurt', 'hae', 'jsl', 'chinese', 'vox', 'yilin', 'xara',
# #      'eww', 'nic', 'alisha', 'dw', 'ew', 'mh', 'cacie', 'devil', 'jonas', 'zig', 'vongola', 'tina', 'lewis', 'thursday',
# #      'hailie', 'mono', 'lim', 'chicago', 'tis', 'jc', 'mature', 'jo', 'mc', 'curtis', 'bobbie', 'tsukushi', 'right',
# #      'uf', 'tzu', 'monday', 'j', 'borg', 'ny', 'gw', 'mx', 'ruckus', 'wang', 'vay', 'joshua', 'lt', 'jarang', 'vb',
# #      'california', 'decaf', 'zelos', 'sk', 'il', 'ehm', 'ri', 'jekyll', 'charlie', 'itm', 'wil', 'angelo', 'urahara',
# #      'haku', 'say', 'yunsheng', 'py', 'hk', 'rowdy', 'sd', 'adora', 'jay', 'alaska', 'kl', 'urgggg', 'katamari',
# #      'cleveland', 'nh', 'la', 'sai', 'r', 'clinton', 'cheressa', 'vcs', 'jovi', 'ebay', 'yso', 'toshiro', 'peru',
# #      'pangpang', 'nessie', 'jimmy', 'xenia', 'yos', 'hrh', 'afghanistan', 'book', 'wm', 'po', 'wv', 'pp', 'sw', 'bob',
# #      'walmart', 'da', 'johan', 'yang', 'zedekiah', 'zoriy', 'widescreen', 'yaz', 'sr', 'prime', 'chih', 'democrat',
# #      'yugoslavia', 'yuki', 'epo', 'bee', 'kaj', 'onimusha', 'white', 'david', 'kou', 'zhang', 'english', 'dc', 'v',
# #      'int', 'pov', 'tegoshi', 'lu', 'hun', 'jillian', 'jazira', 'fyi', 'sasuke', 'jm', 'damon', 'ann', 'asia', 'aang',
# #      'con', 'ou', 'kubica', 'winky', 'chan', 'du', 'sarah', 'abby', 'snerk', 'mount', 'bonjour', 'cathy', 'cas',
# #      'rachel', 'evan', 'chris', 'wal', 'rh', 'kat', 'rm', 'europeans', 'yah', 'soutien', 'mao', 'ho', 'ls', 'jazeera',
# #      'eos', 'huang', 'saturday', 'vt', 'wa', 'teddy', 'ivan', 'jews', 'sux', 'swansboro', 'chang', 'nat', 'jae', 'gk',
# #      'kojima', 'merihell', 'xd', 'secret', 'hillary', 'gm', 'frank', 'tuesday', 'bush', 'wes', 'ys', 'stockholm',
# #      'babylon', 'draco', 'japanese', 'yugo', 'pw', 'pt', 'iki', 'lena', 'ching', 'becca', 'shen', 'haha', 'herzl',
# #      'wakashi', 'jen', 'abbie', 'smali', 'uaz', 'chen', 'atan', 'doh', 'katakana', 'ting', 'oclc', 'qt', 'joanne', 'jv',
# #      'troy', 'sac', 'jlpt', 'je', 'b', 'mo', 'mariana', 'ollie', 'yaya', 'wednesday', 'colombia', 'paycheck', 'pinyin',
# #      'dxa', 'caedmon', 'bible', 'zodiac', 'chanoch', 'jor', 'uk', 'darien', 'msnbc', 'mon', 'trowa', 'ld', 'vd', 'ii',
# #      'cnas', 'eac', 'september', 'dk', 'amber', 'libby', 'spawn', 'yanks', 'kae', 'hss', 'ka', 'sniffle', 'tj',
# #      'yishun', 'hgh', 'hu', 'jordan', 'jot', 'interior', 'yin', 'ko', 'biglang', 'pv', 'w', 'ie', 'seunghyun', 'boeng',
# #      'tn', 'ibm', 'bi', 'kru', 'hui', 'galen', 'jy', 'br', 'nc', 'yy', 'annie', 'skala', 'alas', 'jantsch', 'sunday',
# #      'vandenberg', 'rosie', 'urbana', 'january', 'sir', 'bas', 'tseng', 'reina', 'har', 'ut', 'p', 'fm', 'aa', 'kikoku',
# #      'sophie', 'ari', 'ubc', 'kc', 'keiki', 'strongsville', 'chien', 'redang', 'k', 'q', 'monterey', 'yvonne', 'glen',
# #      'ephesus', 'israel', 'planescape', 'li', 'juno', 'yichao', 'ik', 'ali', 'buddha', 'mm', 'oc', 'utica', 'shelbie',
# #      'des', 'zonia', 'jenna', 'et', 'zippos', 'grr', 'stanford', 'warren', 'etienne', 'ye', 'kida', 'paradise', 'sara',
# #      'div', 'verac', 'ec', 'isp', 'sp', 'karen', 'serena', 'kamina', 'brolin', 'yeh', 'ey', 'leo', 'ra', 'bin', 'dt',
# #      'tourette', 'paulien', 'zn', 'fi', 'esl', 'jonah', 'kj', 'ruby', 'ch', 'dm', 'ebola', 'yargh', 'ab', 'chiado', 'z',
# #      'obama', 'paintallica', 'seigaku', 'au', 'yeap', 'wasp', 'pb', 'lv', 'tcf', 'misogyny', 'rodin', 'nox', 'mr', 'jd',
# #      'ja', 'elizabeth', 'kongkong', 'tv', 'ianto', 'kyuhyun', 'christmas', 'kinnda', 'american', 'mccain', 'hud',
# #      'zyrtec', 'lol', 'ema', 'elvis', 'un', 'aesc', 'rodolphus', 'vegetarian', 'montana', 'bmw', 'hell', 'michelle',
# #      'kyria', 'yu', 'mg', 'kang', 'shave', 'g', 'google', 'gio', 'savannah', 'zeke', 'cnbc', 'bupkus', 'lyn', 'taiwan',
# #      'vp', 'ette', 'msi', 'azn', 'uncletang', 'e', 'janne', 'sept', 'devon', 'john', 'angelica', 'iu', 'danshi', 'pdf',
# #      'south', 'enough', 'anne', 'vm', 'lj', 'gh', 'harley', 'indians', 'sheldon', 'columbia', 'von', 'ap', 'oy', 'bev',
# #      'windows', 'phien', 'lau', 'nms', 'hiv', 'georgiana', 'boooo', 'tang', 'amanda', 'manhattan', 'pe', 'owen',
# #      'kelso', 'est', 'rt', 'kianna', 'mt', 'befoe', 'wonky', 'mazda', 'janie', 'kelly', 'cy', 'mell', 'aj', 'rhan',
# #      'va', 'ea', 'soi', 'rien', 'tamara', 'heba', 'f', 'jw', 'ueshima', 'l', 'african', 'acne', 'stevi', 'mb', 'gasp',
# #      'ft', 'jpg', 'zondervan', 'hugo', 'zia', 'daniel', 'huntington', 'homophobia', 'jq', 'hallelujah', 'pa', 'bp',
# #      'wbc', 'ambien', 'mug', 'monica', 'siya', 'kfc', 'kandy', 'ick', 'utopia', 'lukus', 'jesus', 'jp'}
# # def cmp(x, y):
# #     if x[3] > y[3]:
# #         return 1
# #     if x[3] < y[3]:
# #         return -1
# #     else:
# #         return 0
# # # for word in c:
# # #     a[word] = word
# # # print(a)
# #
# # count_1 = 0
# # count_0 = 0
# # count_fu1 = 0
# #
# # path = '../../data/to_cause_effect_with_label/icw/to_cue_4.npy'
# # now_data_look = np.load(path, allow_pickle=True)
# # print(len(now_data_look))
# # for word in now_data_look:
# #     if word[2] == 1:
# #         count_1 += 1
# #     elif word[2] == -1:
# #         count_fu1 += 1
# #         word[0], word[1] = word[1], word[0]
# #     else:
# #         count_0 += 1
# #         continue
# #     first_sentence = sorted(word[0], key=functools.cmp_to_key(cmp))
# #     second_sentence = sorted(word[1], key=functools.cmp_to_key(cmp))
# #     print(' '.join([words[0] for words in first_sentence]))
# #     print(' '.join([words[0] for words in second_sentence]))
# #     print(word[2])
# # print(count_1, count_fu1, count_0)
#
#
# import tensorflow as tf
# import numpy as np
# import torch
# print(torch.__version__)
#
# # 选出每一行的最大的前两个数字
# # 返回的是最大的k个数字，同时返回的是最大的k个数字在最后的一个维度的下标
# import tensorflow as tf
#
# dropout = tf.placeholder(tf.float32)
# x = tf.Variable(tf.ones([10, 10]))
# y = tf.nn.dropout(x, dropout)
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# print(sess.run(x, feed_dict={dropout: 0.4}))
# print(sess.run(y, feed_dict={dropout: 0.4}))
# a = [('which_nsubj', 10635), ('that_nsubj', 7373), ('to_aux', 175507), ('when_advmod', 154204), ('as_mark', 120624), ('if_mark', 89488), ('because_mark', 82301), ('while_mark', 41121), ('so_mark', 33849), ('since_mark', 29909), ('before_mark', 26739), ('like_mark', 24392), ('for_mark', 20176), ('until_mark', 19282), ('even_advmod', 17094), ('after_mark', 17008), ('than_mark', 16202), ('just_advmod', 12725), ('where_advmod', 10033), ('although_mark', 9922), ('though_mark', 9434), ('once_mark', 8137), ('so_advmod', 7446), ('only_advmod', 3986), ('unless_mark', 3526), ('whenever_advmod', 2902), ('that_mark', 2118), ('then_advmod', 2117), ('cause_mark', 1857), ('till_mark', 1763), ('still_advmod', 1216), ('whether_mark', 1121), ('why_advmod', 1063), ('once_advmod', 981), ('ever_advmod', 940), ('cuz_mark', 933), ('with_mark', 911), ('how_advmod', 895), ('now_advmod', 833), ('except_mark', 699), ("'cause_mark", 654), ('whilst_mark', 616), ('almost_advmod', 584), ('whereas_mark', 582), ('wherever_advmod', 567), ('right_advmod', 518), ('soon_advmod', 508), ('later_advmod', 493), ('thus_advmod', 490), ('long_advmod', 475), ('also_advmod', 471), ('always_advmod', 457), ('however_advmod', 408), ('already_advmod', 297), ('of_advmod', 282), ('sometimes_advmod', 277), ('again_advmod', 243), ('instead_advmod', 241), ('often_advmod', 217), ('usually_advmod', 206), ('much_advmod', 204), ('lest_mark', 203), ('perhaps_advmod', 199), ('in_mark', 198), ('well_advmod', 190), ('more_advmod', 180), ('first_advmod', 176), ('til_mark', 172), ('as_advmod', 170), ('thereby_advmod', 167), ('ago_advmod', 143), ('far_advmod', 141), ('somehow_advmod', 127), ('otherwise_advmod', 126), ('anywhere_advmod', 117), ('anytime_mark', 107), ('at_mark', 100)]
#
#
#

aaa = set()
aa = [('before', 98004), ('after', 95578),
      ('through', 76477), ('away', 74831), ('upon', 72822), ('without', 70535), ('again', 67983), ('around', 65579),
      ('ever', 57909), ('why', 57777), ('though', 53848), ('yet', 49711), ('against', 49527), ('under', 44739),
      ('while', 43189), ('once', 43034), ('already', 41778), ('between', 39913), ('probably', 39723), ('enough', 39143),
      ('else', 38630), ('first', 38233), ('almost', 37019), ('actually', 36451), ('together', 36294),
      ('until', 31838), ('rather', 31384), ('during', 28267), ('therefore', 26308), ('instead', 24461),
      ('either', 24239), ('behind', 23498), ('longer', 23286), ('sometimes', 22527), ('along', 22446), ('among', 21038),
      ('within', 21015), ('thus', 20970), ('less', 20782), ('better', 20642), ('partly', 19635), ('later', 19466),
      ('above', 18847), ('especially', 18789), ('ago', 17847), ('usually', 17613), ('near', 17509), ('whether', 17420),
      ('easily', 16819), ('finally', 16511), ('alone', 16332), ('merely', 15660), ('anyway', 15596),
      ('quickly', 15356), ('early', 15155), ('inside', 15087), ('although', 14884), ('except', 14594),
      ('outside', 14578), ('apparently', 14540), ('across', 14155), ('completely', 13942), ('nearly', 13815),
      ('towards', 13784), ('indeed', 13570), ('exactly', 13277), ('mostly', 13126), ('forward', 12368), ('kind', 11948),
      ('anymore', 11785), ('beyond', 11348), ('hard', 11214), ('immediately', 10879), ('hardly', 10467),
      ('generally', 10440), ('till', 10233), ('neither', 9971), ('suddenly', 9775), ('clearly', 9537),
      ('certainly', 9519), ('obviously', 9506), ('late', 9380), ('entirely', 9190), ('besides', 9103),
      ('otherwise', 8980), ('toward', 8973), ('fast', 8912), ('possibly', 8734), ('further', 8581), ('totally', 8508),
      ('truly', 8401), ('round', 8372), ('particularly', 8237), ('close', 8112), ('somewhat', 8101), ('slowly', 8066),
      ('fully', 7923), ('next', 7866), ('mainly', 7813), ('absolutely', 7723), ('forth', 7711), ('below', 7667),
      ('unless', 7581), ('directly', 7574), ('unto', 7564), ('perfectly', 7426), ('properly', 7266),
      ('naturally', 7228), ('somehow', 7220), ('somewhere', 7093), ('onto', 6963), ('constantly', 6918),
      ('ahead', 6873), ('apart', 6857), ('slightly', 6812), ('earlier', 6715), ('forever', 6626), ('straight', 6438),
      ('afterwards', 6339), ('eventually', 6338), ('seriously', 6206), ('carefully', 6194),
      ('unfortunately', 6187), ('extremely', 5911), ('kinda', 5834), ('least', 5749), ('anywhere', 5677),
      ('throughout', 5618), ('frequently', 5596), ('basically', 5516), ('definitely', 5510), ('aside', 5431),
      ('fairly', 5287), ('everywhere', 5164), ('despite', 5141), ('surely', 5111), ('scarcely', 5084), ('twice', 5082),
      ('barely', 5032), ('necessarily', 5028), ('matter', 4986), ('recently', 4985), ('greatly', 4940),
      ('lately', 4933), ('whenever', 4881), ('highly', 4825), ('equally', 4801), ('altogether', 4793), ('badly', 4783),
      ('beside', 4704), ('past', 4542), ('half', 4538), ('honestly', 4406), ('hopefully', 4388), ('moreover', 4380),
      ('chiefly', 4379), ('plus', 4316), ('rarely', 4297), ('beneath', 4227), ('precisely', 4227), ('hence', 4205),
      ('practically', 4147), ('best', 4142), ('readily', 4122), ('sort', 4101), ('nevertheless', 3969),
      ('secondly', 3924), ('literally', 3872), ('due', 3759), ('closer', 3758), ('normally', 3696), ('closely', 3618),
      ('per', 3612), ('sooner', 3556), ('largely', 3553), ('deeply', 3542), ('ill', 3509), ('gradually', 3506),
      ('thoroughly', 3498), ('seldom', 3494), ('amongst', 3463), ('rapidly', 3458), ('sufficiently', 3450),
      ('faster', 3419), ('freely', 3406), ('last', 3385), ('quietly', 3381), ('whereas', 3364), ('evidently', 3337),
      ('super', 3312), ('unlike', 3290), ('wholly', 3284), ('personally', 3272), ('nowhere', 3227), ('luckily', 3155),
      ('originally', 3127), ('damn', 3078), ('utterly', 3078), ('consequently', 3062), ('strongly', 2990),
      ('course', 2965), ('deep', 2965), ('thereby', 2921), ('continually', 2913), ('gently', 2817), ('elsewhere', 2813),
      ('whilst', 2812), ('farther', 2791), ('previously', 2714), ('alike', 2680), ('fortunately', 2650),
      ('likewise', 2640), ('likely', 2606), ('solely', 2597), ('firmly', 2577), ('instantly', 2566),
      ('occasionally', 2549), ('partially', 2485), ('safely', 2472), ('way', 2468), ('frankly', 2445), ('little', 2441),
      ('abroad', 2424), ('de', 2372), ('currently', 2361), ('awhile', 2325), ('relatively', 2316),
      ('essentially', 2307), ('online', 2302), ('thereof', 2284), ('formerly', 2271), ('ultimately', 2247),
      ('anti', 2235), ('anyways', 2220), ('plainly', 2208), ('physically', 2161), ('namely', 2131), ('secretly', 2100),
      ('commonly', 2096), ('high', 2087), ('presently', 2070), ('purely', 2012), ('heavily', 2004),
      ('specifically', 1983), ('loud', 1980), ('hitherto', 1955), ('deliberately', 1954), ('harder', 1952),
      ('afterward', 1941), ('comparatively', 1936), ('doubt', 1930), ('shortly', 1923), ('wherever', 1916),
      ('incredibly', 1870), ('regularly', 1868), ('distinctly', 1856), ('daily', 1852), ('differently', 1842),
      ('wide', 1814), ('sadly', 1795), ('underneath', 1790), ('exceedingly', 1787), ('happily', 1779),
      ('upstairs', 1774), ('kindly', 1770), ('accidentally', 1754), ('notwithstanding', 1741), ('strictly', 1737),
      ('terribly', 1734), ('desperately', 1729), ('ta', 1704), ('openly', 1701), ('accordingly', 1701),
      ('specially', 1699), ('real', 1695), ('automatically', 1685), ('willingly', 1679), ('n', 1669), ('nearer', 1660),
      ('tightly', 1651), ('loudly', 1638), ('regardless', 1638), ('lightly', 1636), ('downstairs', 1629),
      ('primarily', 1628), ('lest', 1620), ('therein', 1613), ('softly', 1599), ('backwards', 1586), ('via', 1571),
      ('considerably', 1570), ('technically', 1545), ('thankfully', 1517), ('correctly', 1516), ('meanwhile', 1497),
      ('sometime', 1491), ('purposely', 1486), ('mentally', 1468), ('lol', 1452), ('prior', 1451), ('south', 1428),
      ('whatsoever', 1426), ('widely', 1410), ('promptly', 1376), ('aloud', 1374), ('sure', 1370), ('inevitably', 1364),
      ('anyhow', 1353), ('rightly', 1334), ('short', 1323), ('officially', 1319), ('successfully', 1307),
      ('invariably', 1304), ('undoubtedly', 1301), ('beforehand', 1284), ('initially', 1283), ('steadily', 1264),
      ('nicely', 1259), ('second', 1258), ('effectively', 1255), ('justly', 1240), ('presumably', 1228),
      ('principally', 1228), ('newly', 1227), ('emotionally', 1225), ('north', 1218), ('severely', 1212),
      ('wherein', 1211), ('accurately', 1203), ('tight', 1188), ('strangely', 1180), ('infinitely', 1176),
      ('gladly', 1167), ('virtually', 1166), ('supposedly', 1163), ('furthermore', 1156), ('upwards', 1154),
      ('exclusively', 1153), ('wanna', 1148), ('opposite', 1147), ('everyday', 1143), ('positively', 1135),
      ('doubly', 1133), ('comfortably', 1125), ('halfway', 1090), ('lastly', 1088), ('reasonably', 1074),
      ('whereby', 1067), ('ok', 1052), ('emily', 1051), ('repeatedly', 1047), ('simultaneously', 1039),
      ('sharply', 1034), ('bitterly', 1027), ('seemingly', 1012), ('firstly', 1009), ('nearby', 1007),
      ('roughly', 1000), ('easy', 997), ('permanently', 996), ('eagerly', 986), ('silently', 984), ('earnestly', 975),
      ('publicly', 972), ('temporarily', 972), ('aboard', 963), ('hastily', 955), ('unconsciously', 954), ('xd', 952),
      ('nowadays', 952), ('upward', 948), ('peculiarly', 933), ('alongside', 929), ('wisely', 928), ('violently', 928),
      ('genuinely', 921), ('someday', 898), ('calmly', 893), ('separately', 892), ('spiritually', 892),
      ('thither', 876), ('poorly', 875), ('smoothly', 873), ('instinctively', 873), ('similarly', 865),
      ('legally', 863), ('horribly', 855), ('thirdly', 854), ('swiftly', 852), ('billy', 852), ('typically', 848),
      ('subsequently', 838), ('ashore', 835), ('amid', 835), ('wherefore', 835), ('speedily', 833), ('fine', 824),
      ('doubtless', 819), ('expressly', 798), ('overnight', 794), ('conveniently', 793), ('universally', 790),
      ('upside', 788), ('til', 785), ('vaguely', 781), ('unusually', 777), ('patiently', 776), ('intensely', 774),
      ('dearly', 771), ('randomly', 770), ('abruptly', 770), ('boldly', 766), ('beautifully', 764), ('anytime', 759),
      ('sideways', 759), ('backward', 755), ('evenly', 755), ('sincerely', 753), ('wonderfully', 752),
      ('hereafter', 743), ('awfully', 742), ('lily', 740), ('continuously', 739), ('perpetually', 739),
      ('potentially', 738), ('william', 735), ('consciously', 734), ('ordinarily', 733), ('good', 733),
      ('voluntarily', 726), ('extra', 725), ('intentionally', 720), ('quick', 720), ('painfully', 718), ('wrong', 718),
      ('ridiculously', 714), ('surprisingly', 709), ('increasingly', 707), ('inasmuch', 706), ('morally', 701),
      ('unexpectedly', 700), ('overly', 687), ('approximately', 681), ('intimately', 679), ('reluctantly', 676),
      ('faithfully', 665), ('heartily', 664), ('independently', 663), ('decidedly', 655), ('quicker', 651),
      ('nay', 651), ('securely', 648), ('everytime', 645), ('importantly', 632), ('whence', 630), ('henceforth', 628),
      ('cruelly', 624), ('politely', 615), ('privately', 615), ('deeper', 614), ('actively', 613),
      ('consistently', 612), ('effectually', 602), ('individually', 600), ('remarkably', 598), ('downward', 596),
      ('indirectly', 595), ('astray', 594), ('amidst', 591), ('keenly', 588), ('financially', 580), ('eternally', 578),
      ('louder', 575), ('blindly', 573), ('verily', 572), ('low', 570), ('higher', 570), ('en', 569), ('briefly', 566),
      ('brightly', 564), ('sa', 564), ('albeit', 562), ('live', 562), ('neatly', 561), ('gravely', 561),
      ('thereafter', 557), ('socially', 555), ('oddly', 551), ('cheerfully', 550), ('sexually', 550), ('loosely', 550),
      ('casually', 548), ('fiercely', 547), ('lower', 546), ('cautiously', 546), ('logically', 545), ('sally', 544),
      ('significantly', 541), ('full', 536), ('pleasantly', 535), ('overboard', 534), ('ere', 534), ('vs', 533),
      ('profoundly', 526), ('immensely', 523), ('wildly', 519), ('worse', 515), ('adequately', 515), ('curiously', 508),
      ('thru', 486), ('aside', 442), ('cos', 431), ('minus', 326), ('nearest', 300), ('eachother', 290),
      ('versus', 277), ('whence', 275), ('forward', 262), ('whither', 228), ('therefrom', 226), ('save', 190), ('rob', 176), ('together', 151), ('open', 123), ('verses', 122), ('bout', 118), ('wouldst', 116), ('astride', 100), ('apart', 85), ('thereupon', 58), ('whereof', 56), ('afterward', 55)]
print(len(aa))
bb = [('after', 87401), ('before', 74253), ('through', 73685), ('upon', 72768), ('without', 70424), ('against', 49527), ('under', 44227), ('while', 43157), ('around', 41023), ('between', 39913), ('though', 35972), ('until', 31835), ('during', 28265), ('among', 21037), ('within', 20277), ('along', 18302), ('behind', 17685), ('whether', 17420), ('either', 17347), ('although', 14754), ('above', 14748), ('except', 14591), ('near', 14412), ('across', 13819), ('towards', 13781), ('beyond', 11330), ('till', 10233), ('neither', 9787), ('toward', 8970), ('yet', 7947), ('unless', 7575), ('unto', 7563), ('inside', 7107), ('onto', 6963), ('na', 6404), ('back', 5720), ('throughout', 5607), ('outside', 5340), ('away', 5175), ('despite', 5137), ('once', 5082), ('besides', 4875), ('below', 4587), ('beside', 4542), ('plus', 4310), ('beneath', 4034), ('past', 3994), ('due', 3758), ('per', 3611), ('amongst', 3455), ('round', 3337), ('whereas', 3289), ('unlike', 3287), ('whilst', 2772), ('de', 2332), ('forth', 2058), ('ta', 1704), ('notwithstanding', 1683), ('lest', 1575), ('via', 1570), ('n', 1519), ('underneath', 1156), ('nearer', 964), ('xd', 884), ('amid', 834), ('opposite', 781), ('next', 716), ('inasmuch', 700), ('alongside', 652), ('nay', 651), ('aboard', 629), ('amidst', 591), ('en', 568), ('sa', 563), ('albeit', 539), ('vs', 519), ('thru', 486), ('aside', 442), ('cos', 431), ('minus', 326), ('nearest', 300), ('eachother', 290), ('versus', 277), ('whence', 275), ('forward', 262), ('whither', 228), ('therefrom', 226), ('save', 190), ('rob', 176), ('together', 151), ('open', 123), ('verses', 122), ('bout', 118), ('wouldst', 116), ('astride', 100), ('apart', 85), ('thereupon', 58), ('whereof', 56), ('afterward', 55)]
print(len(bb))

for word in aa:
    aaa.add(word[0])
print(aaa)
