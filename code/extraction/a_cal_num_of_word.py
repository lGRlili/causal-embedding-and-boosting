from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from datetime import datetime
from glob import glob
import numpy as np
import pickle
import os

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


def read(input_dir, output_dir):
    print(input_dir, output_dir)
    word_frequent = defaultdict(int)
    file_list = list(sorted(glob(os.path.join(input_dir, '*'))))
    print(file_list)
    out_name = output_dir + '/word_frequent' + '.file'
    for file_name in file_list:
        txt_list = list(sorted(glob(os.path.join(file_name, '*txt'))))
        print(len(txt_list))
        for i, txt_path in enumerate(txt_list):

            original_sents = open(txt_path).readlines()
            paper = ' '.join(original_sents)
            words = paper.lower().split()
            for word in words:
                if word.encode('UTF-8').isalpha():
                    word_frequent[word] += 1
                else:
                    pass
                # word_frequent[word] += 1
            print(txt_path)
            print(len(word_frequent))
        with open(out_name, "wb") as f:
            pickle.dump(word_frequent, f)

        print("save safely")
        print_tiem()


def isalpha_para(parameter):
    parameter_new = {}
    for key, value in parameter.items():
        # if value > max_num:
        #     parameter_new[key] = value
        if key.encode('UTF-8').isalpha():
            parameter_new[key] = value
        else:
            pass
    print(len(parameter))
    print(len(parameter_new))
    return parameter_new


def merge():
    word_frequent_total = defaultdict(int)
    data_path = '../../data/'
    input_list = ['extraction/gut', 'extraction/bok', 'extraction/icw']
    for input_dir in input_list:
        in_name = data_path + input_dir + '/word_frequent' + '.file'
        print(in_name)
        with open(in_name, "rb") as f:
            word_frequent = pickle.load(f)

        # word_frequent = np.load(in_name, allow_pickle=True)
        # word_frequent = dict(word_frequent[0])
        print(type(word_frequent))

        for key in word_frequent:
            # if value > max_num:
            #     parameter_new[key] = value
            if key.encode('UTF-8').isalpha():
                word_frequent_total[key] += word_frequent[key]
            else:
                pass
        print(len(word_frequent))
        print(len(word_frequent_total))
    out_name = data_path + 'extraction/' + 'word_frequent' + '.file'
    with open(out_name, 'wb') as f:
        pickle.dump(word_frequent_total, f)


def str_id():
    input_name = data_path + 'extraction/' + 'word_frequent' + '.file'
    with open(input_name, 'rb') as f:
        word_frequent = pickle.load(f)
    str_id_word = {}
    id_str_word = {}
    word_list = set()
    for word in word_frequent:
        if word_frequent[word] > 20:
            word_list.add(word)
    word_list = list(word_list)
    print(len(word_frequent))
    print(len(word_list))
    for i, word in enumerate(word_list):
        print(i)
        str_id_word[word] = i + 1
        id_str_word[i + 1] = word
        # break
    with open(data_path + 'extraction/' + 'str_id_word' + '.file', 'wb') as f:
        pickle.dump(str_id_word, f)
    with open(data_path + 'extraction/' + 'id_str_word' + '.file', 'wb') as f:
        pickle.dump(id_str_word, f)


def fonction_word_str_id():
    # fonction_word_dict = {
    #     'if': 7267622, 'for': 4967746, 'to': 28004848, 'and': 11776719, 'so': 1586259, 'at': 2408845, 'of': 13164690,
    #     'with': 3482744, 'from': 2019906, 'out': 1625440, 'into': 1007984, 'but': 2129188, 'because': 3046210,
    #     'or': 1884088, 'unless': 383045, 'that': 4423223, 'on': 2917733, 'without': 368165, 'up': 1591283,
    #     'about': 762524,
    #     'through': 431124, 'along': 109344, 'down': 608902, 'as': 2711927, 'per': 18693, 'by': 1878747, 'like': 575168,
    #     'since': 1163378, 'while': 236487, 'than': 701623, 'off': 537428, 'de': 17478, 'before': 2167972,
    #     'though': 148830,
    #     'under': 267807, 'around': 218790, 'back': 39807, 'either': 79771, 'after': 1298525, 'among': 131338,
    #     'outside': 26073, 'minus': 1256, 'although': 54701, 'albeit': 2595, 'behind': 98323, 'tp': 264, 'once': 212844,
    #     'until': 156232, 'within': 128220, 'above': 73773, 'cuz': 18935, 'na': 24374, 'beyond': 65245,
    #     'against': 280551,
    #     'besides': 23138, 'across': 100631, 'below': 23044, 'near': 82334, 'between': 198281, 'upon': 527115, 'n': 7399,
    #     'whether': 122548, 'due': 13157, 'except': 64672, 'toward': 54099, 'despite': 21414, 'onto': 37855,
    #     'whereas': 13409, 'cause': 37188, 'during': 142170, 'versus': 1264, 'ahold': 698, 'unlike': 9640, 'via': 7486,
    #     'throughout': 25704, 'xd': 2480, 'whilst': 16509, 'cos': 3288, 'towards': 81920, 'beneath': 24925, 'yet': 46973,
    #     'rob': 1073, 'beside': 31547, 'ee': 218, 'ta': 7178, 'inside': 34748, 'vs': 3078, 'till': 73519,
    #     'neither': 38513,
    #     'oe': 70, 'w': 277, 'bu': 134, 'together': 972, 'plus': 14438, 'past': 24908, 'apart': 725, 'outta': 656,
    #     'hp': 69,
    #     'underneath': 5103, 'amongst': 20372, 'fucking': 206, 'alongside': 4168, 'bi': 173, 'ot': 401, 'ny': 327,
    #     'aboard': 5965, 'gb': 148, 'ng': 268, 'ac': 792, 'kai': 770, 'away': 31641, 'next': 4097, 'amidst': 5077,
    #     'vp': 1103, 'betwen': 21, 'abc': 57, 'round': 24781, 'forth': 19050, 'wa': 270, 'ike': 175, 'pa': 619,
    #     'pr': 456,
    #     'eachother': 686, 'ur': 993, 'u': 403, 'fr': 156, 'thru': 2068, 'en': 4150, 'v': 1506, 'tt': 338, 'az': 512,
    #     'aiba': 183, 'aside': 3798, 'ap': 369, 'fo': 966, 'gm': 119, 'nd': 374, 'moore': 113, 'rv': 55, 'online': 506,
    #     'atop': 1651, 'uni': 544, 'stephen': 118, 'befor': 107, 'ago': 256, 'ni': 346, 'st': 163, 'wes': 170, 'kl': 184,
    #     'od': 197, 'nw': 93, 'bon': 136, 'tn': 173, 'furnace': 59, 'unto': 41640, 'tb': 130, 'tx': 237, 'hr': 695,
    #     'ga': 379, 'opposite': 3776, 'forward': 2122, 'til': 573, 'ii': 827, 'lest': 9832, 'sleeve': 129, 'tu': 184,
    #     'bout': 414, 'fl': 165, 'os': 198, 'sa': 1507, 'ar': 995, 'aj': 158, 'ob': 742, 'tho': 214, 'imp': 375,
    #     'ds': 136,
    #     'ed': 145, 'ir': 385, 'forever': 33, 'nearest': 1243, 'ab': 357, 'oz': 159, 'fc': 49, 'save': 1121, 'aa': 57,
    #     'un': 1598, 'ru': 98, 'sm': 197, 'osama': 54, 'au': 117, 'ou': 84, 'mi': 908, 'unholy': 89, 'te': 640,
    #     'ocd': 63,
    #     'mr': 962, 'pg': 172, 'qc': 35, 'rowan': 57, 'ez': 630, 'lb': 188, 'af': 215, 'therefor': 163, 'gg': 294,
    #     'mk': 105, 'ft': 113, 'dr': 164, 'onboard': 170, 'cc': 79, 'forwards': 197, 'sd': 139, 'op': 87, 'tc': 48,
    #     'dc': 176, 'wiv': 98, 'pe': 137, 'pf': 100, 'la': 282, 'amid': 7675, 'less': 32, 'att': 260, 'vc': 57,
    #     'ct': 116,
    #     'notwithstanding': 12527, 'open': 1013, 'ianto': 100, 'toronto': 102, 'wid': 1643, 'adipex': 51, 'bo': 142,
    #     'brazil': 68, 'ne': 191, 'yo': 75, 'ia': 112, 'nearer': 4925, 'jd': 63, 'ow': 314, 'qb': 76, 'ji': 149,
    #     'uv': 557,
    #     'clasp': 78, 'mg': 321, 'tk': 91, 'wt': 134, 'il': 140, 'wordpress': 24, 'xi': 270, 'astride': 507, 'rom': 448,
    #     'aginst': 81, 'allah': 147, 'iv': 200, 'mn': 90, 'kp': 76, 'rl': 83, 'afterward': 308, 'wur': 86, 'uk': 124,
    #     'iz': 248, 'bridle': 52, 'andre': 486, 'lax': 45, 'foul': 56, 'dp': 60, 'nay': 5851, 'vi': 93, 'ss': 46,
    #     'stevia': 67, 'ja': 35, 'undeserving': 59, 'ze': 238, 'wouldst': 1241, 'thereupon': 479, 'verses': 582,
    #     'oprah': 248, 'rx': 68, 'outran': 55, 'mor': 94, 'le': 204, 'ffor': 57, 'thorpe': 98, 'ib': 121, 'whar': 254,
    #     'statesmanlike': 91, 'ov': 589, 'ln': 51, 'ph': 43, 'ug': 680, 'og': 132, 'times': 41, 'sep': 80, 'bmw': 40,
    #     'whither': 3431, 'uw': 31, 'rc': 39, 'io': 97, 'decanter': 124, 'cv': 42, 'gfs': 15, 'vile': 395,
    #     'herewith': 138,
    #     'iowa': 46, 'syracuse': 38, 'ic': 37, 'niya': 62, 'ul': 187, 'mohey': 11, 'leafy': 33, 'believeth': 53,
    #     'fae': 83,
    #     'oftentimes': 49, 'inward': 251, 'therefrom': 1197, 'wisher': 129, 'aright': 419, 'av': 408, 'wreath': 32,
    #     'bien': 111, 'tartar': 41, 'ef': 474, 'roundabout': 24, 'xt': 38, 'thereunto': 268, 'betwixt': 335, 'home': 52,
    #     'ffrom': 19, 'b': 18, 'later': 33, 'inasmuch': 3270, 'mikheil': 15, 'nether': 28, 'qp': 11, 'acetaminophen': 44,
    #     'lk': 35, 'whence': 2072, 'moreover': 43, 'trak': 117, 'worldliness': 19, 'gainst': 193, 'didst': 267,
    #     'canst': 352, 'whereof': 301, 'unbolt': 17, 'whereby': 51, 'qa': 25, 'wi': 18, 'di': 324, 'saith': 185,
    #     'forthwith': 52, 'ashen': 41, 'whereunto': 196, 'maketh': 44, 'tanto': 51, 'hereunto': 116, 'fide': 180,
    #     'wheresoever': 156, 'hath': 126, 'whan': 167, 'engross': 89, 'whin': 130, 'muffin': 13, 'vnto': 920,
    #     'vpon': 972,
    #     'yf': 147, 'ayenst': 46, 'widout': 186, 'togither': 90, 'aux': 56, 'thow': 47, 'agaynst': 85, 'bycause': 66,
    #     'throng': 24, 'distil': 67, 'sufferest': 33, 'lighteth': 34, 'therevnto': 32, 'uppon': 59, 'therby': 28,
    #     'wainscot': 38, 'ane': 67, 'playground': 13, 'acomplia': 14, 'yhat': 12, 'zey': 82, 'preventedfrom': 15,
    #     'seeketh': 28}
    fonction_word_dict = {'towards', 'hastily', 'precisely', 'heavily', 'perpetually', 'ago', 'independently', 'thankfully', 'decidedly', 'hardly', 'seldom', 'eternally', 'equally', 'undoubtedly', 'vaguely', 'surely', 'whilst', 'loud', 'patiently', 'first', 'particularly', 'best', 'longer', 'regardless', 'sometimes', 'per', 'lol', 'thither', 'comparatively', 'late', 'financially', 'least', 'lest', 'namely', 'wherein', 'damn', 'repeatedly', 'closer', 'astride', 'altogether', 'faithfully', 'backwards', 'minus', 'while', 'mainly', 'inside', 'consequently', 'twice', 'closely', 'kind', 'eachother', 'typically', 'low', 'carefully', 'unlike', 'automatically', 'therein', 'properly', 'unfortunately', 'thereby', 'somewhat', 'live', 'beneath', 'continually', 'rarely', 'individually', 'casually', 'rather', 'consistently', 'farther', 'shortly', 'further', 'overly', 'spiritually', 'calmly', 'literally', 'personally', 'gradually', 'intensely', 'subsequently', 'anymore', 'loosely', 'daily', 'mostly', 'likely', 'thereof', 'potentially', 'brightly', 'sincerely', 'lightly', 'south', 'full', 'nicely', 'conveniently', 'anyways', 'easy', 'violently', 'highly', 'peculiarly', 'hitherto', 'indirectly', 'wanna', 'xd', 'course', 'reluctantly', 'second', 'aloud', 'blindly', 'open', 'everyday', 'seemingly', 'sometime', 'good', 'invariably', 'fairly', 'horribly', 'earlier', 'lily', 'super', 'publicly', 'solely', 'obviously', 'ta', 'remarkably', 'forth', 'emotionally', 'totally', 'round', 'doubt', 'forever', 'fast', 'inevitably', 'outside', 'afterwards', 'lower', 'principally', 'easily', 'immensely', 'within', 'basically', 'nowhere', 'everywhere', 'relatively', 'behind', 'ordinarily', 'promptly', 'intimately', 'worse', 'anyway', 'secretly', 'currently', 'utterly', 'intentionally', 'whereof', 'honestly', 'beforehand', 'upon', 'barely', 'profoundly', 'scarcely', 'legally', 'actively', 'oddly', 'voluntarily', 'anyhow', 'wisely', 'anywhere', 'deep', 'presumably', 'vs', 'awhile', 'primarily', 'previously', 'less', 'necessarily', 'steadily', 'n', 'finally', 'perfectly', 'curiously', 'luckily', 'definitely', 'practically', 'hopefully', 'readily', 'sure', 'beyond', 'wildly', 'effectually', 'among', 'eagerly', 'quicker', 'securely', 'evidently', 'temporarily', 'amidst', 'entirely', 'someday', 'immediately', 'wherever', 'unusually', 'physically', 'incredibly', 'wrong', 'gladly', 'nearest', 'anti', 'across', 'last', 'strictly', 'henceforth', 'lately', 'effectively', 'likewise', 'possibly', 'furthermore', 'gently', 'rapidly', 'accordingly', 'firmly', 'secondly', 'exactly', 'short', 'positively', 'technically', 'somewhere', 'whereas', 'til', 'importantly', 'openly', 'exceedingly', 'regularly', 'infinitely', 'cos', 'quickly', 'formerly', 'beautifully', 'randomly', 'onto', 'overboard', 'ridiculously', 'though', 'underneath', 'near', 'wholly', 'initially', 'abruptly', 'upside', 'before', 'around', 'against', 'next', 'aside', 'freely', 'wonderfully', 'unconsciously', 'inasmuch', 'naturally', 'sa', 'earnestly', 'online', 'sooner', 'louder', 'cheerfully', 'otherwise', 'directly', 'without', 'specifically', 'truly', 'strangely', 'hereafter', 'besides', 'widely', 'successfully', 'ashore', 'merely', 'tightly', 'backward', 'sadly', 'differently', 'certainly', 'save', 'de', 'downstairs', 'comfortably', 'slowly', 'again', 'continuously', 'firstly', 'tight', 'sufficiently', 'alongside', 'verily', 'else', 'usually', 'dearly', 'boldly', 'significantly', 'way', 'supposedly', 'upstairs', 'between', 'permanently', 'justly', 'moreover', 'frequently', 'forward', 'safely', 'albeit', 'adequately', 'higher', 'considerably', 'indeed', 'unless', 'apart', 'ever', 'fine', 'via', 'why', 'anytime', 'silently', 'ill', 'accidentally', 'politely', 'whether', 'together', 'beside', 'ultimately', 'william', 'socially', 'thru', 'nowadays', 'therefrom', 'seriously', 'wouldst', 'thirdly', 'approximately', 'nearer', 'prior', 'fortunately', 'sort', 'probably', 'keenly', 'once', 'somehow', 'correctly', 'toward', 'cruelly', 'sexually', 'similarly', 'enough', 'largely', 'happily', 'cautiously', 'elsewhere', 'badly', 'partly', 'nearly', 'surprisingly', 'speedily', 'occasionally', 'until', 'gravely', 'despite', 'generally', 'neither', 'greatly', 'amid', 'newly', 'briefly', 'later', 'terribly', 'deeper', 'poorly', 'during', 'purposely', 'below', 'except', 'faster', 'roughly', 'doubly', 'partially', 'quietly', 'almost', 'close', 'north', 'pleasantly', 'everytime', 'commonly', 'yet', 'therefore', 'better', 'smoothly', 'whither', 'officially', 'straight', 'amongst', 'reasonably', 'bitterly', 'loudly', 'fully', 'high', 'half', 'above', 'desperately', 'expressly', 'constantly', 'upward', 'halfway', 'sharply', 'meanwhile', 'evenly', 'unto', 'alone', 'already', 'frankly', 'real', 'sally', 'although', 'kindly', 'lastly', 'especially', 'nearby', 'along', 'hard', 'swiftly', 'versus', 'distinctly', 'astray', 'morally', 'instead', 'overnight', 'essentially', 'kinda', 'upwards', 'nay', 'due', 'thus', 'past', 'originally', 'purely', 'privately', 'matter', 'whence', 'downward', 'exclusively', 'nevertheless', 'harder', 'plainly', 'painfully', 'clearly', 'instinctively', 'verses', 'emily', 'presently', 'willingly', 'throughout', 'away', 'eventually', 'quick', 'aboard', 'whereby', 'wide', 'universally', 'through', 'slightly', 'heartily', 'deeply', 'genuinely', 'whatsoever', 'strongly', 'hence', 'normally', 'virtually', 'neatly', 'deliberately', 'chiefly', 'afterward', 'en', 'accurately', 'softly', 'thereafter', 'alike', 'suddenly', 'unexpectedly', 'whenever', 'plus', 'till', 'wherefore', 'either', 'instantly', 'logically', 'recently', 'early', 'ok', 'absolutely', 'billy', 'separately', 'ahead', 'awfully', 'bout', 'notwithstanding', 'fiercely', 'consciously', 'little', 'completely', 'mentally', 'rightly', 'specially', 'increasingly', 'thereupon', 'actually', 'ere', 'rob', 'after', 'severely', 'apparently', 'abroad', 'extremely', 'sideways', 'doubtless', 'simultaneously', 'extra', 'opposite', 'thoroughly', 'under'}
    # fonction_word_dict = {'furthermore', 'consequently', 'twice', 'closely', }

    print(len(fonction_word_dict))
    str_id_word = {}
    id_str_word = {}
    word_list = set()
    for word in fonction_word_dict:
        word_list.add(word)
    word_list = list(word_list)
    print(len(word_list))
    for i, word in enumerate(word_list):
        print(i)
        str_id_word[word] = i + 1
        id_str_word[i + 1] = word
        # break
    print(len(str_id_word))
    with open(data_path + 'extraction/' + 'fonction_word_str_id_word' + '.file', 'wb') as f:
        pickle.dump(str_id_word, f)
    with open(data_path + 'extraction/' + 'fonction_word_id_str_word' + '.file', 'wb') as f:
        pickle.dump(id_str_word, f)


if __name__ == '__main__':
    choose = 2
    data_path = '../../data/'
    input_list = ['data_txt/icw', 'data_txt/bok', 'data_txt/gut']
    output_list = ['extraction/icw', 'extraction/bok', 'extraction/gut']
    input_path = data_path + input_list[choose]
    output_path = data_path + output_list[choose]
    # 保存过滤后只有英文单词的word
    # read(input_path, output_path)
    # # 合并后共有301w的word
    # merge()
    # str_id()
    fonction_word_str_id()
