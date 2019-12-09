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
    fonction_word_dict = {
        'never': 118492, 'then': 425136, 'now': 222673, 'quite': 26893, 'already': 60390, 'best': 4010, 'generally': 5768,
     'ultimately': 2666, 'indeed': 13312, 'naturally': 5249, 'home': 46019, 'whenever': 3534, 'how': 173953,
     'instead': 52501, 'truly': 12323, 'totally': 7951, 'ahead': 27561, 'when': 233754, 'simply': 27042, 'alone': 21563,
     'extremely': 7624, 'sometimes': 28603, 'normally': 9263, 'nt': 4314, 'also': 113473, 'more': 97703, 'often': 31048,
     'even': 134958, 'really': 76393, 'later': 57606, 'ever': 47052, 'matter': 8219, 'early': 13620, 'ok': 1176,
     'long': 43405, 'anyway': 14379, 'again': 142061, 'else': 35767, 'lily': 3902, 'first': 54745, 'completely': 22263,
     'far': 45597, 'originally': 3593, 'less': 11588, 'here': 123825, 'probably': 36013, 'nearby': 4876,
     'recently': 8284, 'soon': 50242, 'unfortunately': 8389, 'much': 57865, 'finally': 58965, 'faster': 6331,
     'luckily': 3962, 'equally': 2985, 'always': 83922, 'widely': 1883, 'basically': 4728, 'still': 154345,
     'instantly': 8131, 'pretty': 21549, 'whatsoever': 1617, 'potentially': 1082, 'personally': 4503, 'hugh': 1112,
     'effectively': 1890, 'why': 66173, 'wherever': 1530, 'definitely': 7223, 'maybe': 50422, 'late': 8251,
     'well': 126676, 'rather': 41185, 'largely': 1648, 'better': 16555, 'easily': 16031, 'actually': 35578,
     'physically': 4323, 'fully': 12837, 'therefore': 21549, 'perfectly': 7790, 'kind': 6304, 'relatively': 3441,
     'exactly': 16173, 'usually': 20052, 'anymore': 7541, 'closer': 18582, 'gently': 16950, 'hurriedly': 2085,
     'clearly': 14842, 'immediately': 28823, 'apparently': 11694, 'sure': 5964, 'twice': 8252, 'absolutely': 5719,
     'wide': 8632, 'right': 50352, 'however': 49404, 'meanwhile': 4835, 'close': 19793, 'thoroughly': 2796,
     'certainly': 13744, 'quickly': 67926, 'correctly': 1645, 'enough': 48277, 'sadly': 3456, 'safely': 3306,
     'kinda': 1462, 'roughly': 3318, 'suddenly': 43539, 'nowhere': 5970, 'solely': 1513, 'afterwards': 4306,
     'temporarily': 1421, 'little': 2941, 'earlier': 12984, 'regardless': 3864, 'least': 6929, 'likely': 4251,
     'consciously': 1307, 'primarily': 1472, 'slightly': 20192, 'thankfully': 3696, 'somehow': 15843, 'virtually': 1683,
     'perhaps': 28138, 'badly': 6039, 'repeatedly': 2553, 'prior': 2459, 'eventually': 23659, 'frequently': 2994,
     'possibly': 6089, 'initially': 3611, 'barely': 14296, 'simultaneously': 2426, 'namely': 1182, 'thus': 14078,
     'properly': 4339, 'rarely': 4622, 'almost': 48483, 'ill': 2371, 'especially': 14094, 'permanently': 1080,
     'anyways': 1030, 'lately': 3174, 'east': 1020, 'longer': 22868, 'constantly': 7018, 'honestly': 2918,
     'directly': 11324, 'closely': 4927, 'elsewhere': 1803, 'fortunately': 4521, 'fast': 15100, 'sometime': 2555,
     'readily': 1524, 'hopefully': 4320, 'super': 2056, 'somewhere': 14086, 'straight': 17665, 'hardly': 7082,
     'half': 8930, 'nearly': 17166, 'anti': 4445, 'unexpectedly': 1529, 'literally': 4804, 'doubt': 1849,
     'obviously': 12567, 'promptly': 2055, 'cautiously': 3020, 'strictly': 1227, 'further': 14303, 'reasonably': 1237,
     'anywhere': 5197, 'seriously': 6128, 'tightly': 8048, 'newly': 2961, 'practically': 3069, 'desperately': 5215,
     'hard': 29365, 'deep': 9221, 'last': 2030, 'sort': 3410, 'somewhat': 6958, 'particularly': 7191, 'greatly': 3147,
     'purely': 1245, 'seldom': 1302, 'occasionally': 5690, 'emotionally': 2071, 'frantically': 2526, 'harder': 4481,
     'daily': 2026, 'freely': 3115, 'randomly': 1246, 'upstairs': 6137, 'officially': 1869, 'overnight': 1695,
     'awhile': 2444, 'oddly': 1532, 'continually': 2196, 'technically': 1587, 'incredibly': 3569, 'beautifully': 1236,
     'frankly': 1212, 'evidently': 1623, 'freshly': 1591, 'seemingly': 3098, 'high': 6589, 'supposedly': 1805,
     'damn': 2307, 'regularly': 2667, 'forever': 9250, 'highly': 6059, 'partially': 2242, 'altogether': 1854,
     'essentially': 1808, 'vaguely': 2312, 'successfully': 2237, 'mostly': 11415, 'happily': 3307, 'everywhere': 7591,
     'way': 2142, 'second': 3252, 'grimly': 1090, 'halfway': 3676, 'surprisingly': 2644, 'entirely': 6075,
     'moreover': 2704, 'nonetheless': 1785, 'nick': 4631, 'terribly': 2039, 'tight': 5239, 'deeply': 9577,
     'publicly': 1007, 'excitedly': 1306, 'presumably': 1149, 'strongly': 2617, 'downstairs': 5509, 'precisely': 1893,
     'mainly': 2766, 'heather': 1895, 'necessarily': 2263, 'hence': 2726, 'furthermore': 2142, 'merely': 7802,
     'kelly': 3050, 'carefully': 17503, 'wildly': 2566, 'sincerely': 1148, 'shortly': 4480, 'north': 2835,
     'fairly': 5650, 'currently': 4424, 'actively': 1131, 'wherein': 1003, 'boldly': 1026, 'willingly': 1569,
     'presently': 1293, 'politely': 2804, 'softly': 7758, 'sharply': 2940, 'previously': 4101, 'severely': 1456,
     'utterly': 2439, 'abruptly': 6320, 'slowly': 45096, 'rapidly': 4619, 'surely': 7724, 'south': 3260,
     'specifically': 2502, 'backwards': 5594, 'steadily': 1748, 'sally': 4683, 'wanna': 1017, 'similarly': 1942,
     'megan': 1030, 'gradually': 4335, 'aloud': 2235, 'smoothly': 1906, 'increasingly': 2019, 'playfully': 1099,
     'upwards': 2193, 'afterward': 1854, 'overly': 1808, 'accidentally': 1780, 'overhead': 2529, 'brightly': 2407,
     'genuinely': 1773, 'strangely': 2225, 'automatically': 2727, 'intently': 2260, 'dimly': 1086,
     'significantly': 1267, 'lilly': 1438, 'loud': 4236, 'louder': 2746, 'thereof': 1877, 'secretly': 2327,
     'poorly': 1463, 'awkwardly': 2062, 'dearly': 1038, 'fine': 1591, 'short': 2288, 'kyle': 1532, 'importantly': 1912,
     'backward': 3124, 'emily': 5903, 'someday': 2550, 'quick': 1194, 'continuously': 1249, 'typically': 1982,
     'real': 2300, 'calmly': 3655, 'quietly': 13347, 'william': 6369, 'patiently': 1953, 'otherwise': 6942,
     'partly': 2498, 'alike': 1744, 'lightly': 5191, 'deeper': 2263, 'sideways': 3500, 'higher': 1210, 'holly': 2888,
     'farther': 2331, 'loosely': 1067, 'easy': 1329, 'loudly': 5477, 'mentally': 3207, 'upside': 1743, 'nate': 2193,
     'nervously': 3450, 'nevertheless': 4118, 'awake': 1108, 'dramatically': 1204, 'swiftly': 4363, 'firmly': 5868,
     'eagerly': 1935, 'heavily': 7194, 'inward': 1049, 'briefly': 3194, 'firstly': 1277, 'nicely': 1742, 'daniel': 1817,
     'ryan': 3288, 'molly': 2541, 'differently': 1975, 'frank': 1680, 'upward': 2819, 'reluctantly': 4299,
     'uncomfortably': 1119, 'curiously': 1380, 'sooner': 3781, 'neatly': 2660, 'silently': 6791, 'thereafter': 1085,
     'kindly': 2046, 'violently': 2800, 'individually': 1116, 'instinctively': 3299, 'hastily': 2352,
     'momentarily': 3041, 'secondly': 1108, 'billy': 6768, 'behold': 5201, 'accordingly': 1094, 'tenderly': 1136,
     'fiercely': 1256, 'subsequently': 1090, 'abroad': 1251, 'gingerly': 1682, 'casually': 3165, 'low': 1894,
     'tentatively': 1214, 'warmly': 1632, 'gracefully': 1280, 'seth': 2618, 'inevitably': 1401, 'broadly': 1065,
     'downward': 1488, 'consequently': 1595, 'sexually': 1206, 'openly': 2101, 'joel': 1350, 'comfortably': 1379,
     'eddie': 1763, 'undoubtedly': 1373, 'aback': 1189, 'loose': 1126, 'blindly': 1263, 'deliberately': 2493,
     'gladly': 1397, 'approximately': 1337, 'furiously': 1413, 'lower': 1202, 'spiritually': 1126, 'painfully': 1833,
     'therein': 1179, 'hesitantly': 1111, 'visibly': 1342, 'considerably': 1097, 'angrily': 2359, 'verily': 1205,
     'allen': 1141, 'commonly': 1063, 'weakly': 1309, 'proudly': 1374, 'likewise': 1866, 'thoughtfully': 1203,
     'wider': 1016, 'and': 14592725, 'at': 814799, 'through': 190816, 'by': 395268, 'on': 1009393, 'to': 4017890,
     'but': 2654062, 'of': 3142231, 'for': 969255, 'up': 753868, 'before': 131350, 'after': 188913, 'that': 814075,
     'with': 970569, 'as': 617502, 'out': 586286, 'or': 907040, 'from': 675308, 'back': 21029, 'toward': 38460,
     'atop': 2526, 'across': 58951, 'so': 89950, 'about': 286659, 'because': 145205, 'since': 47488, 'while': 58837,
     'between': 64878, 'among': 20318, 'if': 262885, 'except': 18208, 'under': 70534, 'than': 130656, 'besides': 4193,
     'like': 227481, 'down': 309565, 'near': 21017, 'during': 38951, 'either': 38996, 'whether': 30104, 'into': 433900,
     'towards': 39791, 'around': 180482, 'within': 38478, 'neither': 11252, 'against': 90385, 'off': 225298,
     'behind': 76134, 'plus': 8476, 'without': 79385, 'upon': 51023, 'n': 3647, 'na': 6230, 'underneath': 4307,
     'along': 39026, 'de': 1785, 'via': 3536, 'per': 4994, 'cause': 1418, 'whilst': 4022, 'beyond': 16584,
     'albeit': 845, 'despite': 17185, 'until': 60656, 'beside': 23293, 'onto': 50657, 'due': 6111, 'amidst': 1037,
     'till': 6378, 'though': 36663, 'although': 15395, 'versus': 719, 'round': 3316, 'inside': 30360, 'above': 23273,
     'beneath': 13674, 'throughout': 9873, 'whereas': 2298, 'once': 13303, 'aboard': 1979, 'below': 8567,
     'unless': 8187, 'away': 8721, 'cos': 601, 'yet': 4602, 'hr': 210, 'alongside': 2754, 'outside': 13239, 'thru': 820,
     'cuz': 523, 'unlike': 5805, 'opposite': 1722, 'past': 19426, 'mr': 1255, 'minus': 408, 'apart': 323, 'ta': 2686,
     'ahold': 345, 'amongst': 5074, 'forward': 394, 'aside': 1330, 'unto': 23543, 'onboard': 164, 'forth': 2358,
     'lest': 2271, 'next': 1034, 'open': 1511, 'amid': 1078, 'outta': 205, 'online': 1529, 'en': 946, 'together': 320,
     'notwithstanding': 493, 'nearest': 428, 'nearer': 306, 'allah': 287, 'save': 130, 'astride': 330, 'sm': 1187,
     'inasmuch': 212, 'stephen': 103, 'ul': 766, 'btnsendtop': 1382, 'thereupon': 107, 'verses': 614, 'believeth': 163,
     'decanter': 128, 'romulus': 109, 'magma': 108, 'arminda': 223, 'forwards': 160, 'oliver': 111, 'rowan': 131,
     'thorpe': 175, 'nimrod': 150, 'trak': 255}

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
