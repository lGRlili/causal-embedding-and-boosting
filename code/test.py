# # 连词:
# # 36 [('mark', 276139), ('advmod', 119606), ('advcl', 91083), ('aux', 85239), ('nsubj', 9133), ('neg', 3105), ('det', 2646), ('prep', 1893), ('poss', 1739), ('dobj', 1341), ('amod', 1095), ('auxpass', 580), ('compound', 505), ('cc', 505), ('intj', 427), ('npadvmod', 324), ('pobj', 323), ('nummod', 302), ('nsubjpass', 288), ('attr', 232), ('predet', 225), ('dep', 222), ('preconj', 190), ('nmod', 174), ('quantmod', 70), ('expl', 59), ('csubj', 40), ('punct', 38), ('meta', 29), ('ccomp', 12), ('dative', 10), ('parataxis', 9), ('acl', 7), ('csubjpass', 1), ('acomp', 1), ('case', 1)]
# # 699
# # 其中因果的有230wpair,总共占总mark647w中的pair的0.362
# # 其中因果的有310wpair,总共占总mark1031w的pair的0.300
# # 其中因果的有510wpair,总共占总mark2543w的pair的0.200
# mark_bok = [
#     ('as', 3541652), ('if', 1948978), ('because', 704756), ('while', 577535), ('before', 561335), ('for', 412076),
#     ('until', 386317), ('so', 348489), ('like', 335183), ('since', 264864), ('than', 253582), ('though', 240781),
#     ('after', 217382), ('once', 138993), ('although', 132440), ('unless', 68039), ('that', 53409),
#     ('whether', 25772), ('till', 23831), ('with', 20782), ('whereas', 11383), ('whilst', 9602), ('except', 7064),
#     ('lest', 6781), ('cause', 4269), ('in', 2421), ('anytime', 1746), ("'cause", 1481), ('’cause', 1400),
#     ('without', 1231), ('cuz', 792), ('at', 736), ('about', 691), ('besides', 677), ('on', 648), ('til', 642),
#     ('which', 424), ('over', 385), ('’til', 352), ('albeit', 297), ('of', 269), ('to', 265), ('through', 237),
#     ('despite', 190), ('provided', 174), ('by', 165), ('|', 156), ('sensing', 154), ('given', 147), ('tho', 115),
#     ('from', 107), ('unlike', 96), ('für', 86), ('either', 73), ('too', 71), ('however', 70), ("'cuz", 60),
#     ('ugh', 56), ('’cuz', 55), ('both', 55), ('upon', 51), ('thereafter', 51), ('amongst', 46), ('timotha', 45),
#     ('wherewith', 41), ('scepter', 40), ('below', 40), ('’cos', 39), ('cos', 38), ('beneath', 37), ('wie', 36),
#     ('during', 33), ('into', 33), ('the', 33), ("'cos", 32), ('rather', 30), ('sayin', 30), ('now’', 27),
#     ('aif', 27), ('a', 27), ('all', 25), ('beyond', 23), ('alongside', 22), ('beside', 20), ('regardless', 20),
#     ('v', 19), ('everywhere', 19), ('near', 18), ('realise', 18), ('versus', 17), ('along', 17), ('anywhere', 17),
#     ('within', 17), ('toward', 17), ('whomever', 16), ('above', 16), ('out', 15), ('vis', 15), ('…', 14),
#     ('amy', 14), ('on—', 14), ('neith', 14), ('opposite', 13), ('bard', 13), ('against', 13),
#     ('notwithstanding', 13), ('nearest', 13), ('regarding', 13), ('an', 12), ('but', 12), ('following', 12),
#     ('off', 12), ('anto', 12), ('lilith’', 12), ('underneath', 11), ('sometime', 11), ('—then', 11), ('deb', 11)]
# mark_icw = [
#     ('as', 1440541), ('if', 1011890), ('because', 936538), ('while', 409445), ('so', 346075), ('since', 327652),
#     ('before', 316545), ('like', 304267), ('for', 220259), ('until', 215941), ('though', 196033),
#     ('after', 194708), ('than', 167200), ('although', 92708), ('once', 83733), ('unless', 35828),
#     ('cause', 35268), ('that', 31425), ('till', 21489), ('cuz', 14849), ('whether', 12029), ('with', 10954),
#     ("'cause", 8848), ('except', 7700), ('whilst', 5761), ('whereas', 5287), ('til', 2275), ('lest', 2025),
#     ('in', 2010), ('tho', 1385), ('anytime', 1200), ('at', 1152), ('’cause', 1039), ('without', 1039),
#     ('cos', 956), ('on', 871), ('which', 652), ("'cuz", 468), ('about', 453), ('to', 445), ('besides', 433),
#     ('albeit', 374), ('ugh', 370), ('over', 368), ('through', 284), ('eventhough', 252), ('by', 188),
#     ("'cos", 162), ('given', 159), ('from', 151), ('despite', 146), ('’til', 143), ("'coz", 139), ('of', 138),
#     ('sayin', 132), ('too', 128), ('versus', 109), ('during', 97), ('befor', 96), ('beause', 95),
#     ('uk', 95), ('however', 91), ('beacuse', 89), ('provided', 63), ('unlike', 62), ('either', 57),
#     ('onboard', 50), ('into', 48), ('below', 41), ('sometime', 35), ('thereafter', 34), ('wile', 32),
#     ('both', 32), ('regardless', 31), ('minus', 29), ('amongst', 27), ('along', 27), ('upon', 25), ('against', 25),
#     ('toward', 22), ('all', 21), ('’cos', 21), ('next', 21), ('around', 20), ('bein', 18),
#     ('beside', 18), ('under', 17), ('sensing', 17), ('across', 15), ('regarding', 15), ('even', 15), ('chat', 15),
#     ('tha', 15), ('rather', 14), ('everywhere', 14), ('down', 14), ('thhat', 14), ('off', 13),
#     ('out', 12), ('yet', 12), ('wherewith', 12), ('soz', 11)]
# # 2022
# mark_gut = [
#     ('as', 8394869), ('if', 3394364), ('for', 1561585), ('while', 1457713), ('If', 1362161), ('because', 1029880),
#     ('though', 894919), ('than', 869521), ('before', 857726), ('until', 695288), ('As', 687115), ('so', 629179),
#     ('till', 464187), ('since', 450701), ('after', 378252), ('although', 369520), ('that', 367060), ('unless', 248513),
#     ('While', 198350), ('whether', 114831), ('Though', 92490), ('Although', 86460), ('After', 85048), ('with', 78351),
#     ('Before', 75625), ('lest', 68237), ('whilst', 65835), ('whereas', 56432), ('Since', 52629), ('like', 45024),
#     ('once', 44658), ('Because', 43592), ('Whether', 35228), ('Unless', 18287), ('Once', 15109), ('With', 14224),
#     ('That', 13911), ('except', 12982), ('Until', 11065), ("'cause", 10418), ('Whilst', 8107), ('in', 7408),
#     ('For', 6858), ('Till', 4599), ('without', 4299), ('provided', 4064), ('Whereas', 3802), ('at', 3566),
#     ('albeit', 2560), ('’cause', 2367), ('cause', 2328), ('AS', 2325), ('of', 1995), ('IF', 1977), ('Than', 1645),
#     ('on', 1521), ('Except', 1358), ('tho', 1193), ('besides', 1063), ('above', 982), ('about', 956), ('to', 826),
#     ('an', 781), ('wherewith', 771), ('notwithstanding', 743), ('by', 736), ('over', 679), ('either', 653),
#     ('Lest', 627), ('WHILE', 621), ('Besides', 602), ('|', 536), ('through', 508), ('upon', 500), ("'Cause", 454),
#     ('In', 446), ('AFTER', 389), ('_', 358), ('Like', 344), ('ter', 338), ('from', 336), ('whither', 317), ('all', 303),
#     ('Without', 291), ('inasmuch', 288), ('amongst', 274), ('which', 269), ('til', 269), ('BEFORE', 263), ('ere', 262),
#     ('BECAUSE', 220), ('’Cause', 212), ('both', 204), ('THAN', 169), ('â\x80\x9cfor', 166), ('therewith', 165),
#     ('too', 154), ('So', 152), ('Upon', 128), ('On', 125), ('SINCE', 125), ('’cos', 124), ('thereafter', 118),
#     ('At', 105), ('beyond', 104), ('’til', 103), ('Amongst', 100), ('beneath', 95), ('rather', 95), ('below', 95),
#     ('ut', 92), ('against', 91), ('UNLESS', 88), ('sith', 88), ('near', 88), ("'cos", 85), ('Provided', 84),
#     ('ALTHOUGH', 84), ('bein', 83), ('alongside', 80), ('into', 75), ('To', 70), ('unlike', 70), ('FOR', 67),
#     ('during', 57), ('â\x80\x9cthat', 57), ('UNTIL', 56), ('out', 54), ('along', 53), ('thereof', 53), ('the', 52),
#     ('â\x80\x94for', 51), ('toward', 51), ('amidst', 51), ('unto', 50), ('supposing', 48), ('sayin', 48), ('given', 47),
#     ('however', 46), ("w'ile", 41), ('vis', 40), ('herewith', 40), ('regarding', 39), ('By', 39), ('within', 39),
#     ('between', 39), ('From', 39), ('opposite', 38), ('About', 37), ('among', 37), ('anytime', 37), ('THOUGH', 37),
#     ('beside', 37), ('forthwith', 36), ('despite', 35), ('a', 35), ("'cuz", 34), ('WHEREAS', 33), ('under', 33),
#     ('atter', 33), ('merely', 32), ('ontil', 32), ('â\x80\x9cuntil', 32), ('cuz', 30), ('off', 30), ('throughout', 30),
#     ('Through', 29), ('Over', 29), ('following', 29), ('whensoever', 28), ('WHETHER', 28), ('beforehand', 28),
#     ('á', 28), ('Beside', 27), ('â\x80\x9cThat', 27), ('Claverhouse', 27), ('\x93for', 25), ('but', 25),
#     ('wherefrom', 25), ('then', 24), ('whereabout', 23), ('ab', 23), ('wherever', 22), ('wi’', 22), ('according', 21),
#     ("w'ere", 21), ('wie', 21), ('für', 21), ('nearest', 21), ('â\x80\x9cWhoever', 20), ('anywhere', 19), ('iv', 19),
#     ('thither', 18), ("'bout", 18), ('behind', 18), ('eh?’', 17), ('sceptre', 17), ('case', 17), ('’for', 17),
#     ('Cause', 16), ('Notwithstanding', 16), ('then,’', 16), ('so,’', 16), ('TILL', 16), ('THAT', 16), ('befor', 16),
#     ('thereby', 16), ('across', 16), ('verses', 16), ('av', 16), ('Among', 16), ('concerning', 16), ('ONCE', 15),
#     ('wile', 15), ('yet', 15), ('afoor', 15), ('now', 14), ('WITH', 14), ('â\x80\x9cwith', 14), ('ear', 13),
#     ('Against', 13), ('fo', 13), ('therefore', 13), ('this', 13), ('evermore', 13), ('Beyond', 13), ('ob', 13),
#     ('wanst', 13), ('Within', 12), ('wha', 12), ('Either', 12), ('â\x80\x9cUntil', 12), ('scepter', 11), ('af', 11),
#     ('—Then', 11), ('EXCEPT', 11), ('Whenever', 11), ('regardless', 11), ('â\x80\x9cWhenever', 11), ('uv', 11),
#     ('â\x80\x9cLetâ\x80\x99s', 11), ('wa’n’t', 11), ('sometime', 10), ('around', 10), ('minus', 10), ('ago,’', 10),
#     ('Underneath', 10), ("'coz", 10), ('up', 10), ('soever', 10), ('ãd', 10), ('â\x80\x98for', 10), ('en', 10),
#     ('next', 10), ('everywhere', 10), ('whenst', 10)]
# # mark = [('as', 62974), ('if', 44631), ('because', 39240), ('while', 18647), ('so', 15262), ('since', 13570),
# #         ('before', 13071), ('like', 12714), ('for', 9985), ('until', 9639), ('after', 7408), ('than', 7295),
# #         ('though', 4585), ('although', 4430), ('once', 3801), ('unless', 1745), ('cause', 1725), ('till', 1032),
# #         ('that', 1021), ('whether', 599), ('cuz', 555), ('with', 474), ('except', 305), ('whereas', 286),
# #         ('whilst', 259), ('til', 107), ('lest', 105), ('in', 94), ('anytime', 56), ('at', 52), ('cos', 45),
# #         ('without', 39), ('on', 29), ('tho', 29), ('which', 27), ('about', 20), ('ugh', 16), ('eventhough', 16),
# #         ('to', 16), ('albeit', 15), ('coz', 12), ('besides', 11), ('bc', 11)]
# advmod = [('when', 75972), ('even', 8742), ('just', 6458), ('where', 4691), ('so', 3500), ('only', 2009),
#           ('whenever', 1570), ('especially', 1406), ('then', 977), ('how', 677), ('still', 570), ('why', 520),
#           ('once', 467), ('ever', 456), ('almost', 395), ('right', 345), ('now', 338), ('as', 331), ('wherever', 324),
#           ('simply', 264), ('thus', 219), ('finally', 202), ('however', 194), ('probably', 191), ('always', 180),
#           ('mostly', 173), ('at', 170), ('exactly', 159), ('about', 156), ('also', 152), ('shortly', 150),
#           ('long', 149), ('soon', 142), ('slowly', 141), ('sometimes', 139), ('maybe', 139), ('mainly', 136),
#           ('more', 131), ('actually', 124), ('really', 124), ('instead', 121), ('too', 114), ('all', 112),
#           ('already', 108), ('later', 104), ('rather', 104), ('nearly', 103), ('perhaps', 96), ('particularly', 91),
#           ('thereby', 87), ('well', 86), ('quickly', 83), ('gently', 81), ('again', 79), ('obviously', 76),
#           ('anywhere', 76), ('barely', 75), ('apparently', 75), ('completely', 71), ('usually', 69), ('basically', 68),
#           ('often', 66), ('otherwise', 64), ('first', 64), ('somehow', 64), ('suddenly', 64), ('yet', 60),
#           ('partly', 60), ('fully', 59), ('eventually', 59), ('kind', 57), ('immediately', 56), ('very', 56),
#           ('much', 55), ('occasionally', 52), ('constantly', 51)]
# nsubj = [('which', 4710), ('i', 324), ('who', 312), ('that', 305), ('what', 221), ('all', 221), ('he', 147),
#          ('one', 124), ('both', 114), ('eyes', 91), ('it', 90), ('you', 82), ('we', 72), ('some', 71), ('each', 63),
#          ('most', 62), ('she', 60), ('whatever', 52)]
#
# conj = [
#     # ('to', 82689), ('when', 75972), ('as', 63391), ('if', 44636), ('because', 39244), ('so', 18762),
#     #  ('while', 18648), ('since', 13668), ('before', 13123), ('like', 12962), ('for', 10201), ('until', 9652),
#     #  ('even', 8747), ('after', 7998), ('than', 7295), ('which', 5899), ('where', 4737),
#     #  ('though', 4613), ('although', 4432), ('once', 4269), ('just', 6461), ('not', 2863),
#     #  ('only', 2012), ('cause', 1766), ('unless', 1745), ('whenever', 1570), ('that', 1565),
#     #  ('then', 977), ('what', 627), ('whether', 601), ('why', 520), ('who', 354), ('wherever', 325), ('whereas', 286),
#     #  ('but', 263), ('till', 1034), ('whatever', 261), ('whilst', 259), ('however', 198),
#     #  ('how', 677),  ('cuz', 559), ('still', 571), ('with', 551),
#
#     # ('ever', 456), ('at', 423), ('almost', 395), ('never', 275), ('except', 310),
#     # ('in', 336), ('thus', 219), ('about', 193), ('and', 219), ('both', 183), ('always', 180),
#     # ('also', 152), ('more', 149), ('thereby', 87), ('of', 111), ('too', 118), ('later', 104), ('rather', 104),
#     # ('anywhere', 77), ('otherwise', 65), ('somehow', 64), ('whom', 59), ('again', 79), ('either', 79),
#     # ('each', 109), ('already', 108), ('lest', 106), ('most', 92), ('such', 77), ('yet', 60), ('on', 59),
#     #
#     # ('especially', 1406), ('finally', 202), ('probably', 191), ('exactly', 159), ('shortly', 150), ('slowly', 141),
#     # ('mainly', 136), ('actually', 124), ('really', 124), ('particularly', 91), ('gently', 81), ('obviously', 76),
#     # ('barely', 75), ('apparently', 75), ('completely', 71), ('basically', 68), ('partly', 60), ('fully', 59),
#     # ('eventually', 59), ('suddenly', 64), ('immediately', 56), ('constantly', 51), ('nearly', 103),
#
#     ('the', 1279), ('a', 839), ('all', 783), ('his', 740), ('her', 450), ('now', 338), ('long', 164),
#     ('let', 257), ('my', 336), ('one', 230), ('i', 343), ('he', 154), ('she', 60), ('this', 126),
#     ('soon', 142), ('sometimes', 139), ('left', 138), ('no', 132),
#     ('eyes', 102), ('some', 138), ('two', 95), ('it', 92), ('its', 90), ('you', 87), ('we', 77), ('first', 73),
#     ('anytime', 66), ('often', 66), ('much', 63), ('there', 62), ('half', 62), ('their', 60),
#     ('kind', 57), ('very', 56), ('our', 53), ('an', 52), ('hands', 51), ('ending', 51),
# ]
#
# # 动词ing 介词 连接的状语成分
# adv = [('followed', 234), ('til', 154), ('tilting', 66), ('carrying', 110),
#        ('trying', 2674), ('looking', 2587), ('being', 1845), ('having', 1678), ('watching', 1394), ('making', 1367),
#        ('using', 1349), ('going', 1106), ('saying', 1094), ('knowing', 1060), ('taking', 1016), ('getting', 1014),
#        ('waiting', 969), ('get', 880), ('leaving', 846), ('see', 735),
#        ('doing', 724), ('feeling', 712), ('thinking', 683), ('seeing', 655), ('asking', 632),
#        ('telling', 615), ('considering', 608), ('pulling', 607), ('holding', 586), ('sitting', 576),
#        ('running', 574), ('hoping', 569), ('talking', 556), ('letting', 468), ('wondering', 433), ('turning', 432),
#        ('speaking', 540), ('giving', 523), ('walking', 518), ('is', 499), ('coming', 496), ('was', 493),
#        ('starting', 418), ('working', 408), ('staring', 408), ('moving', 391), ('had', 385),
#        ('pushing', 358), ('causing', 351), ('right', 350), ('leaning', 347), ('playing', 317), ('putting', 314),
#        ('reading', 306), ('eating', 272), ('simply', 264), ('listening', 305), ('smiling', 301), ('wearing', 299),
#        ('got', 299), ('standing', 291), ('keeping', 289), ('reaching', 260), ('shaking', 251), ('do', 250),
#        ('wanting', 246), ('laughing', 245), ('buy', 238), ('made', 232), ('pressing', 225), ('allowing', 224),
#        ('calling', 223), ('come', 219), ('meaning', 214), ('bringing', 204), ('go', 203), ('searching', 201),
#        ('showing', 200), ('rubbing', 200), ('finding', 196), ('look', 188), ('hanging', 187),
#        ('picking', 178), ('driving', 175), ('mostly', 174), ('setting', 174), ('helping', 173), ('enjoying', 172),
#        ('realizing', 171), ('kissing', 168), ('grabbing', 167), ('were', 166), ('stopping', 166), ('went', 164),
#        ('living', 163), ('glancing', 162), ('ignoring', 160), ('heading', 158), ('sleeping', 155), ('crying', 154),
#        ('sending', 154), ('closing', 153), ('raising', 150), ('forcing', 150),
#        ('opening', 147), ('catching', 146), ('screaming', 146), ('take', 143), ('well', 143),
#        ('adding', 142), ('expecting', 140), ('sliding', 140), ('lying', 139),
#        ('maybe', 139), ('make', 136), ('have', 136), ('called', 134), ('growing', 134), ('assuming', 128),
#        ('staying', 126), ('am', 125), ('breaking', 123), ('instead', 122), ('checking', 122), ('facing', 120),
#        ('figuring', 119), ('drawing', 118), ('are', 117), ('be', 116), ('lifting', 116), ('covering', 116),
#        ('hitting', 115), ('writing', 115), ('leading', 115), ('keep', 115), ('find', 115), ('resting', 114),
#        ('throwing', 113), ('rolling', 113), ('thank', 112), ('eat', 112), ('biting', 112),
#        ('pointing', 109), ('dressed', 108), ('put', 107),
#        ('missing', 107), ('following', 106), ('falling', 106), ('remembering', 105), ('dropping', 105), ('flying', 104),
#        ('wishing', 101), ('waving', 101),
#        ('slipping', 101), ('visit', 100), ('returning', 100), ('depending', 99), ('visiting', 99), ('asked', 99),
#        ('fighting', 99), ('wrapping', 99), ('placing', 98), ('explaining', 97), ('used', 97), ('drinking', 97),
#        ('dragging', 97), ('brushing', 97), ('perhaps', 96), ('s', 96), ('hearing', 95), ('creating', 94),
#        ('deciding', 94), ('licking', 93), ('cutting', 93), ('jump', 92), ('arriving', 90), ('riding', 89),
#        ('pretending', 89), ('revealing', 89), ('laying', 88), ('passing', 88), ('turned', 86), ('learning', 86),
#        ('fucking', 86), ('filled', 83), ('seeking', 83), ('quickly', 83), ('read', 82), ('d', 82), ('reminding', 81),
#        ('could', 81), ('filling', 81), ('waking', 81), ('covered', 81), ('sucking', 81), ('judging', 80), ('set', 80),
#        ('preparing', 80), ('knocking', 80), ('cleaning', 79), ('stepping', 78), ('shifting', 78),
#        ('hiding', 77), ('losing', 77), ('chatting', 77), ('becoming', 77), ('noting', 77),
#        ('kicking', 77), ('struggling', 77), ('wrapped', 76), ('click', 76), ('shooting', 76), ('crossing', 75),
#        ('check', 75), ('paying', 75),
#        ('climbing', 75), ('would', 75), ('continuing', 74), ('touching', 74), ('digging', 73),
#        ('crashing', 73), ('attempting', 73), ('tugging', 73), ('watch', 72), ('needing', 72), ('buying', 72),
#        ('lost', 72), ('breathing', 72), ('blinking', 72), ('jumping', 71), ('blowing', 71), ('should', 71),
#        ('caught', 71), ('wiping', 71), ('offering', 70), ('pick', 70), ('beginning', 70),
#        ('nodding', 70), ('stroking', 70), ('try', 69), ('posted', 69), ('usually', 69), ('na', 69), ('can', 69),
#        ('did', 68), ('tossing', 68), ('praying', 68), ('clutching', 68), ('yelling', 67),
#        ('stretching', 67), ('squeezing', 67), ('shouting', 66), ('believing', 66), ('saw', 66), ('finishing', 65),
#        ('came', 64), ('claiming', 64), ('gripping', 64), ('held', 63), ('refusing', 63), ('referring', 63),
#        ('bending', 63), ('started', 63), ('exposing', 63), ('spreading', 62), ('surrounded', 62),
#        ('found', 62), ('indicating', 61), ('ask', 61), ('providing', 61), ('acting', 60),
#        ('meeting', 60), ('resulting', 60), ('sharing', 60), ('bearing', 60), ('save', 60),
#        ('killing', 59), ('fearing', 59), ('sticking', 59),
#        ('sighing', 59), ('changing', 58), ('born', 57), ('filed', 57), ('spending', 57),
#        ('hugging', 57), ('took', 55), ('demanding', 55), ('grinning', 55),
#        ('granted', 55), ('arching', 54), ('awaiting', 53), ('joining', 53), ('studying', 53),
#        ('gasping', 53), ('panting', 53), ('has', 52), ('wandering', 52), ('discussing', 52),
#        ('rushing', 52),
#        ('locking', 52), ('slamming', 52), ('occasionally', 52), ('begging', 51), ('focusing', 51),
#        ('beating', 51), ('forgetting', 51)]
# Count = [('mark', 276139), ('advmod', 119606), ('advcl', 91083), ('aux', 85239)]
# count = 0
#
# for i in mark_icw:
#     print(i[0], i[1])
#     count += i[1]
# print(count)
# print(len(mark_gut))
#
# # 597593
import pickle
import math
import copy
from datetime import datetime
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
from collections import defaultdict
import numpy as np
import functools
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from reference.test_for_COPA import Cruve_PR
import json
import time
import csv

starts = datetime.now()


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return -1
    if x[0] < y[0]:
        return 1
    return 0

result_list = []
count = 0
with open('test_results_ori.tsv', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        count += 1
        result_list.append([float(line[1]), int(line[2]), count])
with open('test.tsv', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    count = -2
    for line in reader:
        count += 1
        if count < 0:
            continue
        result_list[count] = [result_list[count][0], result_list[count][1], result_list[count][2], line[2], line[3], line[4]]
        # result_list.append([float(line[1]), int(line[2]), count])
print(result_list)

total_congju = sorted(result_list, key=functools.cmp_to_key(cmp))

total_congjus = total_congju
print('排序最高的20名')
for i in range(20):
    print(total_congjus[i])

print('---')
print('排序2000名')
for i in range(20):
    print(total_congjus[i+2000])

print('---')
print('排序5000名')
for i in range(20):
    print(total_congjus[i+5000])

print('---')
print('最末尾5000名')
for i in range(20):
    print(total_congjus[-i-1])
# print(total_congjus[:100])
# print(total_congjus[-100:])

count_map = 0
num_true = 0
num_total = 0
for i in total_congjus:
    num_total += 1
    if i[1] == 1:
        num_true += 1
        count_map += float(num_true / num_total)
    # if i[0] < 1:
    #     break
    #     pass

print(count_map, num_true, num_total)
map_acc = float(count_map) / num_true
print(map_acc)

time.sleep(10)
