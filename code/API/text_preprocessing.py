from blingfire import text_to_sentences
import re


def convert_into_sentences(lines):
    stack = []
    sent_L = []
    n_sent = 0
    for chunk in lines:
        if not chunk.strip():
            if stack:
                sents = text_to_sentences(
                    " ".join(stack).strip().replace('\n', ' ')).split('\n')
                sent_L.extend(sents)
                n_sent += len(sents)
                sent_L.append('\n')
                stack = []
            continue
        stack.append(chunk.strip())

    if stack:
        sents = text_to_sentences(
            " ".join(stack).strip().replace('\n', ' ')).split('\n')
        sent_L.extend(sents)
        n_sent += len(sents)
    return sent_L, n_sent


def fileter_content(paper):
    # 文本中的正文,我们对其进行简单的处理,首先是将换行变为空格;然后将文本进行拼接.
    # 将换行替换为空格
    # 出现两个空格代表这里有换段落的操作,则要添加一个句号.
    # paper = paper.replace('\n', ' ')

    sentence = paper.lower()
    sentence = sentence.replace('&nbsp;', ' ')
    sentence = sentence.replace('&', '')
    text = sentence.replace('nbsp', '')

    # text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
    # text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
    # text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
    # text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    # text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

    # acronym
    text = text.replace('can\'t', 'can not')
    text = text.replace('cannot', 'can not ')
    text = text.replace('what\'s', 'what is')
    text = text.replace('What\'s', 'what is')
    text = text.replace('\'ve', ' have ')
    text = text.replace('n\'t', ' not ')
    text = text.replace('i\'m', 'i am ')
    text = text.replace('\'ll', 'will')
    text = text.replace('\'re', 'are')

    # text = re.sub(r"\'d", " would ", text)

    # text = re.sub(r" e mail ", " email ", text)
    # text = re.sub(r" e \- mail ", " email ", text)
    # text = re.sub(r" e\-mail ", " email ", text)
    # text = re.sub(r",000", '000', text)
    # text = re.sub(r"\'s", " ", text)

    # spelling correction
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" 9 11 ", " 911 ", text)
    # text = re.sub(r" j k ", " jk ", text)
    # text = re.sub(r" usa ", " america ", text)
    # text = re.sub(r" us ", " america ", text)
    # text = re.sub(r" u s ", " america ", text)
    # text = re.sub(r" U\.S\. ", " america ", text)
    # text = re.sub(r" US ", " america ", text)
    # text = re.sub(r" American ", " america ", text)
    # text = re.sub(r" America ", " america ", text)
    # text = re.sub(r" quaro ", " quora ", text)
    # text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
    # text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
    # text = re.sub(r"the european union", " eu ", text)
    # text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = text.replace('?', ' ? ')
    text = text.replace('*', '.  ')
    text = text.replace('+', '+ ')
    text = text.replace("'", '  ')
    text = text.replace('-', ' - ')
    text = text.replace('/', ' / ')
    text = text.replace('\\', ' \ ')
    text = text.replace('=', ' = ')
    text = text.replace('^', ' ^ ')
    text = text.replace(':', ' : ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')

    text = text.replace('!', ' ! ')
    text = text.replace('"', ' " ')
    text = text.replace('|', ' | ')
    text = text.replace(';', ' ; ')
    text = text.replace(')', ' ) ')
    text = text.replace('(', ' ( ')

    text = text.replace('{', ' { ')
    text = text.replace('}', ' } ')
    text = text.replace('[', ' [ ')
    text = text.replace(']', ' ] ')
    # sentence = sentence.replace('@', ' @ ')
    # sentence = sentence.replace('#', ' # ')
    # sentence = sentence.replace('$', ' $ ')
    # sentence = sentence.replace('%', ' % ')
    # symbol replacement
    # text = re.sub(r"&", " and ", text)
    # text = re.sub(r"\|", " or ", text)
    # text = re.sub(r"=", " equal ", text)
    # text = re.sub(r"\+", " plus ", text)
    # text = re.sub(r"₹", " rs ", text)  # 测试！
    # sentence = re.sub(r"\$", " dollar ", text)

    # sentence = sentence.replace('^', ' ^ ')
    # sentence = sentence.replace('&', ' & ')
    # sentence = sentence.replace('*', ' * ')
    # sentence = sentence.replace('(', ' ( ')
    # sentence = sentence.replace(')', ' ) ')

    # sentence = sentence.replace('-', ' - ')
    # sentence = sentence.replace('_', ' _ ')
    # sentence = sentence.replace('=', ' = ')
    # sentence = sentence.replace('+', ' + ')
    # sentence = sentence.replace('~', ' ~ ')
    # sentence = sentence.replace('`', ' \' ')
    # sentence = sentence.replace('\\', ' \\ ')
    # sentence = sentence.replace('|', ' | ')
    # sentence = sentence.replace('"', ' " ')
    # sentence = sentence.replace(';', ' ; ')
    # sentence = sentence.replace(':', ' : ')
    # sentence = sentence.replace('.', ' . ')
    # sentence = sentence.replace('>', ' > ')
    # sentence = sentence.replace(',', ' , ')
    # sentence = sentence.replace('<', ' < ')
    # sentence = sentence.replace('/', ' / ')

    return text


def gut_remove_heda_tail(lines):
    start_flag = ['START OF THIS PROJECT GUTENBERG EBOOK', 'START OF THE PROJECT GUTENBERG EBOOK',
                  'THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS', '["Small Print" V.12.08.93]']
    end_flag = ['END OF THIS PROJECT GUTENBERG EBOOK', 'END OF THE PROJECT GUTENBERG EBOOK',
                '**The Legal Small Print**']

    start_id = [i for i, x in enumerate(lines) if (
                (x.find(start_flag[0]) != -1) or (x.find(start_flag[1]) != -1) or (x.find(start_flag[2]) != -1) or (
                    x.find(start_flag[3]) != -1))]
    end_id = [i for i, x in enumerate(lines) if
              ((x.find(end_flag[0]) != -1) or (x.find(end_flag[1]) != -1) or (x.find(end_flag[2]) != -1))]
    # 获取文本的首部和尾部
    if len(start_id) != 0:
        start_id_max = max(start_id) + 1
    else:
        start_id_max = 0
    if len(end_id) != 0:
        end_id_min = min(end_id) - 1
    else:
        end_id_min = len(lines)
    # 获得文章正文
    choose_lines = lines[start_id_max:end_id_min + 1]

    return choose_lines
