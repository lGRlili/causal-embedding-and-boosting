import spacy
sig_str = 'he carefully got out of bed , so as not to wake essie.'
sig_str = 'cutting up the apples, so as to cook them '
sig_str = 'recently ,   i was going from los angeles to minneapolis with a stop in denver.  '
sig_str = 'the regents had been waiting until after the investigation to do her job review'
sig_str = 'there is also an american going off ultram tramadol, after products to display a list of overdose of ultram ingredients'
sig_str = 'I am so tired that I want to sleep right now.  '
sig_str = "My sister is so young that she can't go to school."
sig_str = ' I get up early so that I can catch the early bus'
sig_str = 'For the whole world to catch up with American levels of car ownership, the global fleet would have to quadruple'
sig_str = ' just lifting the stockpot and filmed her forehead in sweat ,wiping down the hob'
sig_str = "Thanks to Kris at Pygma for \n converting his creation \n to this nice souvenir.   \n \n Just for seplit . \n \n Now, on with this week's story...       .    The last month has been hectic. Turbo charged. Lot's of work because I was learning from Tim, my partner in crime. This hasn't been helped by the intense pressure in town due to the political transition coming to an end.\n"


print(sig_str.split('Just for seplit .'))

sig_str = sig_str.lower()


print(sig_str.lower())

nlp = spacy.load('en_core_web_sm')
nlp.remove_pipe('ner')
ddoc = nlp(sig_str)
# doc_list.append(ddoc)
for sent in ddoc.sents:
    sentences = []
    print(sent)
    print('word'.rjust(11), 'word lemma'.rjust(11), 'word pos'.rjust(11), 'relationship'.rjust(11),
          'father_word'.rjust(11), 'fa_word pos'.rjust(11), 'id'.rjust(3), '子节点')
    for token in sent:
        print(token.text.rjust(11), token.lemma_.rjust(11), token.pos_.rjust(11), token.dep_.rjust(11),
              token.head.text.rjust(11), token.head.pos_.rjust(11), str(token.i).rjust(3),
              [child.i for child in token.children],[child.i for child in token.lefts],
              [child.i for child in token.rights],[child.i for child in token.ancestors])
    print('-------')