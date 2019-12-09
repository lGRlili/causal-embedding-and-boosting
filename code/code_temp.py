# # from msworks.tools import *
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet as wn
# vnet3 = nltk.corpus.util.LazyCorpusLoader(
#     'verbnet3', nltk.corpus.reader.verbnet.VerbnetCorpusReader, r'(?!\.).*\.xml'
# )
#
#
# class BasicEventRule(object):
#     stopnouns = {'it', 'they', 'you', 'he', 'she', 'i', 'we', 'number', 'afternoon', 'self', 'today'} #| stopwords
#
#     def __init__(self, stemmer):
#         self._stemmer_ = stemmer
#
#     def stem(self, w, pos): return self._stemmer_.lemmatize(w, pos)
#
#     @classmethod
#     def judge_verb_passive(cls, ptree, node):
#         if node.pos != 'VBN':
#             return False
#         for idx in node.childs:
#             tmp = ptree.tree[idx]
#             if tmp.dep == 'auxpass':
#                 return True
#             if tmp.dep in {'prep', 'pcomp'} and tmp.text == 'by':
#                 return True
#         return False
#
#     @classmethod
#     def judge_verb_neg(cls, ptree, node):
#         for idx in node.childs:
#             current = ptree.tree[idx]
#             if current.dep == 'neg':
#                 return True
#         return False
#
#     @classmethod
#     def judge_prep4verb(cls, ptree, vnode):
#         prep = []
#         for child in vnode.childs:
#             if child < vnode.idx or abs(vnode.idx - child) > 2:
#                 continue
#             tmp = ptree.tree[child]
#             if tmp.dep in {'prep', 'pcomp'} and tmp.text in {'in', 'to'}:
#                 prep.append(child)
#         return prep
#
#     @classmethod
#     def judge_comp4verb(cls, ptree, vnode):
#         comp = []
#         for child in vnode.childs:
#             if child < vnode.idx:
#                 continue
#             tmp = ptree.tree[child]
#             if tmp.dep in {'ccomp', 'xcomp'}:
#                 comp.append(child)
#         return comp
#
#     @classmethod
#     def judge_noun_causal(cls, noun, causal_category):
#         def get_word_category(w):
#             try:
#                 subsumer = wn.synsets(w)[0]
#                 syns = [synset.name() for synset in subsumer.hypernym_paths()[0]]
#                 return {syn.split('.')[0] for syn in syns}
#             except Exception:
#                 return set()
#
#         if len(get_word_category(noun) & causal_category) > 0:
#             return True
#         return False
#
#     def judge_verb_valid(self, node):
#         stem = self.stem(node.text, 'v')
#         if node.pos.startswith('VB') and stem not in stopverbs and vnet3.classids(stem):
#             return True
#         return False
#
#     def judge_noun_valid(self, node):
#         stem = self.stem(node.text, 'n')
#         if stem in self.stopnouns or len(stem) < 3:
#             return False
#         if node.pos not in {'NN', 'NNS'} or not wn.synsets(stem):
#             return False
#         return True
#
#     def filter_nouns(self, ptree, nouns):
#         assert isinstance(nouns, list)
#         res = [n for n in nouns if self.judge_noun_valid(ptree.tree[n])]
#         return res if res else []
#
#     def _extract_pobj4noun_(self, ptree, noun_root):
#         node = ptree.tree[noun_root]
#         for child in node.childs:
#             if child < noun_root:
#                 continue
#             tmp = ptree.tree[child]
#             if tmp.dep == 'prep' and tmp.text in {'of', 'for'}:
#                 for ind in ptree.tree[child].childs:
#                     if ind < child:
#                         continue
#                     if ptree.tree[ind].dep == 'pobj' and self.judge_noun_valid(ptree.tree[ind]):
#                         return self.filter_nouns(ptree, [ind, noun_root])
#         return []
#
#     def extract_subj(self, ptree, vnode):
#         def _subj_imlp_(subj_root):
#             noun_pobj = self._extract_pobj4noun_(ptree, subj_root)
#             if noun_pobj:
#                 return noun_pobj
#             tmp = [subj_root] + ptree.tree[subj_root].childs
#             tmp.sort()
#             filtered = self.filter_nouns(ptree, tmp)
#             return filtered[-2:] if filtered else None
#
#         subj, tag = [], False
#         for idx in vnode.childs:
#             current = ptree.tree[idx]
#             if current.dep.startswith('nsubj') or current.dep.startswith('csubj'):
#                 res, tag = _subj_imlp_(idx), True
#                 if res is not None:
#                     subj.append(res)
#         return subj, tag
#
#     def extract_dobj(self, ptree, vnode):
#         def _dobj_imlp_(dobj_root):
#             noun_pobj = self._extract_pobj4noun_(ptree, dobj_root)
#             if noun_pobj:
#                 return noun_pobj
#             tmp = [dobj_root] + ptree.tree[dobj_root].childs
#             tmp.sort()
#             filtered = self.filter_nouns(ptree, tmp)
#             return filtered[:2] if filtered else None
#
#         obj, tag = [], False
#         for idx in vnode.childs:
#             current = ptree.tree[idx]
#             if current.dep == 'dobj':
#                 res, tag = _dobj_imlp_(idx), True
#                 if res is not None:
#                     obj.append(res)
#         return obj, tag
#
#     def extract_pobj4verb(self, ptree, prep_root):
#         """
#         :param prep_root:
#         :param ptree:
#         :return:
#         must extract a prep obj
#         """
#         pobj_root = []
#         for child in ptree.tree[prep_root].childs:
#             if child < prep_root:
#                 continue
#             if ptree.tree[child].dep in {'pobj', 'pcomp'}:
#                 pobj_root.append(child)
#         if not pobj_root:
#             return [], [], False
#         pobj = []
#         for ind in pobj_root:
#             noun_pobj = self._extract_pobj4noun_(ptree, ind)
#             if noun_pobj:
#                 pobj.append(noun_pobj)
#         if pobj:
#             return pobj, prep_root, True
#         for ind in pobj_root:
#             tmp = [ind] + ptree.tree[ind].childs
#             tmp.sort()
#             res = self.filter_nouns(ptree, tmp)
#             if res:
#                 pobj.append(res[:2])
#         if pobj:
#             return pobj, prep_root, True
#         return [], [], False
#
#     def _extract_event_of_verb_(self, ptree, node):
#         """
#         :param ptree:
#         :param node:
#         :return:
#         """
#         if not self.judge_verb_valid(node) or self.judge_verb_neg(ptree, node):
#             return None
#         subj, subj_tag = self.extract_subj(ptree, node)
#         if self.judge_verb_passive(ptree, node):
#             if subj:
#                 return [[], (node.idx, []), subj]
#         else:
#             prep4verb = self.judge_prep4verb(ptree, node)
#             if prep4verb:
#                 pobj, prep, pobj_tag = self.extract_pobj4verb(ptree, prep4verb[0])
#                 if pobj_tag and pobj:
#                     return [subj, (node.idx, [prep]), pobj]
#             else:
#                 dobj, dobj_tag = self.extract_dobj(ptree, node)
#                 if dobj_tag:
#                     return [subj, (node.idx, []), dobj] if dobj else None
#                 comp = self.judge_comp4verb(ptree, node)
#                 if comp:  # to be continue
#                     return None
#                 if subj:
#                     return [subj, (node.idx, []), []]
#         return None
#
#     def show_events(self, events, ptree):
#         def _show_event_impl_(event): return [[ptree.tree[i].text for i in item] for item in event]
#
#         for subj, (v, prep), obj in events:
#             stem = self.stem(ptree.tree[v].text, 'v')
#             subj = _show_event_impl_(subj[:1])
#             obj = _show_event_impl_(obj[:1])
#             return subj, (stem, [ptree.tree[ind].text for ind in prep]), obj
#
#     @classmethod
#     def judge_mention_len(cls, arg1, arg2):
#         if len(word_tokenize(arg1)) < 50 and len(word_tokenize(arg2)) < 50:
#             return True
#         return False
#
