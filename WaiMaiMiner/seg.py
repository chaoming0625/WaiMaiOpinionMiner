import os
from oujago import seg


seg.load_userdict(os.path.join(os.path.dirname(os.path.realpath(__file__)), "f_seg/user_dict.txt"))
seg.add_word("快递", 10000)
seg.suggest_freq(("面", "太厚"))
seg.suggest_freq(("价格", "便宜"))
seg.suggest_freq(("服务", "周到"))


def cut(sentence):
    return list(seg.cut(sentence))

