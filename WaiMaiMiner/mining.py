from fgom import common_lib


def write_(sentence, which):
    cuts = common_lib.cut(sentence)
    output = ""

    if which == 1:
        tag = "E"
    elif which == 2:
        tag = "P1"
    elif which == 3:
        tag = "P2"
    elif which == 4:
        tag = "N1"
    elif which == 5:
        tag = "N2"
    elif which == 6:
        tag = "OT"
    else:
        return

    if which != 6:
        if len(cuts) == 1:
            output = "%s/%s" % (cuts[0], "I-" + tag)
        else:
            i = 0
            while True:
                if i == 0:
                    output = "%s/%s\t" % (cuts[i], "B-" + tag)
                elif i + 1 == len(cuts):
                    output += "%s/%s\t" % (cuts[i], "E-" + tag)
                    break
                else:
                    output += "%s/%s\t" % (cuts[i], "M-" + tag)
                i += 1
    else:
        for i in range(len(cuts)):
            output += "%s/%s\t" % (cuts[i], "OT")

    with open(common_lib.miner_hmm_user_add_corpus_filepath, "a", encoding="utf-8") as f:
        f.write("%s\n" % output.strip())


def find_pos(clause, word, start=0):
    try:
        index = clause.index(word, start)
    except ValueError:
        return -1, -1
    return index, index + len(word)


