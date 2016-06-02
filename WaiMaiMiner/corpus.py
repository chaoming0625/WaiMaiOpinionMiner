
import os
import re
from fgom import common_lib
from fgom.corpus import GetTaggedCorpus
from fgom.corpus import BootstrappingMaster


def get_tagged_corpus():
    input_filepath = "f_corpus/tags/"
    output_filename = "f_corpus/hmm_train_corpus.txt"

    corpus = GetTaggedCorpus(input_filepath, output_filename)
    corpus.run()


def bootstrapping():
    bootstrapping_filename = "f_corpus/hmm_bootstrapping_corpus.txt"
    origin_tag_filename = "f_corpus/hmm_train_corpus.txt"

    bootstrap = BootstrappingMaster(bootstrapping_filename, origin_tag_filename)
    bootstrap.run()


def deal1():
    path = 'f_corpus/tags/'
    for file in os.listdir(path):
        with open(os.path.join(path, file), encoding='utf-8') as readf:
            lines = [line.strip() for line in readf.readlines()]

        with open(os.path.join(path, file), 'w', encoding='utf-8') as writef:
            runout = ''
            line_no = 0
            while line_no < len(lines):
                if lines[line_no] == '' and runout:
                    writef.write("%s\n" % runout.strip())
                    runout = ''

                elif lines[line_no]:
                    runout += "%s\t" % lines[line_no]

                line_no += 1


def deal2():
    default = ''
    path = 'f_corpus/tags/'
    re_chr = re.compile('[a-zA-Z0-9:\u4e00-\u9fa5]')

    for file in os.listdir(path):
        # get all lines
        with open(os.path.join(path, file), encoding="utf-8") as readf:
            lines = [line.strip() for line in readf.readlines()]

        with open(os.path.join(path, file), 'w', encoding="utf-8") as writef:
            for line in lines:
                splits = common_lib.re_space_split.split(line.strip())
                words, tags = [], []
                for a_split in splits:
                    if "/" in a_split:
                        try:
                            word, tag = a_split.split("/")
                        except ValueError:
                            raise ValueError("Tag wrong! System cannot recognize it.%s" % writef.name)
                        words.append(word)
                        tags.append(tag.upper())
                i = 0
                punc = False
                runout = ''
                while i < len(words):
                    if re_chr.match(words[i]) is None:
                        runout += "%s/%s\t" % (words[i], tags[i])
                        punc = True

                    elif punc:
                        writef.write("%s\n" % runout.strip())
                        runout = "%s/%s\t" % (words[i], tags[i])
                        punc = False

                    else:
                        runout += "%s/%s\t" % (words[i], tags[i])

                    i += 1

                if runout:
                    writef.write("%s\n" % runout.strip())


if __name__ == "__main__":
    pass
    # get_tagged_corpus()
    # bootstrapping()
    # deal1()
    # deal2()

