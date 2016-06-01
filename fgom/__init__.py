from fgom.corpus import *
from fgom.HMM import *


def get_to_tag_corpus(input_filename, output_filepath, start=0, end=-1, gap=10):
    a = GetToTagCorpus(input_filename, output_filepath, start, end, gap)
    a.run()


def get_tagged_corpus(input_filepath, output_filename, default='OT'):
    a = GetTaggedCorpus(input_filepath, output_filename, default)
    a.run()


def bootstrapping(bootstrapping_filename, origin_tag_filename):
    bootstrap = BootstrappingMaster(bootstrapping_filename, origin_tag_filename)
    bootstrap.run()


