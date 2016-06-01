
from fgom.corpus import GetToTagCorpus

input_filename = "files/waimai_corpus.txt"
output_filepath = "files/tag_corpus/"

corpus = GetToTagCorpus(input_filename, output_filepath)
corpus.run()

