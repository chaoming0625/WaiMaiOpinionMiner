import re
import os


class Segment:
    def __init__(self):
        self.__char_prob = {}
        self.__len_prob = {}
        self.__words = []
        self.__words_length = 0
        self.__split_pattern = re.compile("\s+")
        dict_filepath = os.path.join(os.path.dirname(__file__), "f_seg/seg_dict.txt")
        self._load(dict_filepath)

    def _load(self, seg_dict):
        if not os.path.exists(seg_dict):
            print("Can't find the user dict file. Please check.")
            return

        # declare some temporary variable
        len_num = {}
        char_num = {}
        with open(seg_dict, encoding="utf-8") as f:
            for line in f:
                # split the line, get word and frequency
                splits = self.__split_pattern.split(line.strip())
                word = splits[0]
                freq = int(splits[1])

                # count the word length
                word_length = len(word)
                len_num[word_length] = len_num.get(word_length, 0) + 1

                # append the word
                self.__words.append(word)
                self.__words_length += 1

                # count every single char's num in split word
                for i, char in enumerate(word):
                    if char not in char_num:
                        char_num[char] = {}
                    char_num[char][i] = char_num[char].get(char, 0) + freq

        # translate the num into the probability

        # for word length probability
        total_length_num = sum(len_num.values())
        for length, num in len_num.items():
            self.__len_prob[length] = num / total_length_num

        # for char probability
        for char in char_num.keys():
            char_dict = char_num[char]
            total_num = sum(char_dict.values())
            self.__char_prob[char] = {}
            for pos, num in char_dict.items():
                self.__char_prob[char][pos] = num / total_num

        # sort the word
        self.__words.sort()

    def _binary_insert_right(self, word):
        low, high = 0, self.__words_length
        while low < high:
            middle = (low + high) // 2
            if self.__words[middle] > word:
                high = middle
            else:
                low = middle + 1
        return low

    def _exist(self, word):
        index = self._binary_insert_right(word)
        if self.__words[index - 1] == word:
            return True
        return False

    def _prefix_or_exist(self, word):
        index = self._binary_insert_right(word)
        if self.__words[index - 1] == word or self.__words[index].startswith(word):
            return True
        return False

    def _cut(self, sentence):
        length = len(sentence)
        if length < 2:
            return [sentence]

        i, j, results = 0, 1, []

        while j < length:
            char = sentence[j]

            # unregistered word
            if char not in self.__len_prob:
                self.__char_prob[char] = {}

            # probability of segment
            p1 = self.__len_prob[j - i] * self.__char_prob[char].get(j - i - 1, 0)

            # probability of not segment
            p2 = self.__len_prob[j - i + 1] * self.__char_prob[char].get(j - i, 0)

            part = sentence[i: j]
            if p1 > p2 and self._exist(part):
                results.append(part)
                i = j
            elif not self._prefix_or_exist(sentence[i: j + 1]):
                results.append(part)
                i = j

            j += 1
            if j == length:
                results.append(sentence[i:])
        return results

    def cut(self, sentence):
        if not isinstance(sentence, str):
            print("No support the type of this sentence.")
            return None
        blocks = re.split(r"([^\u4E00-\u9FA5]+)", sentence)
        results = []
        for blk in blocks:
            if re.match(r"[\u4E00-\u9FA5]+", blk):
                results.extend(self._cut(blk))
            else:
                tmp = re.split(r"([^a-zA-Z0-9+#])", blk)
                results.extend([x for x in tmp if x != ""])
        return results


_seg = Segment()
cut = _seg.cut
