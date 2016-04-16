import re
import json
import requests
from WaiMaiMiner import classifier

"""
First page url:
http://waimai.baidu.com/waimai/user/address/select?display=json&_=1460210537402

1, get the number of the comment page
2, get the request json
"""


class Crawler:
    def __init__(self):
        self.base_url = "http://waimai.baidu.com/waimai/comment/getshop?display=" \
                        "json&shop_id=%s&page=%s&count=60"

        self.shop_id = None
        self.maxent = classifier
        self.page_num = 1

        self.info = {}

    def crawl(self, url=None, shop_id=None):
        # get the shop id
        self._get_shop_id(url, shop_id)

        # get the fist page comment
        i = 0
        while i < self.page_num:
            self._get_json_request(self.base_url % (self.shop_id, i + 1))

            i += 1

        # init the page_num variable and filter
        self.page_num = 1
        self._filter()

        return self.info

    def _filter(self):
        """
        get the useful comment, get rid of rubbish comment.

        We use the maximum entropy to classify the polarity of the comment.
        maxEntClassifier's parameters are wrote into the file
        """
        for i, sentence in enumerate(self.info["content"]):
            rubbish_comment = False

            if self._is_english(sentence):
                rubbish_comment = True
            elif self._is_numeric(sentence):
                rubbish_comment = True
            elif self._is_too_short(sentence):
                rubbish_comment = True
            elif self._is_word_repeat(sentence):
                rubbish_comment = True

            if rubbish_comment:
                if self.info["score"][i] >= 4:
                    self.info["rubbish_comment_id"].append((i, 1))
                else:
                    self.info["rubbish_comment_id"].append((i, 0))

            else:
                result = self.maxent.classify(sentence)
                if self.info["score"][i] >= 4:
                    if result == 1:
                        self.info["useful_comment_id"].append((i, 1))
                    else:
                        self.info["useful_comment_id"].append((i, 3))
                else:
                    if result == 0:
                        self.info["useful_comment_id"].append((i, 0))
                    else:
                        self.info["useful_comment_id"].append((i, 2))

    @staticmethod
    def _is_too_short(sentence):
        """judge if the f_waimai is too short or the chinese characters are few
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        if len(sentence) < 5:
            return True
        if len(re.findall(r'[\u4e00-\u9fa5]', sentence)) <= len(sentence) * 0.4:
            return True
        return False

    @staticmethod
    def _is_numeric(sentence):
        """judge if the f_waimai's characters are all or almost numbers
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        match = re.findall("\d+", sentence)
        if match is not None and sum([len(m) for m in match]) >= len(sentence) * 0.75:
            return True
        return False

    @staticmethod
    def _is_english(sentence):
        """judge if the f_waimai's characters are all English
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        match = re.findall("[a-zA-Z]+", sentence)
        if match is not None and sum([len(m) for m in match]) >= len(sentence) * 0.75:
            return True
        return False

    @staticmethod
    def _is_word_repeat(sentence):
        """check if the f_waimai is always the repeat word
            :param sentence: a f_waimai
            :return: True or False
            """
        repeat_words, length = [], 0
        for word in sentence:
            times = sentence.count(word)
            if times >= 4 and word not in repeat_words:
                repeat_words.append(word)
                length += times
        if length > len(sentence) / 2:
            return True
        return False

    def _get_shop_id(self, url, id):
        if url is not None:
            shop_id = re.search("\d+", url)
            if shop_id is None:
                raise ValueError("Bad url")

            self.shop_id = shop_id.group()

        elif id is not None:
            self.shop_id = id

        else:
            raise ValueError("Bad url")

    def _get_json_request(self, url):
        try:
            result = requests.get(url)
        except requests.ConnectionError:
            raise ValueError("Bad url")

        content = json.loads(result.text)
        result = content["result"]
        if self.page_num == 1:
            self._get_initial_info(result)

        content = result["content"]

        for a_json in content:
            self._get_a_json_info(a_json)

    def _get_initial_info(self, result):
        # get the page number
        self.page_num = result["comment_num"] // 60 + 1

        # get the average score information
        average_score = {}
        average_score["average_dish_score"] = float(result["average_dish_score"])
        average_score["average_service_score"] = float(result["average_service_score"])
        average_score["average_score"] = float(result["average_score"])
        self.info["average_score"] = average_score

        # get the score detail
        self.info["score_detail"] = result["score_detail"]

        # get the weeks score
        weeks_score = {}
        for key, value in result["weeks_score"].items():
            weeks_score[key] = float(value)
        self.info["weeks_score"] = weeks_score

        # get the recommend dished
        self.info["recommend_dishes"] = result["recommend_dishes"]

        # get the comment num
        self.info["comment_num"] = result['comment_num']

        # initialize the self.info variable
        self.info["content"] = []
        self.info["cost_time"] = []
        self.info["service_score"] = []
        self.info["dish_score"] = []
        self.info["sfrom"] = []
        self.info["score"] = []
        self.info["create_time"] = []
        self.info["arrive_time"] = []
        self.info["useful_comment_id"] = []
        self.info["rubbish_comment_id"] = []

    def _get_a_json_info(self, a_json):
        self.info["content"].append(a_json["content"])
        self.info["cost_time"].append(a_json["cost_time"])
        self.info["service_score"].append(int(a_json["service_score"]))
        self.info["dish_score"].append(int(a_json["dish_score"]))
        self.info["score"].append(int(a_json["score"]))
        self.info["sfrom"].append(a_json["sfrom"][3:] if "na-" in a_json["sfrom"] else a_json["sfrom"])
        self.info["create_time"].append(a_json["create_time"])
        self.info["arrive_time"].append(a_json["arrive_time"])


_crawler = Crawler()
crawl = _crawler.crawl


def __test1():
    shop_id = "1430806214"
    shop_id = "1452459851"
    # id = "1438397794"
    crawler = Crawler()
    a = crawler.crawl(shop_id)
    pass


if __name__ == "__main__":
    pass
    __test1()
