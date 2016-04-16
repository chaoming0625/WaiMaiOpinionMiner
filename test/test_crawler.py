from WaiMaiMiner import crawler
import json


def test1():
    id = "1430806214"
    id = "1452459851"
    # id = "1438397794"
    result = crawler.open(id)

    with open("files/test_crawler.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=4, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    test1()
