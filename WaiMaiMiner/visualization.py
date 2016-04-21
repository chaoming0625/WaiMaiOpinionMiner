import matplotlib.pyplot as plt
from pylab import mpl
from random import choice

# set the font
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def score_detail(result):
    if result is not None:
        label_scores = sorted(result["score_detail"].items())

        # labels
        labels = "1分", "2分", "3分", "4分", "5分"
        # sizes
        sizes = [pair[1] for pair in label_scores]
        # explode
        the_max = max(sizes)
        the_index = sizes.index(the_max)
        explode = [0, 0, 0, 0, 0]
        explode[the_index] = 0.1
        explode = tuple(explode)
        # colors
        colors = ['yellowgreen', "gold", "lightskyblue", "lightcoral", "blueviolet"]
        # patches and texts
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct="%1.2f%%", shadow=True, startangle=0)
        plt.title("店铺整体评分分布", loc="left", fontsize=20)
        plt.axis("equal")

        plt.show()


def dish_score_detail(result):
    if result is not None:
        label_scores = result["dish_score"]

        # labels
        labels = "1分", "2分", "3分", "4分", "5分"
        # sizes
        sizes = [label_scores.count(i) for i in range(5)]
        # explode
        the_max = max(sizes)
        the_index = sizes.index(the_max)
        explode = [0, 0, 0, 0, 0]
        explode[the_index] = 0.1
        explode = tuple(explode)
        # colors
        colors = ['yellowgreen', "gold", "lightskyblue", "lightcoral", "blueviolet"]
        # patches and texts
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct="%1.2f%%", shadow=True, startangle=0)
        plt.title("商品质量评分分布", loc="left", fontsize=20)
        plt.axis("equal")

        plt.show()


def service_score_detail(result):
    if result is not None:
        label_scores = result["service_score"]

        # labels
        labels = "1分", "2分", "3分", "4分", "5分"
        # sizes
        sizes = [label_scores.count(i) for i in range(5)]
        # explode
        the_max = max(sizes)
        the_index = sizes.index(the_max)
        explode = [0, 0, 0, 0, 0]
        explode[the_index] = 0.1
        explode = tuple(explode)
        # colors
        colors = ['yellowgreen', "gold", "lightskyblue", "lightcoral", "blueviolet"]
        # patches and texts
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct="%1.2f%%", shadow=True, startangle=0)
        plt.title("配送服务评分分布", loc="left", fontsize=20)
        plt.axis("equal")

        plt.show()


def bar_auto_label(rects, suffix="分"):
    colors = ["g", "r", "c", "m", "y", "b", "chartreuse", "lightgreen", "skyblue",
              "dodgerblue", "slateblue", "blueviolet", "purple", "mediumorchid",
              "fuchsia", "hotpink", "lightcoral", "coral", "darkorange", "olive",
              "lawngreen", "yellowgreen", "springgreen", "cyan", "indigo", "darkmagenta",
              "orchid", "lightpink", "darkred", "orangered", "goldenrod", "lime", "aqua",
              "steelblue", "plum", "m", "tomato", "greenyellow", "darkgreen", "darkcyan",
              "violet", "crimson"]

    for i, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, 1.01 * height, "%s%s" % (float(height), suffix))
        color = choice(colors)
        colors.remove(color)
        rect.set_color(color)


def average_score(result):
    if result is not None:
        average_scores = result["average_score"]

        title = "各项评价平均指标"
        y_label = "分数"

        labels = ("商品质量", "配送服务", "整体评价")
        label_pos = (0, 1, 2)
        heights = (average_scores["average_dish_score"],
                   average_scores["average_service_score"],
                   average_scores["average_score"])

        plt.title(title, fontsize=20)
        plt.ylabel(y_label)
        plt.ylim(0, 5)

        plt.xticks(label_pos, labels)
        rects = plt.bar(left=label_pos, height=heights, width=0.35, align="center")
        bar_auto_label(rects)

        plt.show()


def weeks_score(result):
    if result is not None:
        weeks_score_ = result["weeks_score"]

        title = "整体评价变化趋势"
        y_label = "分数"
        labels = ("近三周总体评价", "近两周总体评价", "近一周总体评价")
        label_pos = tuple(range(len(labels)))
        heights = (weeks_score_["last_three_week"],
                   weeks_score_["last_two_week"],
                   weeks_score_["last_one_week"])
        plt.title(title, fontsize=20)
        plt.ylabel(y_label)
        plt.ylim(0, 5)
        plt.xticks(label_pos, labels)

        rects = plt.bar(left=label_pos, height=heights, width=0.35, align="center")
        bar_auto_label(rects)

        plt.show()


def s_from(result):
    if result is not None:
        sfrom = result["sfrom"]
        # title
        title = "订餐终端分布"
        # total sources
        sources = tuple(set(sfrom))
        # size
        sizes = [sfrom.count(source) for source in sources]
        # colors
        colors = ['yellowgreen', "gold", "lightskyblue", "lightcoral", "blueviolet"]
        # pie
        plt.pie(sizes, labels=sources, colors=colors,
                autopct="%1.2f%%", shadow=True, startangle=0)
        plt.title(title, loc="left", fontsize=20)
        plt.axis("equal")
        plt.show()


def recommend_dishes1(result):
    if result is not None:
        recommend_dishes = sorted(result["recommend_dishes"].items(),
                                  key=lambda dish: dish[1], reverse=True)[:20]
        title = "菜品推荐榜"
        y_label = "次数"
        labels = [dish[0] for dish in recommend_dishes]
        label_pos = tuple(range(len(labels)))
        heights = tuple([dish[1] for dish in recommend_dishes])

        plt.title(title, fontsize=20)
        plt.ylabel(y_label)
        # plt.margins(0.05)
        plt.xticks(label_pos, labels, rotation=40)

        rects = plt.bar(left=label_pos, height=heights, width=0.35, align="center")
        bar_auto_label(rects, "次")
        plt.subplots_adjust(bottom=0.2)

        plt.show()


def barh_auto_label(rects, suffix="次"):
    colors = ["g", "r", "c", "m", "y", "b", "chartreuse", "lightgreen", "skyblue",
              "dodgerblue", "slateblue", "blueviolet", "purple", "mediumorchid",
              "fuchsia", "hotpink", "lightcoral", "coral", "darkorange", "olive",
              "lawngreen", "yellowgreen", "springgreen", "cyan", "indigo", "darkmagenta",
              "orchid", "lightpink", "darkred", "orangered", "goldenrod", "lime", "aqua",
              "steelblue", "plum", "m", "tomato", "greenyellow", "darkgreen", "darkcyan",
              "violet", "crimson"]

    for i, rect in enumerate(rects):
        width = rect.get_width()
        plt.text(1.01 * width, rect.get_y(), "%s%s" % (int(width), suffix))
        color = choice(colors)
        colors.remove(color)
        rect.set_color(color)


def recommend_dishes2(result):
    if result is not None:
        recommend_dishes = sorted(result["recommend_dishes"].items(),
                                  key=lambda dish: dish[1])[-30:]
        title = "菜品推荐榜(前30)"
        x_label = "次数"
        labels = [dish[0] for dish in recommend_dishes]
        label_pos = tuple(range(len(labels)))
        heights = tuple([dish[1] for dish in recommend_dishes])

        plt.title(title, fontsize=20)
        plt.xlabel(x_label)
        plt.yticks(label_pos, labels)

        rects = plt.barh(bottom=label_pos, width=heights, alpha=0.35, align="center")
        barh_auto_label(rects)

        plt.show()


def cost_time(result):
    if result is not None:
        cost_times = result["cost_time"]

        title = "送餐时间分布"

        sources = ("非常快\n(15min内)", "比较快\n(15-30min内)", "比较慢\n(30-60min内)",
                   "很慢\n(60-100min内)", "简直无法忍受\n(大于100min)")
        sizes = [0] * len(sources)
        for a_time in cost_times:
            if a_time <= 15:
                sizes[0] += 1
            elif a_time <= 30:
                sizes[1] += 1
            elif a_time <= 60:
                sizes[2] += 1
            elif a_time <= 100:
                sizes[3] += 1
            else:
                sizes[4] += 1
        colors = ['yellowgreen', "gold", "lightskyblue", "lightcoral", "blueviolet"]
        # explode
        the_max = max(sizes)
        the_index = sizes.index(the_max)
        explode = [0, 0, 0, 0, 0]
        explode[the_index] = 0.1
        explode = tuple(explode)
        # pie
        plt.pie(sizes, labels=sources, colors=colors, explode=explode,
                autopct="%1.2f%%", shadow=True, startangle=0)
        plt.title(title, loc="left", fontsize=20)
        plt.axis("equal")
        plt.show()


def topic(result):
    from random import randint
    if result:
        fig, ax = plt.subplots()

        index = tuple(range(5))
        h1 = []
        h2 = []
        for i in range(5):
            h1.append(randint(20, 50))
            h2.append(randint(-10, -2))

        a = ax.barh(index, h1, color="r", alpha=.5)
        b = ax.barh(index, h2, color="b", alpha=.5)
        ax.set_yticks([i + 0.5 for i in index])
        ax.set_yticklabels(("服务", "份量", "配送", "味道", "其他"))
        ax.margins(0.2)
        ax.legend((a[0], b[0]), ('好评', '差评'))
        plt.show()


def _test():
    from WaiMaiMiner import crawler

    # get the crawler info
    shop_id = "1452459851"
    result = crawler.crawl(shop_id)

    score_detail(result)
    dish_score_detail(result)
    service_score_detail(result)
    average_score(result)
    weeks_score(result)
    s_from(result)
    # recommend_dishes1(result)
    recommend_dishes2(result)
    cost_time(result)


if __name__ == "__main__":
    # _test()
    topic(1)


