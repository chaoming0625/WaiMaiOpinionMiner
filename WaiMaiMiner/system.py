import tkinter as tk
from threading import Thread

from WaiMaiMiner.crawler import crawl
from WaiMaiMiner import visualization
from WaiMaiMiner.mining import parse, write


def get_result():
    global result
    try:
        result = crawl(url_tv.get())
        prompt_text.set("爬取完毕\(^o^)/YES!")
    except ValueError:
        prompt_text.set("错误的URL，请重新输入。")


def analyse_button_event():
    prompt_text.set("正在爬取评论，请稍等......")
    t = Thread(target=get_result)
    t.start()


def test_tag(parse_result, sentence, type_, foreground, i, check_tv):
    j = 0
    # indent = 7
    index = 7
    # print(parse_result)
    if check_tv.get():
        for a in parse_result[type_]:
            # print(a)
            # print(index)
            index = sentence.index(a, index)
            text.tag_add("tag%d_%d" % (i, j), "%d.%d" % (i, index), "%d.%d" % (i, index + len(a)))
            text.tag_config("tag%d_%d" % (i, j), foreground=foreground)
            index += len(a)
            j += 1


def text_tag_config(sentence, i):
    sentence = "%5d. %s" % (i, sentence.strip())
    text.insert(tk.END, "%s\n" % sentence)
    # print(sentence)
    if sentence:
        parse_result = parse(sentence[7:])
        test_tag(parse_result, sentence, "entity", TOPIC_BG, i, check_tv1)
        test_tag(parse_result, sentence, "pos1", POSITIVE_BG1, i, check_tv2)
        test_tag(parse_result, sentence, "pos2", POSITIVE_BG2, i, check_tv3)
        test_tag(parse_result, sentence, "neg1", NEGATIVE_BG1, i, check_tv4)
        test_tag(parse_result, sentence, "neg2", NEGATIVE_BG2, i, check_tv5)


def all_button_event(which):
    if result is not None:
        text.delete(1.0, tk.END)

        comments = result["content"]
        scores = result["score"]

        # print(check_var.get())

        if check_var.get() is True:
            comments = [comments[a[0]] for a in result["useful_comment_id"]]
            scores = [scores[a[0]] for a in result["useful_comment_id"]]

        if which == ALL:
            i = 1
            for comment in comments:
                text_tag_config(comment, i)
                i += 1

        elif which == GOOD:
            j = 1
            for i in range(len(scores)):
                if scores[i] >= 4:
                    text_tag_config(comments[i], j)
                    j += 1

        elif which == MEDIUM:
            j = 1
            for i in range(len(scores)):
                if 2 <= scores[i] <= 3:
                    text_tag_config(comments[i], j)
                    j += 1

        elif which == BAD:
            j = 1
            for i in range(len(scores)):
                if scores[i] <= 1:
                    text_tag_config(comments[i], j)
                    j += 1

        elif which == TASTE:
            keywords = ["味", "口感", "吃"]
            j = 1
            for i in range(len(scores)):
                for keyword in keywords:
                    if keyword in comments[i]:
                        text_tag_config(comments[i], j)
                        j += 1

        elif which == SPEED:
            keywords = ["速度", "送"]
            j = 1
            for i in range(len(scores)):
                for keyword in keywords:
                    if keyword in comments[i]:
                        text_tag_config(comments[i], j)
                        j += 1

        elif which == WEIGHT:
            keywords = ["量"]
            j = 1
            for i in range(len(scores)):
                for keyword in keywords:
                    if keyword in comments[i]:
                        text_tag_config(comments[i], j)
                        j += 1

        elif which == SERVICE:
            keywords = ["服务", "态度"]
            j = 1
            for i in range(len(scores)):
                for keyword in keywords:
                    if keyword in comments[i]:
                        text_tag_config(comments[i], j)
                        j += 1

        elif which == OTHER:
            pass


def write_into_file(sentence, which):
    if sentence:
        write(sentence, which)
        error_tv.set("OK!")


# some variable
ALL = "all comments"
GOOD = "good comments"
MEDIUM = "medium comments"
BAD = "bad comments"
TASTE = "about taste"
SPEED = "about speed"
WEIGHT = "about weight"
SERVICE = "about service"
OTHER = "others"

TOPIC_BG = "#FF00FF"
POSITIVE_BG1 = "#f90b38"
POSITIVE_BG2 = "#FF7F00"
NEGATIVE_BG1 = "#00CD66"
NEGATIVE_BG2 = "#0000EE"

all_direction = tk.E + tk.N + tk.W + tk.S
result = None

# get the root
root = tk.Tk()

# fix the window size
root.resizable(False, False)

# Frame 1
frame1 = tk.Frame(root, bd=2, relief=tk.SUNKEN)
frame1.pack(fill=tk.BOTH, expand=tk.YES, anchor=tk.CENTER)

# URL StringVar, URL Entry, Button and corresponding event function
row_num = 0
url_tv = tk.StringVar()
url_tv_column_span = 9
tk.Entry(frame1, textvariable=url_tv).grid(row=row_num, column=0, columnspan=url_tv_column_span,
                                           padx=2, sticky=all_direction)
tk.Button(frame1, text="分析", command=analyse_button_event).grid(
    row=row_num, column=url_tv_column_span, sticky=all_direction)

# Label
row_num = 1
tk.Label(frame1, text="提示：").grid(
    row=row_num, column=0, columnspan=2, pady=10, sticky=all_direction)
prompt_text = tk.StringVar()
tk.Label(frame1, textvariable=prompt_text).grid(
    row=row_num, column=2, columnspan=8, pady=5, sticky=all_direction)
prompt_text.set("请输入店铺URL.")

# Buttons and Checkbutton
row_num = 2
columnspan = 2
tk.Button(frame1, text="All", command=lambda: all_button_event(ALL)).grid(
    row=row_num, column=columnspan*0, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="好评", command=lambda: all_button_event(GOOD)).grid(
    row=row_num, column=columnspan*1, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="中评", command=lambda: all_button_event(MEDIUM)).grid(
    row=row_num, column=columnspan*2, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="差评", command=lambda: all_button_event(BAD)).grid(
    row=row_num, column=columnspan*3, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
check_var = tk.BooleanVar()
check_button = tk.Checkbutton(frame1, text="有效评论", variable=check_var)
check_button.select()
check_button.grid(row=row_num, column=columnspan * 4, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
row_num = 3
tk.Button(frame1, text="味道", command=lambda: all_button_event(TASTE)).grid(
    row=row_num, column=columnspan*0, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="送餐", command=lambda: all_button_event(SPEED)).grid(
    row=row_num, column=columnspan*1, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="份量", command=lambda: all_button_event(WEIGHT)).grid(
    row=row_num, column=columnspan*2, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="服务", command=lambda: all_button_event(SERVICE)).grid(
    row=row_num, column=columnspan*3, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
tk.Button(frame1, text="其它", command=lambda: all_button_event(OTHER)).grid(
    row=row_num, column=columnspan*4, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)

# Radiobutton
row_num = 4
columnspan = 2
check_tv1 = tk.BooleanVar()
c = tk.Checkbutton(frame1, text="评价对象", variable=check_tv1, onvalue=True, fg=TOPIC_BG)
c.grid(row=row_num, column=columnspan*0, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
c.select()
check_tv2 = tk.BooleanVar()
tk.Checkbutton(frame1, text="正向评价", variable=check_tv2, onvalue=True, fg=POSITIVE_BG1).grid(
    row=row_num, column=columnspan*1, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
check_tv3 = tk.BooleanVar()
tk.Checkbutton(frame1, text="正向描述", variable=check_tv3, onvalue=True, fg=POSITIVE_BG2).grid(
    row=row_num, column=columnspan*2, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
check_tv4 = tk.BooleanVar()
tk.Checkbutton(frame1, text="负向评价", variable=check_tv4, onvalue=True, fg=NEGATIVE_BG1).grid(
    row=row_num, column=columnspan*3, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)
check_tv5 = tk.BooleanVar()
tk.Checkbutton(frame1, text="负向描述", variable=check_tv5, onvalue=True, fg=NEGATIVE_BG2).grid(
    row=row_num, column=columnspan*4, columnspan=columnspan, padx=2, pady=2, sticky=all_direction)


#  Text Widget and associated Scrollbar Widget
row_num = 5
text = tk.Text(frame1, height=20, width=60)
text.grid(row=row_num, column=0, columnspan=10, padx=2, pady=2)
scrollbar = tk.Scrollbar(frame1, orient=tk.VERTICAL, command=text.yview)
scrollbar.grid(row=row_num, column=9, rowspan=1, sticky=all_direction)
text.configure(yscrollcommand=scrollbar.set)

# Frame 2: labelFrame
row_num = 6
frame2 = tk.LabelFrame(root, text="纠错面板")
# frame2.grid(row=row_num, column=0, columnspan=11, sticky=all_direction)
frame2.pack(fill=tk.BOTH, expand=tk.YES)

# Entry
error_tv = tk.StringVar()
tk.Entry(frame2, textvariable=error_tv).grid(row=0, column=0, columnspan=11, sticky=all_direction)

# Radiobutton
columnspan = 2
radio_iv = tk.IntVar()
tk.Radiobutton(frame2, text="评价对象", variable=radio_iv, value=1).grid(
    row=1, column=columnspan*0, columnspan=columnspan, sticky=all_direction, padx=2)
tk.Radiobutton(frame2, text="正向评价", variable=radio_iv, value=2).grid(
    row=1, column=columnspan*1, columnspan=columnspan, sticky=all_direction, padx=2)
tk.Radiobutton(frame2, text="正向描述", variable=radio_iv, value=3).grid(
    row=1, column=columnspan*2, columnspan=columnspan, sticky=all_direction, padx=2)
tk.Radiobutton(frame2, text="负向评价", variable=radio_iv, value=4).grid(
    row=1, column=columnspan*3, columnspan=columnspan, sticky=all_direction, padx=2)
tk.Radiobutton(frame2, text="负向描述", variable=radio_iv, value=5).grid(
    row=1, column=columnspan*4, columnspan=columnspan, sticky=all_direction, padx=2)
tk.Button(frame2, text="纠错", command=lambda: write_into_file(error_tv.get(), radio_iv.get())).grid(
    row=1, column=columnspan*5, padx=2, sticky=all_direction)

# Frame 3: LabelFrame
frame3 = tk.LabelFrame(root, text="统计面板", padx=2, pady=2, relief=tk.GROOVE)
frame3.pack(fill=tk.BOTH, expand=tk.YES)

# Buttons
columnspan = 3
padx = 10
tk.Button(frame3, text="店铺整体评分分布", command=lambda: visualization.score_detail(result)).grid(
    row=0, column=columnspan*0, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="商品质量评分分布", command=lambda: visualization.dish_score_detail(result)).grid(
    row=0, column=columnspan*1, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="配送服务评分分布", command=lambda: visualization.service_score_detail(result)).grid(
    row=0, column=columnspan*2, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="各项评价平均指标", command=lambda: visualization.average_score(result)).grid(
    row=1, column=columnspan*0, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="整体评价变化趋势", command=lambda: visualization.weeks_score(result)).grid(
    row=1, column=columnspan*1, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="订餐终端分布", command=lambda: visualization.s_from(result)).grid(
    row=1, column=columnspan*2, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="商品推荐榜", command=lambda: visualization.recommend_dishes2(result)).grid(
    row=2, column=columnspan*0, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)
tk.Button(frame3, text="送餐时间分布", command=lambda: visualization.cost_time(result)).grid(
    row=2, column=columnspan*1, columnspan=columnspan, sticky=all_direction, padx=padx, pady=3)

# main loop
root.mainloop()

