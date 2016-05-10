import tkinter as tk
from threading import Thread

from WaiMaiMiner.crawler import crawl
from WaiMaiMiner import visualization
from WaiMaiMiner.mining import parse, write_, train


def write_into_file(sentence, which):
    if sentence:
        write_(sentence, which)
        error_tv.set("OK, 纠错成功!")


def retrain_thread():
    train()
    error_tv.set("训练完毕 (*＾-＾*)")


def retrain():
    error_tv.set("正在重新训练模型，请稍等......")
    t = Thread(target=retrain_thread)
    t.start()


def get_result():
    global result
    try:
        result = crawl(url_sv.get())
        prompt_label.set("爬取完毕\(^o^)/YES!")
    except ValueError:
        prompt_label.set("错误的URL，请重新输入。")


def analyse_button_event():
    prompt_label.set("正在爬取评论，请稍等......")
    t = Thread(target=get_result)
    t.start()


def button_event():
    print("This is a button event.")


result = None
all_column_num = 12

# get the root
root = tk.Tk()

# fix the window size
# root.geometry('1800x800+0+0')
# root.resizable(False, False)

"""
left_window, to take all the button
"""
left_window = tk.Frame(root, bd=2, relief=tk.SUNKEN)
left_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

"""
Attention: Each frame should be packed 'fill=tk.BOTH, expand=tk.YES'
"""

"""
frame1, to hold the tk.Entry and the prompt label
"""
row = 0
# URL Entry, to get the url entry
url_sv = tk.StringVar()
tk.Entry(left_window, textvariable=url_sv).grid(
    row=row, column=0, columnspan=all_column_num - 1, padx=2, sticky='nsew')
# '分析' Button
tk.Button(left_window, text="分析", command=analyse_button_event).grid(
    row=row, column=all_column_num - 1, sticky='nsew')

# '提示' Label
row += 1
tk.Label(left_window, text="提示：").grid(
    row=row, column=0, columnspan=2, pady=5, sticky='nsew')
prompt_label = tk.StringVar()
tk.Label(left_window, textvariable=prompt_label).grid(
    row=row, column=2, columnspan=all_column_num - 1, pady=5, sticky='nsew')
prompt_label.set("请输入店铺URL.")

"""
frame2, to hold the.all comments frame
"""
# Prompt Label
row += 1
tk.Label(left_window, text="全部评论:").grid(
    row=row, column=0, sticky='nsew')

row += 1
columnspan = 3
# Button
tk.Button(left_window, text="全部", command=lambda: button_event()).grid(
    row=row, column=columnspan * 0, columnspan=columnspan, padx=2, sticky='nsew')
tk.Button(left_window, text="好评", command=lambda: button_event()).grid(
    row=row, column=columnspan * 1, columnspan=columnspan, padx=2, sticky='nsew')
tk.Button(left_window, text="中评", command=lambda: button_event()).grid(
    row=row, column=columnspan * 2, columnspan=columnspan, padx=2, sticky='nsew')
tk.Button(left_window, text="差评", command=lambda: button_event()).grid(
    row=row, column=columnspan * 3, columnspan=columnspan, padx=2, sticky='nsew')

row += 1
columnspan = 3
# Checkbutton
object_check_var = tk.BooleanVar()
object_check_button = tk.Checkbutton(left_window, text="对象", variable=object_check_var)
object_check_button.select()
object_check_button.grid(row=row, column=columnspan * 0, columnspan=columnspan, padx=2, pady=2, sticky='nsew')

positive_check_var = tk.BooleanVar()
tk.Checkbutton(left_window, text="正向观点", variable=positive_check_var).grid(
    row=row, column=columnspan * 1, columnspan=columnspan, padx=2, pady=2, sticky='nsew'
)
negative_check_var = tk.BooleanVar()
tk.Checkbutton(left_window, text="负向观点", variable=negative_check_var).grid(
    row=row, column=columnspan * 2, columnspan=columnspan, padx=2, pady=2, sticky='nsew'
)

useful_check_var = tk.BooleanVar()
useful_check_button = tk.Checkbutton(left_window, text="有效评论", variable=useful_check_var)
useful_check_button.select()
useful_check_button.grid(row=row, column=columnspan * 3, columnspan=columnspan, padx=2, pady=2, sticky='nsew')

"""
frame4, to hold the.taste comments frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(row=row, column=0, pady=0, sticky='nsew')

row += 1
# LabelFrame
taste_frame = tk.LabelFrame(left_window, text="味道评论", padx=2, relief=tk.GROOVE)
taste_frame.grid(row=row, column=0, sticky='nsew', columnspan=all_column_num)
# Button
columnspan = 4
tk.Button(taste_frame, text="全部", command=lambda: button_event()).grid(
    row=0, column=0, columnspan=columnspan, padx=2, sticky='nsew')


"""
frame5, to hold the.delivery comments frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(row=row, column=0, pady=0, sticky='nsew')

row += 1
# LabelFrame
delivery_frame = tk.LabelFrame(left_window, text="配送评论", padx=2, pady=2, relief=tk.GROOVE)
delivery_frame.grid(row=row, column=0, sticky='nsew', columnspan=all_column_num)
# Button
columnspan = 4
tk.Button(delivery_frame, text="全部", command=lambda: button_event()).grid(
    row=0, column=0, columnspan=columnspan, padx=2, sticky='nsew')

"""
frame6, to hold the.service comments frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(row=row, column=0, pady=0, sticky='nsew')

# LabelFrame
row += 1
service_frame = tk.LabelFrame(left_window, text="服务评论", padx=2, pady=2, relief=tk.GROOVE)
service_frame.grid(row=row, column=0, sticky='nsew', columnspan=all_column_num)
# Button
columnspan = 4
tk.Button(service_frame, text="全部", command=lambda: button_event()).grid(
    row=0, column=0, columnspan=columnspan, padx=2, sticky='nsew')

"""
frame7, to hold the.other comments frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(
    row=row, column=0, pady=0, sticky='nsew')

# LabelFrame
row += 1
other_frame = tk.LabelFrame(left_window, text="其他评论", padx=2, pady=2, relief=tk.GROOVE)
other_frame.grid(row=row, column=0, sticky='nsew', columnspan=all_column_num)
# Button
columnspan = 4
tk.Button(other_frame, text="全部", command=lambda: button_event()).grid(
    row=0, column=0, columnspan=columnspan, padx=2, sticky='nsew')

"""
frame7, to hold the.improvement frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(
    row=row, column=0, pady=0, sticky='nsew')

# Label
row += 1
tk.Label(left_window, text="改进面板:").grid(
    row=row, column=0, sticky='nsew')

padx = 5
# Entry
row += 1
error_tv = tk.StringVar()
tk.Entry(left_window, textvariable=error_tv).grid(
    row=row, column=0, columnspan=all_column_num, sticky='nsew', padx=padx)

row += 1
columnspan = 4
# Radiobutton and button
radio_iv = tk.IntVar()
tk.Radiobutton(left_window, text='正向观点', variable=radio_iv, value=2).grid(
    row=row, column=columnspan * 0, columnspan=columnspan, sticky='nesw', padx=padx
)
tk.Radiobutton(left_window, text='对象', variable=radio_iv, value=1).grid(
    row=row, column=columnspan * 1, columnspan=columnspan, sticky='nesw', padx=padx
)
tk.Button(left_window, text='确定', command=lambda: write_into_file(error_tv.get(), radio_iv.get())).grid(
    row=row, column=columnspan * 2, columnspan=columnspan, padx=padx, sticky='nesw')

row += 1
tk.Radiobutton(left_window, text='负向观点', variable=radio_iv, value=4).grid(
    row=row, column=columnspan * 0, columnspan=columnspan, sticky='nesw', padx=padx
)
tk.Radiobutton(left_window, text='其他', variable=radio_iv, value=6).grid(
    row=row, column=columnspan * 1, columnspan=columnspan, sticky='nesw', padx=padx
)
tk.Button(left_window, text='重新训练', command=lambda: retrain()).grid(
    row=row, column=columnspan * 2, columnspan=columnspan, padx=padx, sticky='nesw')

"""
frame2, to hold the.statistic frame
"""
# split label
row += 1
tk.Label(left_window, text="").grid(
    row=row, column=0, pady=0, sticky='nsew')

row += 1
columnspan = 12
# LabelFrame
frame1 = tk.LabelFrame(left_window, text="统计面板", padx=2, pady=2, relief=tk.GROOVE)
# frame2.pack(fill=tk.BOTH, expand=tk.YES)
frame1.grid(row=row, column=0, sticky='nsew', columnspan=columnspan)
# Buttons
columnspan = 3
padx = 2
tk.Button(frame1, text="店铺整体评分分布", command=lambda: visualization.score_detail(result)).grid(
    row=0, column=columnspan * 0, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="商品质量评分分布", command=lambda: visualization.dish_score_detail(result)).grid(
    row=0, column=columnspan * 1, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="配送服务评分分布", command=lambda: visualization.service_score_detail(result)).grid(
    row=0, column=columnspan * 2, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="各项评价平均指标", command=lambda: visualization.average_score(result)).grid(
    row=1, column=columnspan * 0, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="整体评价变化趋势", command=lambda: visualization.weeks_score(result)).grid(
    row=1, column=columnspan * 1, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="订餐终端分布", command=lambda: visualization.s_from(result)).grid(
    row=1, column=columnspan * 2, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="商品推荐榜", command=lambda: visualization.recommend_dishes2(result)).grid(
    row=2, column=columnspan * 0, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="送餐时间分布", command=lambda: visualization.cost_time(result)).grid(
    row=2, column=columnspan * 1, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)
tk.Button(frame1, text="各评价对象分布", command=lambda: visualization.topic(result)).grid(
    row=2, column=columnspan * 2, columnspan=columnspan, sticky='nsew', padx=padx, pady=3)

# right_window, to hold the text panel
right_window = tk.Frame(root, bd=2, relief=tk.GROOVE)
right_window.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)

# Text widget, to hold the comment
text = tk.Text(right_window)
# content = open("test.txt", encoding="utf-8").read()
# text.insert(tk.END, content)
text.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

# Associated Scrollbar Widget
scrollbar = tk.Scrollbar(right_window, orient=tk.VERTICAL, command=text.yview)
scrollbar.pack(side=tk.LEFT, fill=tk.Y)
text.configure(yscrollcommand=scrollbar.set)

root.mainloop()
