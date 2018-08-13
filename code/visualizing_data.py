# -*- coding: utf-8 -*-

#数据可视化

import matplotlib.pyplot as plt
from collections import Counter

# 线型图
def make_chart_simple_line_chart(plt):
    # 年份数据
    years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    # 每年对应的gdp数据
    gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

    # 替换sans-serif字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决坐标轴负数显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 创建线型图,横轴为年份,纵轴为gdp
    plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

    # 添加标题
    plt.title(u"每年 GDP")

    # 为y轴添加注释
    plt.ylabel(u"十亿美元")
    plt.show()

# 条形图
def make_chart_simple_bar_chart(plt):
    # 电影名称
    movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
    # 获奖数目
    num_oscars = [5, 11, 3, 8, 10]

    # 处理中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 横坐标
    xs = [i  for i, _ in enumerate(movies)]

    # 根据横纵坐标画图
    plt.bar(xs, num_oscars)
    plt.ylabel(u"获奖数量")
    plt.title(u"最喜爱的电影")

    # 电影名称标记x轴
    plt.xticks([i for i, _ in enumerate(movies)], movies)
    
    plt.show()




# 直方图
def make_chart_histogram(plt):
    # 处理中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 源数据
    grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
    # 划分分数等级
    decile = lambda grade: grade // 10 * 10
    # 每个分数级别对应的人数
    histogram = Counter(decile(grade) for grade in grades)

    # 第三个参数(8)设置条形宽度
    plt.bar([x for x in histogram.keys()],
            histogram.values(),
            8)
    # x轴范围(-5~105),y轴范围(0~5)
    plt.axis([-5, 105, 0, 5])

    # x轴注释(0, 10, ..., 100)
    plt.xticks([10 * i for i in range(11)])
    plt.xlabel(u"分数等级")
    plt.ylabel(u"学生个数")
    plt.title(u"考试分数分布图")
    plt.show()


# 糟糕的直方图
def make_chart_misleading_y_axis(plt, mislead=True):
    # 处理中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    mentions = [500, 505]
    years = [2013, 2014]

    plt.bar([2012.6, 2013.6], mentions, 0.8)
    plt.xticks(years)
    plt.ylabel("# of times I heard someone say 'data science'")

    # if you don't do this, matplotlib will label the x-axis 0, 1
    # and then add a +2.013e3 off in the corner (bad matplotlib!)
    plt.ticklabel_format(useOffset=False)

    if mislead:
        # misleading y-axis only shows the part above 500
        plt.axis([2012.5,2014.5,499,506])
        plt.title("Look at the 'Huge' Increase!")
    else:
        plt.axis([2012.5,2014.5,0,550])
        plt.title("Not So Huge Anymore.")       
    plt.show()


# 线图
def make_chart_several_line_charts(plt):
    # 处理中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据(方差,偏差,总误差)
    variance     = [1,2,4,8,16,32,64,128,256]
    bias_squared = [256,128,64,32,16,8,4,2,1]
    total_error  = [x + y for x, y in zip(variance, bias_squared)]

    xs = range(len(variance))

    # 调用多次,可以在同一画布上添加多条线
    plt.plot(xs, variance,     'g-',  label=u'方差')    # 绿色实线
    plt.plot(xs, bias_squared, 'r-.', label=u'偏差的平方')      # 红色点虚线
    plt.plot(xs, total_error,  'b:',  label=u'总误差') # 蓝色点线

    # loc=9表示将图例放置在顶部中央
    plt.legend(loc=9)
    plt.xlabel(u"模型复杂度")
    plt.title(u"偏差-方差权衡图")
    plt.show()


# 散点图
def make_chart_scatter_plot(plt):
    # 处理中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据(朋友数,在社交网站时间,用户标签)
    friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
    minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    plt.scatter(friends, minutes)
    
    # 对每个点添加标记
    for label, friend_count, minute_count in zip(labels, friends, minutes):
        plt.annotate(label,
                     xy=(friend_count, minute_count), # 将标记放在对应点上
                     xytext=(5, -5), # 需要部分偏离
                     textcoords='offset points')

    plt.title(u"日分钟数与朋友数")

    plt.xlabel(u"朋友的个数")
    plt.ylabel(u"花在网站上的日分钟数")
    plt.show()

make_chart_scatter_plot(plt)

def make_chart_scatterplot_axes(plt, equal_axes=False):

    test_1_grades = [ 99, 90, 85, 97, 80]
    test_2_grades = [100, 85, 60, 90, 70]

    plt.scatter(test_1_grades, test_2_grades)
    plt.xlabel("test 1 grade")
    plt.ylabel("test 2 grade")

    if equal_axes:
        plt.title("Axes Are Comparable")
        plt.axis("equal")
    else:
        plt.title("Axes Aren't Comparable")

    plt.show()

def make_chart_pie_chart(plt):

    plt.pie([0.95, 0.05], labels=["Uses pie charts", "Knows better"])

    # make sure pie is a circle and not an oval
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    print 111

    # make_chart_simple_line_chart(plt)
    #
    # make_chart_simple_bar_chart(plt)
    #
    # make_chart_histogram(plt)
    #
    # make_chart_misleading_y_axis(plt, mislead=True)
    #
    # make_chart_misleading_y_axis(plt, mislead=False)
    #
    # make_chart_several_line_charts(plt)
    #
    # make_chart_scatterplot_axes(plt, equal_axes=False)
    #
    # make_chart_scatterplot_axes(plt, equal_axes=True)
    #
    # make_chart_pie_chart(plt)
