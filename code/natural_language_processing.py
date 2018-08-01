# -*- coding: utf-8 -*-

from __future__ import division
import math, random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt

# 根据坐标判定其重要性的词云展示
from typing import List, Any, Union


def plot_resumes(plt):
    data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

    def text_size(total):
        """equals 8 if total is 0, 28 if total is 200"""
        return 8 + total / 200 * 20

    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))
    plt.xlabel("Popularity on Job Postings")
    plt.ylabel("Popularity on Resumes")
    plt.axis([0, 100, 0, 100])
    plt.show()

#
# n-gram models
#

# 爬取的字符异常处理
def fix_unicode(text):
    return text.replace(u"\u2019", "'")

# 从制定网址爬取数据,将每个单词放进数组,返回网页全部单词的巨大数组
def get_document():

    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')

    content = soup.find("div", "article-body")        # find article-body div
    regex = r"[\w']+|[\.]"                            # matches a word or a period

    document = []


    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document

# 二院代数(输入字典list)
def generate_using_bigrams(transitions):
    # 按"."的下一个单词作为一句话的起始单词
    current = "."   # this means the next word will start a sentence
    result = []
    while True:
        # 从单词表中选取下一个单词的列表
        next_word_candidates = transitions[current]    # bigrams (current, _)
        # 从起始单词的列表中随机选择一个单词
        current = random.choice(next_word_candidates)  # choose one at random
        result.append(current)                         # append it to results
        if current == ".": return " ".join(result)     # if "." we're done

# 三元语法
def generate_using_trigrams(starts, trigram_transitions):
    # 随机选择一个起始单词
    current = random.choice(starts)   # choose a random starting word
    prev = "."                        # and precede it with a '.'
    result = [current]
    while True:
        # 获取已知前两个单词的后一个单词的列表
        next_word_candidates = trigram_transitions[(prev, current)]
        # 随机选择一个单词作为下一个单词
        next = random.choice(next_word_candidates)

        prev, current = current, next
        result.append(current)
        # 设置截断点
        if current == ".":
            return " ".join(result)

# 判断是否为终端
def is_terminal(token):
    return token[0] != "_"

# 扩大(递归)
def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        # ignore terminals
        # 如果为具体单词时跳出本次循环
        if is_terminal(token): continue


        # choose a replacement at random
        # 非具体单词,从语法表中选取其对应的分词(可能是具体单词可能是终端语法)
        replacement = random.choice(grammar[token])

        # 如果随机选取的分词为正常单词:直接使用正常单词添加数组,如果非正常单词则将其位置替换成其对应的分词(可能是具体单词可能是终端语法)
        # 如果发现非终端符号,随机选择替代者
        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        # 递归
        return expand(grammar, tokens)

    # if we get here we had all terminals and are done
    # 直到递归到全部为有效字符时返回最终数组
    return tokens

# 使用语法生成句子
def generate_sentence(grammar):
    return expand(grammar, ["_S"])

#
# Gibbs Sampling
# 吉布斯采样
#

# 投骰子,得到一个随机数
def roll_a_die():
    return random.choice([1,2,3,4,5,6])

# 返回(第一次获取的点数,两次点数之和)
def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

# 输入第一次得到的点数,返回随机的两次点数之和
def random_y_given_x(x):
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()

# 输入两次点数之和,随机返回第一次点数的取值
def random_x_given_y(y):
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)

# gibbs采样
def gibbs_sample(num_iters=100):
    # 值初始化
    x, y = 1, 2 # doesn't really matter

    # 根据计算次数(迭代次数不断更新x,y取值,最后返回结果)
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

# 直接取值同gibbs采样取值对比
def compare_distributions(num_samples=1000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

#
# TOPIC MODELING
# 主题建模
#

def sample_from(weights):

    # 权重和
    total = sum(weights)
    # 随机选择权重和范围内权重
    rnd = total * random.random()       # uniform between 0 and total

    # 遍历给定权重,如果随机选择的权重小于等于原始给定的权重,返回原始权重的位置,即主题序号
    for i, w in enumerate(weights):
        rnd -= w                        # return the smallest i such that
        if rnd <= 0: return i           # sum(weights[:(i+1)]) >= rnd

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4

# 计数列表,每个文档一个列表
document_topic_counts = [Counter()
                         for _ in documents]


# 计数列表,每个主题一个列表
topic_word_counts = [Counter() for _ in range(K)]

# 数字的一个列表,每个主题一个列表
topic_counts = [0 for _ in range(K)]

# 获取每个文章的长度
document_lengths = map(len, documents)


# 统计不同单词的数量
distinct_words = set(word for document in documents for word in document)

# 单词去重个数
W = len(distinct_words)

# 文章的个数
D = len(documents)



# 输入主题和文档序号,(文档中当前主题的数据量<平滑>/当前的文档长度<平滑>)
def p_topic_given_document(topic, d, alpha=0.1):
    """the fraction of words in document _d_
    that are assigned to _topic_ (plus some smoothing)"""

    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))



# 输入单词和主题序号,(当前主题下的当前单词的数据量<平滑>/主题下全部的数据量<平滑>/)
def p_word_given_topic(word, topic, beta=0.1):
    """the fraction of words assigned to _topic_
    that equal _word_ (plus some smoothing)"""

    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))

# 输入文章序号,单词,和主题序号
def topic_weight(d, word, k):
    """given a document and a word in that document,
    return the weight for the k-th topic"""
    # (文档中当前主题的数据量<平滑>/当前的文档长度<平滑>) * (当前主题下的当前单词的数据量<平滑>/主题下全部的数据量<平滑>/)
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

# 选择一个新的主题(输入:文档序号,单词)
def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])



random.seed(0)

# 为每个单词随机分配文章主题
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]


# 遍历文章的每一个单词,将基本数据填满
for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):

        # 第d个文章下的属于topic的主题加一
        document_topic_counts[d][topic] += 1
        # 第topic的主题下的单词word出现次数加一
        topic_word_counts[topic][word] += 1
        # 主题下数据量加一
        topic_counts[topic] += 1



# 使用吉布斯采样的思想进行采样
for iter in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):

            # remove this word / topic from the counts
            # so that it doesn't influence the weights
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # choose a new topic based on the weights
            # 为当前文章的当前单词重新分配新的主题
            new_topic = choose_new_topic(d, word)
            # 将新的主题关联到文章d的第i个位置即word所在的位置
            document_topics[d][i] = new_topic

            # and now add it back to the counts
            # 将更新完主题的单词放回总体数据当中
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1



if __name__ == "__main__":

    # 根据坐标判定其重要性的词云展示
    plot_resumes(plt)

    # 根据指定网址获取其全部单词组成的数组
    document = get_document()

    # 将全部词语错开整合,整合成(单词,下一个单词)
    bigrams = zip(document, document[1:])
    transitions = defaultdict(list)
    for prev, current in bigrams:
        transitions[prev].append(current)

    random.seed(0)
    print "bigram sentences"# 双二元语法
    # 导入字段list
    for i in range(10):
        # 全部根据下一个单词来逐步添加单词形成新的句子,随机的
        print i, generate_using_bigrams(transitions)
    print

    # trigrams
    # 三元语法

    # 获取连续的两个单词
    trigrams = zip(document, document[1:], document[2:])
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in trigrams:

        if prev == ".":              # if the previous "word" was a period
            starts.append(current)   # then this is a start word

        trigram_transitions[(prev, current)].append(next)

    print "trigram sentences"
    for i in range(10):
        # 输入起始语句列表,以及字典list(key为两个连续的单词,value为每句话含有这个单词顺序的后一个单词的列表)
        print i, generate_using_trigrams(starts, trigram_transitions)
    print

    # 使用语法来生成句子
    # 定义语法规则
    grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
    }

    print "grammar sentences"
    for i in range(10):
        print i, " ".join(generate_sentence(grammar))
    print

    print "gibbs sampling"# 吉布斯采样
    comparison = compare_distributions()
    for roll, (gibbs, direct) in comparison.iteritems():
        print roll, gibbs, direct


    # topic MODELING
    # 主题建模

    # 以下均为主题建模的展示阶段,真正运算阶段见上方函数

    # 遍历每个主题下每个单词的出现次数
    for k, word_counts in enumerate(topic_word_counts):
        # word_counts.most_common():返回一个topN的列表
        for word, count in word_counts.most_common():
            if count > 0: print k, word, count

    # 定义主题名称
    topic_names = ["Big Data and programming languages",
                   "Python and statistics",
                   "databases",
                   "machine learning"]

    # 整合文章和文章中的主体的数据量
    for document, topic_counts in zip(documents, document_topic_counts):
        print document
        for topic, count in topic_counts.most_common():
            if count > 0:
                print topic_names[topic], count,'|',
        print
