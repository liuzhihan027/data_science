# -*- coding: utf-8 -*-

from __future__ import division
import math, random
from collections import defaultdict, Counter

from typing import List

from linear_algebra import dot

# 基本数据
users_interests = [
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

# 统计每个词的频次(most_common函数为进行TOPN)
popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests).most_common()


# 推荐,除去给定的兴趣列表后的排名最靠前的兴趣
def most_popular_new_interests(user_interests, max_results=5):
    suggestions = [(interest, frequency) 
                   for interest, frequency in popular_interests
                   if interest not in user_interests]
    return suggestions[:max_results]

#
# user-based filtering
# 基于用户的协同过滤方法
#

# 计算余弦相似度
def cosine_similarity(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))

# 获取全部用户的全部兴趣(去重)
unique_interests = sorted(list({ interest 
                                 for user_interests in users_interests
                                 for interest in user_interests }))


# 生成用户兴趣向量
def make_user_interest_vector(user_interests):
    """given a list of interests, produce a vector whose i-th element is 1
    if unique_interests[i] is in the list, 0 otherwise"""
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

# 为每一个用户生成用户兴趣向量
user_interest_matrix = map(make_user_interest_vector, users_interests)

# 对每两个用户之间计算余弦相似度
user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                     for interest_vector_i in user_interest_matrix]

# 对于给定用户编号,找到与其最相似的用户,需要排除自身和与其相似度为0的用户
def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)                      # find other
             for other_user_id, similarity in                 # users with
                enumerate(user_similarities[user_id])         # nonzero 
             if user_id != other_user_id and similarity > 0]  # similarity

    return sorted(pairs,                                      # sort them
                  key=lambda (_, similarity): similarity,     # most similar
                  reverse=True)                               # first


# 将这个兴趣的,其他用户也感兴趣的用户相似度相加,为对这个兴趣的权重,根据参数筛选掉已有兴趣属性
def user_based_suggestions(user_id, include_current_interests=False):
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                         key=lambda (_, weight): weight,
                         reverse=True)

    # and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) 
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

#
# Item-Based Collaborative Filtering
# 基于物品的协同过滤算法
#

# 将用户兴趣矩阵进行转置
interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_matrix]
                        for j, _ in enumerate(unique_interests)]

# 再次计算余弦相似度
interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

# 输入兴趣id,寻找与其最相似的兴趣
def most_similar_interests_to(interest_id):
    # 找到这个兴趣的向量
    similarities = interest_similarities[interest_id]
    # 排除自身和相似度为0的兴趣向量
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]

    return sorted(pairs,
                  key=lambda (_, similarity): similarity,
                  reverse=True)

# 输入用户id获取基于物品的推荐
def item_based_suggestions(user_id, include_current_interests=False):
    # 初始化结果参数
    suggestions = defaultdict(float)
    # 从用户矩阵中选取当前用户矩阵
    user_interest_vector = user_interest_matrix[user_id]

    # 根据用户感兴趣的物品获取其物品与其物品相似度的和
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # 对与其物品相似度的和进行排序
    suggestions = sorted(suggestions.items(),
                         key=lambda (_, similarity): similarity,
                         reverse=True)

    # 根据参数选取是否推荐已经感兴趣商品
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) 
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


if __name__ == "__main__":

    print "Popular Interests"
    print popular_interests
    print

    print "Most Popular New Interests"
    print "already like:", ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"]
    print most_popular_new_interests(["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"])
    print
    print "already like:", ["R", "Python", "statistics", "regression", "probability"]
    print most_popular_new_interests(["R", "Python", "statistics", "regression", "probability"])
    print    

    print "User based similarity"
    print "most similar to 0"
    print most_similar_users_to(0)

    print "Suggestions for 0"
    print user_based_suggestions(0)
    print

    print "Item based similarity"
    print "most similar to 'Big Data'"
    print most_similar_interests_to(0)
    print

    print "suggestions for user 0"
    print item_based_suggestions(0)

