# -*- coding: utf-8 -*-

from __future__ import division
from collections import Counter, defaultdict
from functools import partial
import math, random

#(-plogp和)的表示方法
def entropy(class_probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

#每个特征值一个结果的数据个数/特征值结果总数,特征值结果概率集合
def class_probabilities(labels):
    #特征值下数据结果的的个数
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

#计算单个特征值的熵
def data_entropy(labeled_data):
    #每个特征值下的数据结果集合
    labels = [label for _, label in labeled_data]
    #特征值结果概率集合
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

#计算信息熵
def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets"""
    #获取指定特征下的每个特征值的数据量,为特征总量
    total_count = sum(len(subset) for subset in subsets)
    #len(subset):特征下值的个数
    #total_count:特征下的数据总量
    #
    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets )

#得到指定特征下的每一类值的全部原始数据
def group_by(items, key_fn):
    """returns a defaultdict(list), where each input item 
    is in the list whose key is key_fn(item)"""
    #初始化字典,定义如果没有查询值的话返回list型空数据
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups

#得到指定特征下全部值的全部数据:特征 {特征值1:[(x1,y1),(x2,y2),.....],特征值2:[(xa,ya),(xb,yb)....]}
def partition_by(inputs, attribute):
    """returns a dict of inputs partitioned by the attribute
    each input is a pair (attribute_dict, label)"""
    return group_by(inputs, lambda x: x[0][attribute])    

#返回特征的信息熵
def partition_entropy_by(inputs,attribute):
    """computes the entropy corresponding to the given partition"""        
    partitions = partition_by(inputs, attribute)
    #partitions.values()为整个特征下的全部数据[[(x1,y1),(x2,y2)...],[(xa,ya),(xb,yb),,,]]
    return partition_entropy(partitions.values())        

#执行分类,递归
def classify(tree, input):
    """classify the input using the given decision tree"""

    # if this is a leaf node, return its value
    #如果直接是一个叶子节点则直接返回
    if tree in [True, False]:
        return tree
   
    # otherwise find the correct subtree
    #第一层分解,特征,对应数组
    attribute, subtree_dict = tree

    #获取待分类数据中的特征对应的值
    subtree_key = input.get(attribute)  # None if input is missing attribute

    if subtree_key not in subtree_dict: # if no subtree for key,
        subtree_key = None              # we'll use the None subtree

    #分割完毕,更新树模型
    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree, input)     # and use it to classify the input

#构建决策树函数,递归
def build_tree_id3(inputs, split_candidates=None):

    # if this is our first pass, 
    # all keys of the first input are split candidates
    #首次分裂计算

    if split_candidates is None:
        #获取全部特征
        split_candidates = inputs[0][0].keys()

    # count Trues and Falses in the inputs
    #数据总数
    num_inputs = len(inputs)
    #(true)数据数
    num_trues = len([label for item, label in inputs if label])
    #(false)数据数量
    num_falses = num_inputs - num_trues
    
    if num_trues == 0:                  # if only Falses are left
        return False                    # return a "False" leaf
        
    if num_falses == 0:                 # if only Trues are left
        return True                     # return a "True" leaf

    if not split_candidates:            # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf
                            
    # otherwise, split on the best attribute
    #获取当前信息熵最小的特征
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))

    #当前最优特征下的每个值的全部数据
    partitions = partition_by(inputs, best_attribute)

    #获取剩下的特征集合
    new_candidates = [a for a in split_candidates 
                      if a != best_attribute]
    
    # recursively build the subtrees
    #对当前分区数据的再次分区,递归,使用初次分裂的数据和新的特征值进行下此分裂
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.iteritems() }
    #默认情况
    subtrees[None] = num_trues > num_falses # default case
    #返回当前最好属性,和其分裂策略
    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


if __name__ == "__main__":
    #训练集:(x,y),其中x为特征,y为结果
    inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
    ]

    #展示每个特征的信息熵
    print '展示每个特征的信息熵'
    for key in ['level','lang','tweets','phd']:
        print key, partition_entropy_by(inputs, key)
    print

    senior_inputs = [(input, label)
                     for input, label in inputs if input["level"] == "Senior"]

    print '对剩下的特征求信息熵'
    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(senior_inputs, key)
    print

    print "building the tree"
    print "构建决策树"
    tree = build_tree_id3(inputs)
    print tree

    # (
    #     'level', {
    #         'Senior': (
    #             'tweets', {
    #                 'yes': True,
    #                 None: False,
    #                 'no': False
    #             }
    #         ),
    #         None: True,
    #         'Mid': True,
    #         'Junior': (
    #             'phd', {
    #                 'yes': False,
    #                 None: True,
    #                 'no': True
    #             }
    #         )
    #     }
    # )

    print "分类测试1"
    print "Junior / Java / tweets / no phd", classify(tree, 
        { "level" : "Junior", 
          "lang" : "Java", 
          "tweets" : "yes", 
          "phd" : "no"} )

    print "分类测试2"
    print "Junior / Java / tweets / phd", classify(tree, 
        { "level" : "Junior", 
                 "lang" : "Java", 
                 "tweets" : "yes", 
                 "phd" : "yes"} )

    print "Intern", classify(tree, { "level" : "Intern" } )
    print "Senior", classify(tree, { "level" : "Senior" } )

