# -*- coding: utf-8 -*-

from __future__ import division  # want 3 / 2 == 1.5
import re, math, random  # regexes, math functions, random numbers
import matplotlib.pyplot as plt  # pyplot
from collections import defaultdict, Counter
from functools import partial


#
# functions for working with vectors
#

#两个点的矢量和
def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]


#向量减法
def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]

#全部点的矢量和
def vector_sum(vectors):
    return reduce(vector_add, vectors)

#标量乘法
def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


# this isn't right if you don't from __future__ import division
def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    # 数乘向量(1/元素个数,全部点的矢量和)
    return scalar_multiply(1 / n, vector_sum(vectors))

#乘积求和
def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n
    """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# 向量平方
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


#向量大小
def magnitude(v):
    return math.sqrt(sum_of_squares(v))


# 两个向量距离的平方和
def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

# 两个向量的距离
def distance(v, w):
    return math.sqrt(squared_distance(v, w))


#
# functions for working with matrices
#

#得到矩阵行数和列数
def shape(A):
    # type: (object) -> object
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A, i):
    return A[i]

#得到一列
def get_column(A, j):
    # type: (object, object) -> object
    return [A_i[j] for A_i in A]

#创建矩阵
def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix 
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]


def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0


identity_matrix = make_matrix(5, 5, is_diagonal)

#          user 0  1  2  3  4  5  6  7  8  9
#
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9


#####
# DELETE DOWN
#


def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")

    num_rows, num_cols = shape(A)

    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)


def make_graph_dot_product_as_vector_projection(plt):
    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0, 0]

    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0, 0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(vâ¢w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v, w, o), marker='.')
    plt.axis('equal')
    plt.show()
