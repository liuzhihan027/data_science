# -*- coding: utf-8 -*-

from __future__ import division
from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# k_means主类
class KMeans:
    """performs k-means clustering"""

    # 初始化参数
    def __init__(self, k):
        self.k = k          # number of clusters(聚类中心个数)
        self.means = None   # means of clusters(聚类中心)

    # 分类(返回单个点到每一个中心的距离,取距离最近的点即中心点返回中心点的索引)
    def classify(self, input):
        """return the index of the cluster closest to the input"""
        # min(range(self.k)--定位后面的i的取值,key=lambda i: squared_distance(input, self.means[i]))--对每个key取最小值


        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    # 训练
    def train(self, inputs):

        # 选取随机的k个值,为初始中心值
        self.means = random.sample(inputs, self.k)
        assignments = None

        
        while True:
            # Find new assignments
            # 所有点对应的索引集合
            new_assignments = map(self.classify, inputs)

            # If no assignments have changed, we're done.
            if assignments == new_assignments:                
                return

            # Otherwise keep the new assignments,
            assignments = new_assignments    

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # avoid divide-by-zero if i_points is empty
                # 防止空串,更新第i个中心点
                if i_points:
                    # 属性内的全部点的矢量和的均值来更新中心点
                    self.means[i] = vector_mean(i_points)    

# 全部点距离中心点的距离的和(整体误差)
def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)
    
    return sum(squared_distance(input,means[cluster])
               for input, cluster in zip(inputs, assignments))

# 画出不同k值得到的聚类中心的误差的趋势
def plot_squared_clustering_errors(plt):

    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.show()

#
# using clustering to recolor an image
# 用聚类法重建图像
#

def recolor_image(input_file, k=5):

    img = mpimg.imread(path_to_png_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels) # this might take a while    

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # index of the closest cluster
        return clusterer.means[cluster]     # mean of the closest cluster

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

#
# hierarchical clustering
# 层次聚类
#

def is_leaf(cluster):
    """a cluster is a leaf if it has length 1"""
    return len(cluster) == 1

def get_children(cluster):
    """returns the two children of this cluster if it's a merged cluster;
    raises an exception if this is a leaf cluster"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]

def get_values(cluster):
    """returns the value in this cluster (if it's a leaf cluster)
    or all the values in the leaf clusters below it (if it's not)"""
    if is_leaf(cluster):
        return cluster # is already a 1-tuple containing value
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]

def cluster_distance(cluster1, cluster2, distance_agg=min):
    """finds the aggregate distance between elements of cluster1
    and elements of cluster2"""
    return distance_agg([distance(input1, input2)
                        for input1 in get_values(cluster1)
                        for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0] # merge_order is first element of 2-tuple

def bottom_up_cluster(inputs, distance_agg=min):
    # start with every input a leaf cluster / 1-tuple
    clusters = [(input,) for input in inputs]
    
    # as long as we have more than one cluster left...
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min([(cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]],
                     key=lambda (x, y): cluster_distance(x, y, distance_agg))

        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them, using merge_order = # of clusters left
        merged_cluster = (len(clusters), [c1, c2])

        # and add their merge
        clusters.append(merged_cluster)

    # when there's only one cluster left, return it
    return clusters[0]

def generate_clusters(base_cluster, num_clusters):
    # start with a list with just the base cluster
    clusters = [base_cluster]
    
    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]
        # and add its children to the list (i.e., unmerge it)
        clusters.extend(get_children(next_cluster))

    # once we have enough clusters...
    return clusters

if __name__ == "__main__":

    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(0) # so you get the same results as me
    clusterer = KMeans(3)
    clusterer.train(inputs)
    print "3-means:"
    print clusterer.means
    print

    random.seed(0)
    clusterer = KMeans(2)
    clusterer.train(inputs)
    print "2-means:"
    print clusterer.means
    print

    # 获取全部k得出的误差,寻找拐点
    print "errors as a function of k"

    for k in range(1, len(inputs) + 1):
        print k, squared_clustering_errors(inputs, k)
    print

    # 画出不同k值聚类的总误差趋势
    plot_squared_clustering_errors(plt)

    # 图像聚类
    # 将路径下图像装载成数组
    # img[i][j]表示第i行第j列的像素
    path_to_png_file = r"image.png"
    import matplotlib.image as mping
    img = mping.imread( path_to_png_file )

    top_row = img[0]
    top_left_pixel = top_row[0]
    print top_left_pixel
    red,green,blue,aa = top_left_pixel

    # 将像素数据扁平化
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(5) # 设定参数
    clusterer.train(pixels) # 迭代寻找聚类中心(很慢!很慢!很慢!)

    # 使用聚类中心对数据进行分类
    def recolor(pixel):
        cluster = clusterer.classify(pixel)
        return clusterer.means[cluster]

    # 按行进行处理
    new_img = [[recolor(pixel) for pixel in row] for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()



    print "bottom up hierarchical clustering"
    # 自下向上层次聚类

    base_cluster = bottom_up_cluster(inputs)
    print base_cluster
    #
    # print
    # print "three clusters, min:"
    # for cluster in generate_clusters(base_cluster, 3):
    #     print get_values(cluster)
    #
    # print
    # print "three clusters, max:"
    # base_cluster = bottom_up_cluster(inputs, max)
    # for cluster in generate_clusters(base_cluster, 3):
    #     print get_values(cluster)
