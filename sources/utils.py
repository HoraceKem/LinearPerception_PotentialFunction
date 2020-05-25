import termcolor
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools


def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])
def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])
def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])


def combination(list_all, n):
    return list(itertools.combinations(list_all, n))


def compare_d(d, cluster_num, current_label):
    positive_correction = []
    negative_correction = []
    keep = []
    all_good_flag = True
    for index in range(cluster_num):
        if index != current_label:
            if d[index] >= d[current_label]:
                negative_correction.append(index)
                all_good_flag = False
            else:
                keep.append(index)
    if all_good_flag:
        keep.append(current_label)
    else:
        positive_correction.append(current_label)
    return positive_correction, negative_correction, keep


def compute_K(x, y, cluster, data, label, item):
    K = 0
    for i in item:
        if label[i] == cluster:
            symbol = 1
        else:
            symbol = -1
        K = K + symbol * math.exp(-pow((x - data[i][0]), 2) - pow((y - data[i][1]), 2))
    return K


def visualize_linear(cluster_num, sample_num, data, W):
    color_list = ['r', 'g', 'b', 'c', 'k', 'm', 'y']
    plt.figure(figsize=(5.3, 5))
    for j in range(cluster_num):
        plt.scatter(data[j*sample_num:(j+1)*sample_num, 0], data[j*sample_num:(j+1)*sample_num, 1], c=color_list[j])
    # x = np.linspace(-10, 10)
    group = combination(list(range(cluster_num)), 2)
    y = []
    x = np.linspace(data.min(0)[0] - 1, data.max(0)[0] + 1, 10000)
    for k in group:

        a = W[k[0]][0] - W[k[1]][0]
        b = W[k[0]][1] - W[k[1]][1]
        c = W[k[0]][2] - W[k[1]][2]


        y.append((-a * x -c) / b)
        plt.plot(x, y[-1], c=color_list[len(y)-1])
        plt.xlim((data.min(0)[0] - 1, data.max(0)[0] + 1))
        plt.ylim((data.min(0)[1]-1, data.max(0)[1]+1))
    plt.fill_between(x, y[0], y[2], where=y[0] <= y[2], facecolor='red', interpolate=True, alpha=0.3)
    plt.fill_between(x, -10, y[1], where=y[1] >= y[2], facecolor='blue', interpolate=True, alpha=0.3)
    plt.fill_between(x, -10, y[0], where=y[1] <= y[2], facecolor='blue', interpolate=True, alpha=0.3)
    plt.fill_between(x, y[1], 10,  where=y[1] >= y[2], facecolor='green', interpolate=True, alpha=0.3)
    plt.fill_between(x, y[2], 10, where=y[1] <= y[2], facecolor='green', interpolate=True, alpha=0.3)
    plt.show()


def visualize_nonlinear(cluster_num, sample_num, data, label, item_all):
    color_list = ['r', 'g', 'b', 'c', 'k', 'm', 'y']
    plt.figure(figsize=(5.3, 5))
    for j in range(cluster_num):
        plt.scatter(data[j * sample_num:(j + 1) * sample_num, 0],
                    data[j * sample_num:(j + 1) * sample_num, 1],
                    c=color_list[j])

    for k in range(cluster_num):
        xpts = np.linspace(data.min(0)[0] - 1, data.max(0)[0] + 1, 200)
        ypts = np.linspace(data.min(0)[1] - 1, data.max(0)[1] + 1, 200)
        xpts, ypts = np.meshgrid(xpts, ypts)
        z = np.vectorize(lambda x, y: compute_K(x, y, k, data, label, item_all[k]))
        plt.contour(xpts, ypts, z(xpts, ypts), 0, colors=color_list[k])
    plt.show()