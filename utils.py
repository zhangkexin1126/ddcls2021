import matplotlib.pyplot as plt
import numpy as np
import os




def displaypca2d(components, label, colors, dataid, epo):
    ep = str(epo)
    maker = ['o', 'x']
    data = np.concatenate((components, label), axis=1)
    plt.figure(figsize=(8, 6))  # 设置画布大小
    id = 0
    for i in range(len(colors)):
        x = data[data[:, 2] == i][:, 0]
        y = data[data[:, 2] == i][:, 1]
        plt.scatter(x, y, c=colors[i], marker=maker[i])
        for k in range(len(x)):
            plt.text(x[k], y[k], dataid[id])
            id = id + 1
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.xticks([])
    plt.yticks([])
    # 设置标题
    plt.title("PCA Scatter Plot")
    #savename = 'pac_' + ep
    #savepath = '/home/kexin/phdwork/ddcls2021/code/saves/pca'
    #plt.savefig(os.path.join(savepath, savename))
    plt.show()

def displaypca3d(components, label, colors, epo):
    maker = ['o', 'x']
    ep = str(epo)
    data = np.concatenate((components, label), axis=1)
    plt.figure(figsize=(8, 8))  # 设置画布大小
    ax = plt.axes(projection='3d')  # 设置三维轴
    for i in range(len(colors)):
        x = data[data[:, 3] == i][:, 0]
        y = data[data[:, 3] == i][:, 1]
        z = data[data[:, 3] == i][:, 2]
        ax.scatter3D(x, y, z, c=colors[i], marker=maker[i])
    plt.xticks([])
    plt.yticks([])
    #plt.axis('off')
    #plt.xlabel('First')
    #plt.ylabel('Second')
    #ax.set_zlabel('Third')
    # 设置标题
    plt.title("PCA Scatter Plot")
    #savename = 'pac3d_' + ep
    #savepath = '/home/kexin/phdwork/ddcls2021/code/saves/pca'
    #plt.savefig(os.path.join(savepath, savename))
    plt.show()





if __name__ == '__main__':
    data = np.arange(24).reshape((8, 3))
    colors=['red', 'blue']
    x = data[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y = data[:, 1]  # [ 1  4  7 10 13 16 19 22]
    z = data[:, 2]
    plt.figure(figsize=(6, 6))  # 设置画布大小
    ax = plt.axes(projection='3d')  # 设置三维轴
    ax.scatter3D(x, y, z, c='red')  # 三个数组对应三个维度（三个数组中的数一一对应）
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    plt.show()