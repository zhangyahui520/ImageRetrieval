"""
faiss DEMO，3种特征检索方法
    https://blog.csdn.net/kanbuqinghuanyizhang/article/details/80774609
"""
import time
import numpy as np
import faiss  # make faiss available
import matplotlib.pyplot as plt

similar_index_list = []

print(f"[INFO] 创建虚拟特征")
d = 2048  # dimension
nb = 100000  # database size
nq = 10000  # nb of queries
k = 5  # we want to see 4 nearest neighbors
test_sample_num = 1  # 测试使用的特征向量数目
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.  # 训练数据集
print(f"[INFO] 查询数据库:{xb.shape}")
xq = np.random.random((test_sample_num, d)).astype('float32')
xq[:, 0] += np.arange(1) / 1000.  # 查询数据集
print(f"[INFO] 查询特征shape:{xq.shape}")
print(f"[INFO] 查询特征:{xq}")

print(f"\n[INFO] 数据添加进faiss数据库")
index = faiss.IndexFlatL2(d)  # build the index
print(f"[INFO] index.is_trained:{index.is_trained}")
index.add(xb)  # add vectors to the index

print(f"[INFO] 使用L2排序索引查询，不需要训练，但是需要两两计算")
start = time.time()
D, I = index.search(xq, k)  # actual search
print(f"[INFO] 查询耗时：{'%0.4f'}s" % (time.time() - start))
print(f"[INFO] 相似用户的ID:{I}")  # neighbors of the 5 first queries
similar_index_list.append(list(I[0]))
print(f"[INFO] 相似向量的距离:{D}")  # neighbors of the 5 last queries
# D表示与相似向量的距离(distance)，维度，I表示相似用户的ID。


print(f"\n[INFO] 使用倒排文件查询")
nlist = 100  # 聚类中心的个数
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # 倒排文件检索
print(type(index))
assert not index.is_trained
index.train(xb)  # 训练介个
import pickle
print(f"[INFO] 保存模型")
with open("./index.model", "wb") as mod:
    pickle.dump(index, mod)
assert index.is_trained
print(f"[INFO] 加载模型")
with open("./index.model", 'rb') as file:
    model = pickle.load(file)
model.add(xb)  # add may be a bit slower as well
start = time.time()
D, I = model.search(xq, k)  # actual search
print(f"[INFO] 查询耗时：{'%0.4f'}s" % (time.time() - start))
print(f"[INFO] 相似用户的ID:{I}")  # neighbors of the 5 first queries
# similar_index_list.append(list(I[0]))
print(f"[INFO] 相似向量的距离:{D}")  # neighbors of the 5 last queries

# 保存模型测试

# 加载模型测试

assert False

print(f"\n[INFO] 使用乘积量化查询")
nlist = 100  # 聚类中心的个数
# 前面定义的维度为d = 64，向量的数据类型为float32。这里压缩成了8个字节。所以压缩比率为 (64*32/8) / 8 = 32
m = 16  # number of bytes per vector
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 初始化
index.train(xb)
index.add(xb)  # 数据入库
start = time.time()
index.nprobe = 2  # make comparable with experiment above
D, I = index.search(xq, k)
print(f"[INFO] 查询耗时：{'%0.4f'}s" % (time.time() - start))
print(f"[INFO] 相似用户的ID:{I}")  # neighbors of the 5 first queries
similar_index_list.append(I[0])
print(f"[INFO] 相似向量的距离:{D}")  # neighbors of the 5 last queries

title_list = ["L2", "IndexIVFFlat", "PQ"]
for idx, sub in enumerate(similar_index_list):
    plt.plot(sub, label=title_list[idx])
plt.legend()
plt.show()
