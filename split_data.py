import numpy as np
def split_train_test(data, test_ratio):
    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    #permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    #test_ratio为测试集所占的百分比
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #iloc选择参数序列中所对应的行
    return data[train_indices], data[test_indices]
root = "/home/lwx/HOG/data/csgo225labeled/new_label.txt"
with open(root, 'r') as f:
    data = f.readlines()
data = np.array(data)
train_set, test_set = split_train_test(data, 0.2)
print(len(train_set), "train +", len(test_set), "test")

train_root = "/home/lwx/HOG/data/csgo225labeled/train.txt"
test_root = "/home/lwx/HOG/data/csgo225labeled/test.txt"
with open(train_root,'w') as f:
    f.write(''.join(train_set))
with open(test_root,'w') as f:
    f.write(''.join(test_set))