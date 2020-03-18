import numpy as np
from numpy import *
from sklearn.metrics import accuracy_score

def cosine_distance(v1, v2):
    v1_sq = np.inner(v1, v1)
    v2_sq = np.inner(v2, v2)
    dis = 1 - np.inner(v1, v2) / math.sqrt(v1_sq * v2_sq)
    return dis

def kNNClassify(newInput, dataSet, test_id, k):
    numSamples = dataSet.shape[0]

    distance_cos = [0] * numSamples
    for i in range(numSamples):
        distance_cos[i] = cosine_distance(newInput, dataSet[i])

    diff = (tile(newInput, (numSamples, 1)) - dataSet)
    squareDiff = diff ** 2
    squareDist = sum(squareDiff, axis=1)
    distance_euc = squareDist**0.5

    distance = distance_euc
    # sort the distance
    sortedDisIndices = np.argsort(distance)
    classCount = {}  # difine a dictionary

    for i in range(k):
        voteLabel = test_id[sortedDisIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def compute_accuracy(att_pre, test_visual, test_idex, test_id):
    outpre = [0] * test_visual.shape[0]
    test_label = np.squeeze(np.asarray(test_idex))
    test_label = test_label.astype("float32")

    for i in range(test_visual.shape[0]):  # CUB 2933
        outputLabel = kNNClassify(test_visual[i, :], att_pre, test_id, 1)
        outpre[i] = outputLabel
    # compute averaged per class accuracy
    outpre = np.array(outpre, dtype='int')
    unique_labels = np.unique(test_label)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(test_label == l)[0]
        acc += accuracy_score(test_label[idx], outpre[idx])
    acc = acc / unique_labels.shape[0]
    return acc

class Test_nn():
    def __init__(self, sess, att, f_out, cla_num):
        self.att = att
        self.f_out = f_out
        self.cla_num = cla_num
        self.sess = sess

    def test_zsl(self, data):
        attribute = data['attribute']
        test_feats = data['test_unseen_fea']  # 5685 x 2048
        test_idex = data['test_unseen_idex']
        test_id = np.unique(test_idex)
        test_attr = attribute[test_id]

        att_pre = self.sess.run(self.f_out, feed_dict={self.att: test_attr})
        acc = compute_accuracy(att_pre, test_feats, test_idex, test_id)
        return acc

    def test_seen(self, data):
        attribute = data['attribute']
        test_feats = data['test_seen_fea'] #4958 x 2048
        test_idex = data['test_seen_idex']
        test_id = np.arange(self.cla_num)
        test_attr = attribute[test_id]

        att_pre = self.sess.run(self.f_out, feed_dict={self.att: test_attr})
        acc = compute_accuracy(att_pre, test_feats, test_idex, test_id)
        return acc

    def test_unseen(self, data):
        attribute = data['attribute']
        test_feats = data['test_unseen_fea']  # 5685 x 2048
        test_idex = data['test_unseen_idex']
        test_id = np.arange(self.cla_num)
        test_attr = attribute[test_id]

        att_pre = self.sess.run(self.f_out, feed_dict={self.att: test_attr})
        acc = compute_accuracy(att_pre, test_feats, test_idex,test_id)
        return acc
