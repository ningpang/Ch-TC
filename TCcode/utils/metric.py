import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score, precision_score, recall_score

def cal_f1(x,y):
    return 2.0 * x * y / (x + y + 1e-20)

def calc_evaluation(y_ans,y_pred):
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)
    y_prob = [softmax(y) for y in y_pred]
    y_pred = np.array([np.argmax(i) for i in y_prob])
    y_prob = np.array([y_prob[i][j] for i,j in enumerate(y_pred)])
    
    true_num = len([i for i in y_ans if len(i)>1 or i[0]>0])
    
    test_result = []

    y_label = np.zeros(len(y_pred))

    for idx in range(len(y_ans)):
        rel = y_pred[idx]
        y_label[idx] = y_ans[idx][0]
        test_result.append({'score': y_prob[idx], 'flag': 1 if rel in y_ans[idx] else 0})
        # if rel != 0: # 只算预测类别为非NA的
        #     test_result.append({'score': y_prob[idx], 'flag':1 if rel in y_ans[idx] else 0 })
    precision_v = precision_score(y_label, y_pred, average='macro')
    recall_v = recall_score(y_label, y_pred, average='macro')
    f1_v = f1_score(y_label, y_pred, average='macro')
    f2_v = f1_score(y_label, y_pred, average='micro')
    sorted_test_result = sorted(test_result, key=lambda x: x['score'])
    prec = []
    recall = [] 
    correct = 0
    
    for i, item in enumerate(sorted_test_result[::-1]):
        correct += item['flag']
        prec.append(float(correct) / (i + 1))
        recall.append(float(correct) / true_num) 
        
    prec = np.array(prec)
    recall = np.array(recall)
    
    indxs = [i for i in range(len(recall))]

    x = recall[indxs]
    y = prec[indxs]
    auc = sklearn.metrics.auc(x=x, y=y)
    
    f1 = cal_f1(prec,recall)
    
    # return prec[-1],recall[-1],f1[-1],auc
    return precision_v, recall_v, f1_v, f2_v