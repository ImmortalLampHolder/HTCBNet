import numpy as np

def R2(target,predict):
    target = np.array(target)
    predict = np.array(predict)
    return 1 - (np.sum(np.power((target-predict),2))/np.sum(np.power((target - np.mean(target)),2)))


# Mean Predict Error
def MPE(target,predict):
    target = np.array(target)
    predict = np.array(predict)
    return np.mean(np.abs(target-predict)/target).item()