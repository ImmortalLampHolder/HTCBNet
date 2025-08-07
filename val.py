import math
import torch
from nets.network import HTCBNet
from dataloader.sequence import Sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.metrics import R2,MPE


model_name = "HTCBNet"

#load model and weights
weights_path = f"./weights/{model_name}.pth"

model = HTCBNet()
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)
device     = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model = model.eval()

#load test data
batch_size=30
val_data  = np.load("./dataset/val_data.npy")
val_label = np.load("./dataset/val_label.npy")
valdataset = Sequence(val_data,val_label)
valloader  = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

#validation
len  = valdataset.__len__()
step = math.ceil(len/batch_size)
y_truth = []
y_pred  = []
pbar = tqdm(total=step, desc=f'test', postfix=dict, mininterval=0.3)
for iteration, batch in enumerate(valloader):
    if iteration >= step:
            break
    data, label = batch[0], batch[1]
    with torch.no_grad():
        data    = data.to(device)
        label   = label.to(device)
        outputs = model(data)

    for target, pre_d in zip(label, outputs):
        y_truth.append(target.cpu().numpy())
        y_pred.append(pre_d[-1].cpu().numpy())
    pbar.update(1)
pbar.close()
y_truth = np.array(y_truth).reshape(len)
y_pred  = np.array(y_pred).reshape(len)

r2 = R2(y_truth, y_pred)
mpe = MPE(y_truth, y_pred)
print(f"R2:{r2}\tMPE:{mpe}")
import matplotlib.pyplot as plt

# 假设有一组数据点
x = range(len)

# 创建散点图
plt.scatter(x, y_truth,label="target")
plt.scatter(x, y_pred,label="pred")
plt.title(f"R2:{round(r2,4)}\tMPE:{round(mpe,4)}")
# 显示图表
plt.legend()
plt.tick_params(direction='in')
plt.savefig(f"./logs/{model_name}_val.png")
plt.show()