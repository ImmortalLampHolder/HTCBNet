import math
import os.path

import torch
import pandas as pd
import os.path as op
from nets.network import HTCBNet
import numpy as np
import seaborn as sns
from utils.metrics import R2,MPE
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'

model_name = "HTCBNet"


class JRCModel():
    def __init__(self,weights_path=f"./weights/{model_name}.pth"):
        self.model = HTCBNet()
        state_dict = torch.load(weights_path)
        self.model.load_state_dict(state_dict)
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.eval()
        self.layers = list()
        self.add_hooks()

    def add_hooks(self, layer_name=''):
        def feature_forward_hook(module, fea_in, fea_out):
            self.layers.append(fea_out.detach().cpu().numpy())

        for (name, module) in self.model.named_modules():
            if 'sign' in name:
                module.register_forward_hook(hook=feature_forward_hook)

    def on_predict_end(self,save_path):
        if len(self.layers[0].shape) > 3:
            for index,attn in enumerate(self.layers[-2:]):
                shape = attn.shape
                if len(shape) == 5:# local
                    for item,attni in enumerate(attn[0]):
                        graph = np.zeros((400, 400))
                        for depth,attnij in enumerate(attni):
                            coordx = depth*50
                            graph[coordx:coordx+50,coordx:coordx+50] = attnij
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(graph, annot=True, cmap="bwr")
                        plt.xlabel("Key Tokens")
                        plt.ylabel("Query Tokens")
                        plt.savefig(f"{save_path}/attn{index}_local_{item}.png")
                        plt.close()
                        del graph

                elif len(shape) == 4:# gobal
                    for item, attni in enumerate(attn[0]):
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(attni, annot=True, cmap="bwr")
                        plt.xlabel("Key Tokens")
                        plt.ylabel("Query Tokens")
                        plt.savefig(f"{save_path}/attn{index}_gobal_{item}.png")
                        plt.close()

        else:
            for index,feat in enumerate(self.layers):
                if feat.shape[-1] == 800:
                    feat = feat[:,:,:400]
                heatmap = np.mean(feat,axis=(0,1))
                np.save(f"{save_path}/feature{index}.npy",heatmap)
                plt.scatter(np.linspace(0,100,400),heatmap)
                plt.xlabel("Coordination")
                plt.ylabel("Feature Activation Value")
                plt.savefig(f"{save_path}/feature{index}.png")
                plt.close()
        self.layers.clear()

    def __call__(self,x,save_path):
        x = torch.from_numpy(x.reshape(1, 1, 400)).to(self.device)
        output = self.model(x).cpu().detach().numpy()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.on_predict_end(save_path)
        return output[0][0] * 20


if __name__ == "__main__":
    y_pred = list()
    y_truth = list()
    predictor = JRCModel()
    base_path = "./dataset"
    outline = np.load(op.join(base_path, "barton.npy"))

    for item,line in enumerate(outline):
        data  = line[1:]
        label = line[0]
        perdict = predictor(data,f"./heatmap/barton{item}")
        y_truth.append([label])
        y_pred.append(perdict)

    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)
    r2 = R2(y_truth, y_pred)
    mpe = MPE(y_truth, y_pred)
    print(f"R2:{r2}\tMPE:{mpe}")
    x = range(1,len(y_truth)+1)

    # 创建散点图
    plt.scatter(x, y_truth, label="target")
    plt.scatter(x, y_pred, label="pred")
    plt.title(f"R2:{round(r2,4)}  MPE:{round(mpe,4)}")
    # 显示图表
    plt.legend()
    plt.tick_params(direction='in')
    plt.savefig(f"./logs/{model_name}_test.png")
    plt.show()
    frame = pd.DataFrame(y_pred)
    writer = pd.ExcelWriter(f"./logs/{model_name}_test.xlsx")
    frame.to_excel(writer)
    writer.close()
