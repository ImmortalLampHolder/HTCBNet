import numpy as np
import os
if __name__ == "__main__":
    base_path = "./dataset"
    dataset = np.load(os.path.join(base_path,"dataset.npy"))
    np.random.shuffle(dataset)
    traindata = dataset[:3200,...]
    testdata  = dataset[3200:,...]
    np.save(os.path.join(base_path,"train_data.npy"),traindata[...,1:])
    np.save(os.path.join(base_path, "train_label.npy"), traindata[..., :1])
    np.save(os.path.join(base_path, "val_data.npy"), testdata[..., 1:])
    np.save(os.path.join(base_path, "val_label.npy"), testdata[..., :1])