import numpy as np
from nets.network import HTCBNet
from dataloader.sequence import Sequence
from torch.utils.data import DataLoader
from utils.callbacks import LossHistory,EvalCallback,History
import torch.nn as nn
import torch.optim as optim
from utils.utils import set_optimizer_lr,weights_init,get_lr_scheduler
import torch
import os
import torch.backends.cudnn as cudnn
from utils.utils import ModelEMA
from utils.fit import fit_one_epoch


if __name__ == "__main__":
    #训练相关参数
    cuda = True
    Init_Epoch = 0
    epochs = 100
    save_period = epochs
    save_dir = './logs'
    model_path = './logs/'
    #==============================
    #优化器参数
    #=================================
    optimizer_type = "adam"
    momentum = 0.999
    weight_decay = 5e-4
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    lr_decay_type = "step"
    #=================================
    #数据集参数
    #=================================
    data_path = "./dataset"
    batch_size  = 100
    train_data  = np.load(os.path.join(data_path,"train_data.npy"))
    train_label = np.load(os.path.join(data_path,"train_label.npy"))
    val_data    = np.load(os.path.join(data_path, "val_data.npy"))
    val_label   = np.load(os.path.join(data_path, "val_label.npy"))
    num_train = train_label.shape[0]
    num_val   = val_label.shape[0]
    traindataset = Sequence(train_data,train_label)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valdataset = Sequence(val_data,val_label)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
    #================================
    #设备检测
    device = torch.device('cuda' if torch.cuda.is_available() and cuda is True else 'cpu')
    #模型
    model = HTCBNet()
    weights_init(model)
    model.to(device)
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.to(device)
    # ----------------------------#
    #   权值平滑
    # ----------------------------#
    ema = ModelEMA(model_train)
    #损失函数
    loss = nn.MSELoss()
    #记录Loss
    #time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + model_name)
    # loss_history = LossHistory(log_dir)
    #
    # eval_callback = EvalCallback(log_dir)
    callback = LossHistory(log_dir)
    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    if ema:
        ema.updates = epoch_step * Init_Epoch

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#

    for epoch in range(Init_Epoch, epochs):
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, ema, loss, callback, optimizer, epoch, epoch_step,
                      epoch_step_val, batch_size,trainloader, valloader,device, epochs, cuda, save_period, save_dir,model_name)

