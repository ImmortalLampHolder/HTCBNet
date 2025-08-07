import os
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch

class LossHistory():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15

        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, log_dir ):
        super(EvalCallback, self).__init__()
        self.log_dir = log_dir
        self.maps = [0]
        self.epoches = [0]
        self.logpath = os.path.join(self.log_dir, "epoch_acc.txt")
        with open(self.logpath,'w') as f:
            f.write('0\n')
            f.close()

    def on_epoch_end(self, model_eval,valloader,device,cuda,epoch_step_val,epoch,epochs):
        print('Start Get Acc')
        acc = 0.
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3)
        for iteration, batch in enumerate(valloader):
            temp = 0.
            if iteration >= epoch_step_val:
                break
            data, label = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    data = data.to(device)
                    label = label.to(device)
                outputs = model_eval(data)
                temp= ((torch.argmax(outputs,dim=1) == torch.argmax(label,dim=1)).type(torch.float).sum().item())
            pbar.set_postfix(**{'val acc': temp / (iteration + 1)})
            acc += temp
            pbar.update(1)
        pbar.close()
        with open(self.logpath,'a') as f:
            f.write(str(acc/epoch_step_val) + '\n')
            f.close()
        print('Finish Validation')

class History():
    def __init__(self,log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir)
        self.losses    = []
        self.val_loss = []
        self.acc     = []
        self.val_acc = []
        self.writer = SummaryWriter(self.log_dir)

    def append(self,epoch, loss, val_loss,acc,val_acc):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.acc.append(acc)
        self.val_acc.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('acc', acc, epoch)
        self.writer.add_scalar('val_acc', val_acc, epoch)
        self.loss_plot()
        self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.acc))

        plt.figure()
        plt.plot(iters, self.acc, 'red', linewidth=2, label='train acc')
        plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='val acc')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")
