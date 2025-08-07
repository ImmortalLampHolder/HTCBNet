import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, SleepLoss, callback, optimizer, epoch, epoch_step,
                      epoch_step_val,batch_size, trainloader, valloader,device, epochs, cuda, save_period, save_dir, model_name="best_epoch_weights"):
    loss = 0.
    val_loss = 0.
    print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(trainloader):
        if iteration >= epoch_step:
            break

        data, label = batch
        with torch.no_grad():
            if cuda:
                data = data.to(device)
                label = label.to(device)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model_train(data)
        loss_value = SleepLoss(outputs, label)
        # ----------------------#
        #   反向传播
        # ----------------------#
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
        optimizer.step()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3)
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(valloader):
        if iteration >= epoch_step_val:
            break
        data, label = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                data = data.to(device)
                label = label.to(device)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train_eval(data)
            loss_value = SleepLoss(outputs, label)

        val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)
    pbar.close()
    print('Finish Validation')
    callback.append_loss(epoch,loss/epoch_step,val_loss/epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    if ema:
        save_state_dict = ema.ema.state_dict()
    else:
        save_state_dict = model.state_dict()

    if (epoch + 1) % save_period == 0 or epoch + 1 == epochs:
        torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
        epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    if len(callback.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(callback.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(save_state_dict, os.path.join(save_dir, f"{model_name}.pth"))
    torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))