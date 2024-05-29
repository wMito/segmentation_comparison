import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import random
from UNet import UNet
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainModel():
    def __init__(self, model, optimizer):
        self.model = model.to(device)
        self.optimizer = optimizer

    def train(self, train, val, plot_val, epochs, batch_size, plot=True, name=""):
        train_x, train_y = train
        all_losses_train = []
        all_losses_val = []
        all_dices = []
        all_ious = []
        for epoch in range(epochs):
            temp = list(zip(train_x, train_y))
            random.shuffle(temp)
            train_x, train_y = zip(*temp)
            train_x, train_y = list(train_x), list(train_y)
            with tqdm(range(0, len(train_x), batch_size), unit="batch") as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    x = train_x[i:i+batch_size]
                    y = train_y[i:i+batch_size]
                    x, y = torch.stack(x).type(torch.float), torch.stack(y).type(torch.float)
                    x, y = x.to(device), y.to(device)
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = self.loss_fun(pred, y)
                    all_losses_train.append(loss.item())
                    tepoch.set_postfix(loss=loss.item())
                    loss.backward()
                    self.optimizer.step()
                    if i % 1000 == 0:
                      # every 1000 try val
                      dice, iou, loss_val = self.validate(val, batch_size)
                      tqdm.write(f"IOU: {iou}, Dice: {dice}, Loss val: {loss_val}")
                      all_dices.append(dice)
                      all_ious.append(iou)
                      all_losses_val.append(loss_val.item())
                dice, iou, loss_val = self.validate(val, batch_size)
                self.visualise_state(plot_val, plot=False)
                tqdm.write(f"All of val IOU: {iou}, Dice: {dice}, Loss val: {loss_val}")
                all_dices.append(dice)
                all_ious.append(iou)
                all_losses_val.append(loss_val.item())
            pred.detach().cpu().min()
            pred = self.get_pred_mask(pred)
        self.visualise_state(plot_val, plot=True)
        try:
            state = {
              'epoch': epoch,
              'state_dict': self.model.state_dict(),
              'optimizer': self.optimizer.state_dict(),
                 }
            torch.save(state, f'state_dict/model_state_{epoch}_{name}.pth')
        except:
              pass

        if plot:
            self.plot_metrics(all_losses_train, all_losses_val, all_dices, all_ious)

    def loss_fun(self, pred, gt):
        bce = nn.BCELoss()
        return bce(pred, gt) + (1-self.dice_coef(gt, self.get_pred_mask(pred)))

    def validate(self, val, batch_size, subset=False):
        val_x, val_y = val
        with torch.no_grad():
          self.model.eval()
          # zmniejszenie danych w zbiorze walidacyjnym
          if subset:
              temp = list(zip(val_x, val_y))
              random.shuffle(temp)
              val_x, val_y = zip(*temp)
              val_x, val_y = list(val_x), list(val_y)
              val_x, val_y = val_x[0:len(val_x)//3], val_y[0:len(val_y)//3]
          all_losses = []
          all_dices = []
          all_ious = []
          for i in range(0, len(val_x), batch_size):
              x = val_x[i:i+batch_size]
              y = val_y[i:i+batch_size]
              #x, y = torch.stack(x).type(torch.float), torch.stack(y).type(torch.float)
              x, y = x.type(torch.float), y.type(torch.float)
              x, y = x.to(device), y.to(device)
              pred = self.model(x)
              loss = self.loss_fun(pred, y)
              all_losses.append(loss)
              pred = self.get_pred_mask(pred)
              all_dices.append(self.dice_coef(y, pred))
              all_ious.append(self.iou(y, pred))
          pred.detach().cpu().min()
          all_losses = torch.Tensor(all_losses)
          all_dices = torch.Tensor(all_dices)
          all_ious = torch.Tensor(all_ious)
        return torch.mean(all_dices), torch.mean(all_ious), torch.mean(all_losses)

    def visualise_state(self, plot_val, plot=True):
        X_val, y_val = plot_val
        all_losses = []
        all_dices = []
        all_ious = []
        with torch.no_grad():
            self.model.eval()
            for i in range(0, len(X_val)):
                x = X_val[i]
                y = y_val[i]
                x, y = x.type(torch.float), y.type(torch.float)
                x, y = x.to(device), y.to(device)
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                pred = self.model(x)
                loss = self.loss_fun(pred, y)
                all_losses.append(loss)
                pred = self.get_pred_mask(pred)
                all_dices.append(self.dice_coef(y, pred))
                all_ious.append(self.iou(y, pred))
                if plot:
                    fig, axs = plt.subplots(1, 2)

                    axs[0].imshow(x[0].permute(1,2,0).detach().cpu(), cmap='gray')
                    mask_rgb = y[0].detach().cpu().repeat((3, 1, 1)).permute((1,2,0))
                    mask_rgb_2 = torch.zeros(mask_rgb.shape, dtype=torch.int32)
                    mask_rgb_2[torch.all(mask_rgb == True, dim=2)] = torch.tensor([255, 0, 0], dtype=torch.int32)
                    axs[1].imshow(mask_rgb_2)
                    axs[1].imshow(pred[0].permute(1,2,0).detach().cpu(), cmap='gray', alpha=0.7)

                    plt.show()
        dice, iou, loss = torch.mean(torch.stack(all_dices)), torch.mean(torch.stack(all_ious)), torch.mean(torch.stack(all_losses))
        tqdm.write(f"------\n10 val pics results IOU: {iou}, Dice: {dice}, Loss val: {loss}\n------")

    def get_pred_mask(self, pred, threshold=0.5):
        return (pred >= threshold)

    def dice_coef(self, gt, pred):
        overlap = torch.sum(pred*gt)
        total_area = torch.sum(pred) + torch.sum(gt) + 0.0000001
        return torch.round(torch.mean((2*overlap)/total_area), decimals=5)

    def iou(self, gt, pred):
        overlap = (torch.sum(pred*gt))
        union = torch.sum(pred) + torch.sum(gt) - overlap + 0.0000001
        return torch.round(torch.mean(overlap/union), decimals=5)

    def plot_metrics(self, losses_train, losses_val, dices, ious):
        plt.plot(list(range(len(losses_train))), losses_train, label="train")
        plt.title("Losses train")
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(list(range(len(losses_val))), losses_val, label="val")
        plt.title("Losses val")
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(list(range(len(dices))), dices)
        plt.title("Dice metric")
        plt.xlabel("Every 1000 iter")
        plt.ylabel("Dice")
        plt.show()
        plt.plot(list(range(len(ious))), ious)
        plt.title("IOU metric")
        plt.xlabel("Every 1000 iter")
        plt.ylabel("IOU")
        plt.show()