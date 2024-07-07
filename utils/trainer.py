import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: callable,
        device="cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def fit_tensor(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 1,
        X_val=None,
        y_val=None,
        acc_fn=None,
    ):
        do_val = (X_val is not None) and (y_val is not None)
        X_train.to(self.device)
        y_train.to(self.device)
        if do_val:
            X_val.to(self.device)
            y_val.to(self.device)

        progress_bar = tqdm(range(epochs), unit="epoch", desc="Training")
        for _ in progress_bar:
            self.model.train()
            outputs = self.model(X_train)
            loss = self.loss_fn(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            training_info = {"loss": loss.item()}
            if acc_fn is not None:
                acc = acc_fn(outputs, y_train)
                training_info["acc"] = acc

            if do_val:
                self.model.eval()
                outputs = self.model(X_val)
                val_loss = self.loss_fn(outputs, y_val)
                training_info["val_loss"] = val_loss.item()

                if acc_fn is not None:
                    val_acc = acc_fn(outputs, y_val)
                    training_info["val_acc"] = val_acc
            progress_bar.set_postfix(training_info)
