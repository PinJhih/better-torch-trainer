import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

    def fit_dataloader(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        val_loader=None,
    ):
        for epoch in range(epochs):
            training_info = {"loss": 0.0, "val_loss": None}

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}: ")
            for step, data in enumerate(progress_bar):
                X_train, y_train = data
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                self.model.train()
                outputs = self.model(X_train)
                loss = self.loss_fn(outputs, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                training_info["loss"] += loss.item()
                progress_bar.set_postfix(loss=(training_info["loss"] / (step + 1)))

            training_info["loss"] /= len(train_loader)

            if val_loader is not None:
                val_loss = 0.0
                num_val_batch = len(val_loader)

                with torch.no_grad():
                    self.model.eval()
                    for data in val_loader:
                        X_val, y_val = data
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)

                        outputs = self.model(X_val)
                        loss = self.loss_fn(outputs, y_val)
                        val_loss += loss.item()
                    training_info["val_loss"] = val_loss / num_val_batch
            progress_bar.set_postfix(training_info)
