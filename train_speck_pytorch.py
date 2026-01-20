import os
from dataclasses import dataclass
from typing import Callable

import speck as sp
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


bs = 4000
wdir = "./good_trained_nets/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cyclic_lr(num_epochs: int, high_lr: float, low_lr: float) -> Callable[[int], float]:
    def res(i: int) -> float:
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (
            high_lr - low_lr
        )

    return res


class ResNet1D(nn.Module):
    def __init__(
        self,
        pairs: int = 2,
        num_blocks: int = 3,
        num_filters: int = 32,
        num_outputs: int = 1,
        d1: int = 64,
        d2: int = 64,
        word_size: int = 16,
        ks: int = 3,
        depth: int = 5,
        reg_param: float = 1e-4,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.num_blocks = num_blocks
        self.word_size = word_size
        self.depth = depth
        self.final_activation = final_activation
        self.reg_param = reg_param

        in_channels = 2 * num_blocks * pairs

        self.conv01 = nn.Conv1d(in_channels, num_filters, kernel_size=1, padding=0)
        self.conv02 = nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv03 = nn.Conv1d(in_channels, num_filters, kernel_size=5, padding=2)
        self.conv04 = nn.Conv1d(in_channels, num_filters, kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(num_filters * 4)
        self.relu = nn.ReLU()

        self.res_blocks = nn.ModuleList()
        current_filters = num_filters * 4
        current_ks = ks
        for _ in range(depth):
            conv1 = nn.Conv1d(
                current_filters,
                current_filters,
                kernel_size=current_ks,
                padding=current_ks // 2,
            )
            bn1 = nn.BatchNorm1d(current_filters)
            conv2 = nn.Conv1d(
                current_filters,
                current_filters,
                kernel_size=current_ks,
                padding=current_ks // 2,
            )
            bn2 = nn.BatchNorm1d(current_filters)
            self.res_blocks.append(nn.ModuleDict({"conv1": conv1, "bn1": bn1, "conv2": conv2, "bn2": bn2}))
            current_ks += 2

        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.Linear(current_filters * word_size, 512)
        self.bn_fc0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, d1)
        self.bn_fc1 = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.bn_fc2 = nn.BatchNorm1d(d2)
        self.fc_out = nn.Linear(d2, num_outputs)

        if final_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif final_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def l2_penalty(self) -> torch.Tensor:
        if self.reg_param <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                l2 = l2 + module.weight.pow(2).sum()
        return self.reg_param * l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.view(batch, self.word_size, -1).permute(0, 2, 1)
        convs = torch.cat(
            [self.conv01(x), self.conv02(x), self.conv03(x), self.conv04(x)], dim=1
        )
        x = self.bn0(convs)
        x = self.relu(x)

        for block in self.res_blocks:
            residual = x
            out = block["conv1"](x)
            out = block["bn1"](out)
            out = self.relu(out)
            out = block["conv2"](out)
            out = block["bn2"](out)
            out = self.relu(out)
            x = residual + out

        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc0(x)
        x = self.bn_fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.out_act(x)
        return x


@dataclass
class TrainResult:
    model: nn.Module
    history: dict


def train_speck_distinguisher(
    num_epochs: int, num_rounds: int = 7, depth: int = 1, pairs: int = 1
) -> TrainResult:
    print("pairs = ", pairs)
    print("num_rounds = ", num_rounds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(pairs=pairs, depth=depth, reg_param=1e-5).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cyclic_lr(5, 0.002, 0.0001)
    )

    x_train, y_train = sp.make_train_data(int(10**6), num_rounds, pairs=pairs)
    x_eval, y_eval = sp.make_train_data(int(10**5), num_rounds, pairs=pairs)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=bs,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_eval, dtype=torch.float32),
            torch.tensor(y_eval, dtype=torch.float32),
        ),
        batch_size=bs,
        shuffle=False,
    )

    history = {"val_acc": [], "val_loss": []}
    best_val_loss = float("inf")

    os.makedirs(wdir, exist_ok=True)
    best_path = (
        f"{wdir}best{num_rounds}r_depth{depth}_num_epochs{num_epochs}_pairs{pairs}.pt"
    )

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb) + model.l2_penalty()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                predicted = (preds >= 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.numel()
        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - val_loss: {epoch_val_loss:.6f} - val_acc: {epoch_val_acc:.6f}"
        )

    final_path = (
        f"{wdir}model_{num_rounds}r_depth{depth}_num_epochs{num_epochs}_pairs{pairs}.pt"
    )
    torch.save(model.state_dict(), final_path)
    print("Best validation accuracy: ", max(history["val_acc"]))
    return TrainResult(model=model, history=history)


if __name__ == "__main__":
    rounds = [7]
    pairs = 2
    for r in rounds:
        train_speck_distinguisher(num_epochs=40, num_rounds=r, depth=5, pairs=pairs)
