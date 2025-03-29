import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import numpy.typing as npt


class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = "model.pth"):
        model_folder_path = "./model"
        os.makedirs(model_folder_path, exist_ok=True)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str = "model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))


class QTrainer:
    def __init__(self, model: Linear_QNet, lr: float, gamma: float, device: str):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state_old_: tuple[npt.NDArray[np.int_], ...],
        action_: tuple[list[int], ...],
        reward_: tuple[int, ...],
        state_new_: tuple[npt.NDArray[np.int_], ...],
        done_: tuple[bool, ...],
    ):
        state_old = torch.tensor(np.array(state_old_), dtype=torch.float, device=self.device)
        state_new = torch.tensor(np.array(state_new_), dtype=torch.float, device=self.device)
        action = torch.tensor(np.array(action_), dtype=torch.long, device=self.device)

        # 1: predict Q values with current state
        self.model.train()
        pred_action: torch.Tensor = self.model(state_old)  # list

        # 2: Q_new = r + y * max(next_predicted_Q_value) -> only do this if not done
        self.model.eval()
        target = pred_action.detach().clone()
        with torch.no_grad():
            for index, done in enumerate(done_):
                Q_new = reward_[index]
                if not done:
                    Q_new = Q_new + self.gamma * torch.max(self.model(state_new[index])).item()

                target[index][int(torch.argmax(action[index]).item())] = Q_new

        self.model.train()
        self.optimizer.zero_grad()
        loss: torch.Tensor = self.criterion(pred_action, target)
        loss.backward()
        self.optimizer.step()
