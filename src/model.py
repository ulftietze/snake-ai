import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './saves'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:

    def __init__(self, model, learning_rate, gamma):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, state_next, is_game_over):
        state = torch.tensor(state, dtype=torch.float)
        state_next = torch.tensor(state_next, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            state_next = torch.unsqueeze(state_next, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_game_over = (is_game_over,)

        # 1. predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()

        for index in range(len(is_game_over)):
            Q_new = reward[index]
            if not is_game_over[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(state_next[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
