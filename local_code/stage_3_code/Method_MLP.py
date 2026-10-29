'''
Stage 2 Multi-Layer Perceptron (PyTorch) for 10-class classification
of 784-dimensional inputs.

Everything you might want to change for experiments (layers, activation,
dropout, loss, optimizer, learning rate, batch size, epochs) is an attribute,
so scripts can set it without editing this file.
'''

from local_code.base_class.method import method
import numpy as np
import torch
from torch import nn


class Method_MLP(method, nn.Module):
    data = None

    # ---- hyper-parameters (override these from the script) ----
    input_dim = 784
    output_dim = 10
    hidden_dims = [256, 128]        # one entry per hidden layer
    activation = 'relu'             # 'relu', 'tanh', 'sigmoid', 'leaky_relu'
    dropout = 0.2                   # 0 disables dropout
    loss_name = 'cross_entropy'     # 'cross_entropy', 'nll', 'mse'
    optimizer_name = 'adam'         # 'adam', 'adamw', 'sgd', 'rmsprop'
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 128
    max_epoch = 20
    device_name = 'cpu'             # 'cpu', 'cuda', 'mps', or 'auto'
    # ------------------------------------------------------------

    def __init__(self, mName=None, mDescription=None):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.history = {'train_loss': [], 'train_acc': []}
        self.net = None

    # ---------- model construction ----------
    def get_activation(self):
        return {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
                'leaky_relu': nn.LeakyReLU(0.01)}[self.activation]

    def build_model(self):
        layers = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(self.get_activation())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        # last layer outputs raw scores (logits); the loss applies softmax itself
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

    def get_device(self):
        if self.device_name == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device_name)

    def forward(self, x):
        return self.net(x)

    # ---------- loss / optimizer ----------
    def compute_loss(self, logits, y):
        if self.loss_name == 'cross_entropy':
            return nn.functional.cross_entropy(logits, y)
        if self.loss_name == 'nll':
            return nn.functional.nll_loss(nn.functional.log_softmax(logits, dim=1), y)
        if self.loss_name == 'mse':
            probs = nn.functional.softmax(logits, dim=1)
            one_hot = nn.functional.one_hot(y, self.output_dim).float()
            return nn.functional.mse_loss(probs, one_hot)
        raise ValueError('unknown loss: ' + self.loss_name)

    def make_optimizer(self):
        params = self.parameters()
        lr, wd = self.learning_rate, self.weight_decay
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    # ---------- training / testing ----------
    # (named train_model instead of train, because nn.Module already
    #  uses train()/eval() to switch dropout on and off)
    def train_model(self, X, y):
        device = self.get_device()
        self.to(device)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True)
        optimizer = self.make_optimizer()
        self.history = {'train_loss': [], 'train_acc': []}

        for epoch in range(self.max_epoch):
            self.train()  # dropout on
            total_loss, correct, count = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = self.forward(xb)
                loss = self.compute_loss(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                count += len(yb)

            epoch_loss = total_loss / count
            epoch_acc = correct / count
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            print('Epoch: %3d  Loss: %.4f  Train accuracy: %.4f' % (epoch + 1, epoch_loss, epoch_acc))

    def test(self, X):
        device = self.get_device()
        self.eval()  # dropout off
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 1024):
                logits = self.forward(X_t[i:i + 1024].to(device))
                preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        print('method running...')
        self.build_model()
        print(self.net)
        print('device:', self.get_device())
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            