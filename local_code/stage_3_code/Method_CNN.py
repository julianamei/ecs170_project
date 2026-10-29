'''
Configurable CNN (PyTorch) for stage 3 image classification.
All settings are attributes so scripts can change them without editing this file.
'''

from local_code.base_class.method import method
import numpy as np
import torch
from torch import nn


class Method_CNN(method, nn.Module):
    data = None

    # ---- hyper-parameters (override these from the script) ----
    input_shape = (1, 28, 28)       # (channels, height, width)
    output_dim = 10
    conv_channels = [32, 64]        # one entry per conv layer
    kernel_size = 3
    padding = 1
    stride = 1
    pool = 'max'                    # 'max', 'avg', or None
    pool_size = 2                   # pooling after every conv layer
    hidden_dims = [128]             # fully connected layers after flatten
    activation = 'relu'             # 'relu', 'tanh', 'sigmoid', 'leaky_relu'
    dropout = 0.3                   # applied after FC hidden layers; 0 disables
    loss_name = 'cross_entropy'     # 'cross_entropy', 'nll', 'mse'
    optimizer_name = 'adam'         # 'adam', 'adamw', 'sgd', 'rmsprop'
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 128
    max_epoch = 10
    device_name = 'auto'            # 'cpu', 'cuda', 'mps', or 'auto'
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
        conv = []
        prev = self.input_shape[0]
        for ch in self.conv_channels:
            conv.append(nn.Conv2d(prev, ch, self.kernel_size,
                                  stride=self.stride, padding=self.padding))
            conv.append(self.get_activation())
            if self.pool == 'max':
                conv.append(nn.MaxPool2d(self.pool_size))
            elif self.pool == 'avg':
                conv.append(nn.AvgPool2d(self.pool_size))
            prev = ch
        self.conv_net = nn.Sequential(*conv)

        # work out the flattened size by passing a dummy image through the conv part
        with torch.no_grad():
            flat = self.conv_net(torch.zeros(1, *self.input_shape)).numel()

        fc = [nn.Flatten()]
        prev = flat
        for h in self.hidden_dims:
            fc.append(nn.Linear(prev, h))
            fc.append(self.get_activation())
            if self.dropout > 0:
                fc.append(nn.Dropout(self.dropout))
            prev = h
        fc.append(nn.Linear(prev, self.output_dim))  # raw logits
        self.fc_net = nn.Sequential(*fc)
        self.net = nn.Sequential(self.conv_net, self.fc_net)

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
            self.train()
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
        self.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 512):
                logits = self.forward(X_t[i:i + 512].to(device))
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