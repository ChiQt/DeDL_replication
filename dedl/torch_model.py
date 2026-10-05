"""PyTorch implementation of the two-branch response model."""
import numpy as np


class TorchModel:
    """Two explicit MLP branches; scalar c is shared by all users."""

    def __init__(self, dimension, weights, initial_scale, learning_rate):
        import torch
        from torch import nn
        torch.set_num_threads(2)
        self.torch = torch

        class Network(nn.Module):
            def __init__(self):
                super().__init__()

                def branch(outputs):
                    return nn.Sequential(
                        nn.Linear(dimension, 20), nn.ReLU(),
                        nn.Linear(20, 20), nn.ReLU(),
                        nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, outputs)
                    )

                self.a = branch(1)
                self.b = branch(3)
                self.c = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

            def forward(self, x, treatment):
                index = self.a(x).squeeze(-1) + (self.b(x) * treatment).sum(-1)
                return self.c * torch.sigmoid(index)

        self.net = Network()
        dense_layers = [layer for branch in [self.net.a, self.net.b]
                        for layer in branch if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            for layer, (kernel, bias) in zip(dense_layers, weights):
                layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
                layer.bias.copy_(torch.from_numpy(bias))
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-7
        )

    def bind_data(self, x, treatment, y):
        self.x = self.torch.from_numpy(x)
        self.treatment = self.torch.from_numpy(treatment)
        self.y = self.torch.from_numpy(y)

    def train_epoch(self, order, batch_size):
        self.net.train()
        for start in range(0, len(order), batch_size):
            users = order[start:start + batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.net(self.x[users], self.treatment[users])
            loss = ((predictions - self.y[users]) ** 2).mean()
            loss.backward()
            self.optimizer.step()

    def predict(self, x, treatment):
        self.net.eval()
        with self.torch.no_grad():
            return self.net(self.torch.from_numpy(x), self.torch.from_numpy(treatment)).numpy()

    def parameters(self, x):
        with self.torch.no_grad():
            features = self.torch.from_numpy(x)
            return np.column_stack([
                self.net.a(features).numpy(), self.net.b(features).numpy(),
                np.full(len(x), float(self.net.c))
            ]).astype(np.float64)

    def save(self, path):
        self.torch.save(self.net.state_dict(), str(path) + ".pt")

    def versions(self):
        return {"torch": self.torch.__version__, "numpy": np.__version__}
