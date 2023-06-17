import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 定义真实的数据生成函数
def true_function(x):
    return torch.sin(0.6 * x)


# 生成训练数据
def generate_data():
    x = torch.linspace(-5, 5, 100)
    y = true_function(x) + torch.randn_like(x) * 0.1
    return x, y


# 定义变分自编码器模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 100)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar


# 训练模型
def train(latent_dim):
    model = VAE(latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x_train, y_train = generate_data()
    x_train = x_train.unsqueeze(0)
    y_train = y_train.unsqueeze(0)

    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        decoded, mu, logvar = model(x_train)

        # 计算重构损失和KL散度损失
        reconstruction_loss = criterion(decoded, y_train)
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = reconstruction_loss + kl_divergence_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

    return model


# 测试模型
def main(model):
    x_test = torch.linspace(-5, 5, 100)
    x_test = x_test.unsqueeze(0)
    decoded, _, _ = model(x_test)
    y_pred = decoded.squeeze(0)

    return x_test.squeeze(), y_pred


# 执行训练和测试
latent_dim = 2
model = train(latent_dim)
x_test, y_pred = main(model)

# 可视化结果
x_train, y_train = generate_data()
plt.scatter(x_train, y_train, label="Training Data")
plt.plot(x_test, y_pred.detach().numpy(), color='r', label="Variational Bayesian Regression")
plt.plot(x_test, true_function(x_test), color='g', label="True Function")
plt.legend()
plt.show()