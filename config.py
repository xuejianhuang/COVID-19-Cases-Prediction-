import torch


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=256
n_epochs=1000
optimizer="SGD"
optim_hparas={
    "lr":0.001,
    "momentum":0.9
}
early_stop=60
save_path="model/model.pth"
dropout=0.2
