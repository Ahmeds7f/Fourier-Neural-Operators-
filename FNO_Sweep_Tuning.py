import torch
import wandb
from neuralop import Trainer, LpLoss
from torch.utils.data import TensorDataset, DataLoader
from models import CNN, get_fno

device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.login()

data = torch.load('./Data.pt')
train_input = data["train_in"]
train_output = data["train_sol"]
test_input = data["test_in"]
test_output = data["test_sol"]

def get_loaders(batch_size = 32):
    train_dataset = TensorDataset(train_input, train_output)
    test_dataset = TensorDataset(test_input, test_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {'x': torch.stack([i[0] for i in x]), 'y': torch.stack([i[1] for i in x])})
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {'x': torch.stack([i[0] for i in x]), 'y': torch.stack([i[1] for i in x])})

    return train_loader, test_loader


def sweep():
    with wandb.init() as run:
        config = wandb.config

        model = get_fno(n_modes=config.n_modes, hidden_channels=config.hidden_channels, n_layers=config.n_layers)
        model = model.to(device)

        train_loader, test_loader = get_loaders(batch_size=config.batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        ModelTrainer = Trainer(model=model, n_epochs=15, wandb_log=True, device=device, eval_interval=3, verbose=True)


        ModelTrainer.train(
            train_loader=train_loader,
            test_loaders={'default': test_loader},
            optimizer=optimizer,
            training_loss=LpLoss(d=2, p=2, reduction='sum'),
            scheduler=scheduler
        )

sweep_config_FNO  = {
    'method': 'bayes',
    'metric': {'name': 'default_l2', 'goal': 'minimize'},
    'parameters': {
    'lr': {'min': 1e-4, 'max': 5e-3},
    'n_modes': {'values': [8, 12, 16, 20]},
    'hidden_channels': {'values': [32, 64, 128]},
    'n_layers': {'values': [4, 5, 6]},
    'batch_size': {"values": [12, 24, 32]}}
}


sweep_id = wandb.sweep(sweep_config_FNO, project="MIT-FNO-Project")
wandb.agent(sweep_id, function=sweep, count=20)
