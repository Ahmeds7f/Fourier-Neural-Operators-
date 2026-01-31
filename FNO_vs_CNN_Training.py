import torch
import wandb
from neuralop import Trainer, LpLoss
from torch.utils.data import TensorDataset, DataLoader
from models import CNN, get_fno

device = "cuda" if torch.cuda.is_available() else "cpu"

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

models_to_compare = [
    {"name": "FNO-Baseline-500k", "model_type": "FNO", "params": {"n_modes": 16, "hidden_channels": 20}},
    {"name": "CNN-Baseline-500k", "model_type": "CNN", "params": {"hidden_channels": 80, "n_layers": 8}}
]

if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=32)

    for item in models_to_compare:
        # 1. Initialize W&B Run
        run = wandb.init(
            project="MIT-FNO-Project",
            name=item["name"],
            entity="ahmeds7f-massachusetts-institute-of-technology",
            config=item["params"]
            )

        if item["model_type"] == "FNO":
            model = get_fno(**item["params"]).to(device)
        else:
            model = CNN(**item["params"]).to(device)

        trainer = Trainer(model= model ,n_epochs= 50, wandb_log= True, device= device, eval_interval= 5, verbose=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

        loss_fn = LpLoss(d=2, p=2, reduction='sum')

        trainer.train(
            train_loader=train_loader,
            test_loaders={'default': test_loader},
            optimizer=optimizer,
            training_loss=loss_fn,
            scheduler=scheduler
        )


        model_filename = f"{item['name']}_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': item["params"]
        }, model_filename)

        artifact = wandb.Artifact(item["name"], type="model")
        artifact.add_file(model_filename)
        run.log_artifact(artifact)

        run.finish()
        print(f"{item['name']} complete. Weights uploaded.")

    print("\n All comparisons finished! View your results on WandB.")
