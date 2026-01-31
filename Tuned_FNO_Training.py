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

params = {"hidden_channels" :128, "n_layers" : 6, "n_modes":20}
if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=32)

    run = wandb.init(
        project="MIT-FNO-Project",
        name= "Finetuned_FNO",
        entity="ahmeds7f-massachusetts-institute-of-technology",
        config=params
        )

    model = get_fno(**params).to(device)

    trainer = Trainer(model= model ,n_epochs= 16, wandb_log= True, device= device, eval_interval= 3, verbose=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003085111061679635, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    loss_fn = LpLoss(d=2, p=2, reduction='sum')

    trainer.train(
        train_loader=train_loader,
        test_loaders={'default': test_loader},
        optimizer=optimizer,
        training_loss=loss_fn,
        scheduler=scheduler
    )


    model_filename = "Finetune_FNO_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': params
    }, model_filename)

    artifact = wandb.Artifact("Finetune_FNO", type="model")
    artifact.add_file(model_filename)
    run.log_artifact(artifact)

    run.finish()
    print("Finetune_FNO complete. Weights uploaded.")
