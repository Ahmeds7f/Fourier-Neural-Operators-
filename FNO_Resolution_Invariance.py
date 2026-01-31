import torch
import torch.nn.functional as F
from neuralop import LpLoss
from models import get_fno
device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.load('./Data.pt')
test_input = data["test_in"]
test_output = data["test_sol"]

indices = torch.randint(0, len(test_input), (32,))
test_input = test_input[indices].to(device)
test_output = test_output[indices].to(device)

params = {"hidden_channels": 128, "n_layers": 6, "n_modes": 20}
model = get_fno(**params).to(device)

checkpoint = torch.load("Finetune_FNO_final.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

loss_fn = LpLoss(d=2, p=2, reduction='mean')

resolutions = [32, 64, 128]

with torch.no_grad():
    for res in resolutions:
        test_in_new = F.interpolate(test_input, size = (res,res), mode="bicubic", align_corners=False)
        test_out_new = F.interpolate(test_output, size = (res,res), mode="bicubic", align_corners=False)

        predictions = model(test_in_new)

        loss = loss_fn(predictions, test_out_new)

        print(f"{res}x{res:<10} | {loss:.5f}")
