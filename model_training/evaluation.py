import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, norm, title=None, name=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(norm["mean"])
    std = np.array(norm["std"])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, wrap=True)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('test.jpg' if not name else name)


def evaluate(model, image_datasets, norm, exp_name='exp'):
    model.eval()
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    class_names = image_datasets["train"].classes

    all_preds = []
    all_labels = []

    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        all_labels.extend(labels.tolist())

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())

            titles = [class_names[x] + '\n ' + str(o.detach().numpy().max()) for x, o in zip(preds, outputs)]
            out = torchvision.utils.make_grid(inputs)
            imshow(out, norm, title=titles, name=f"{exp_name}_{i}.jpg")

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(f"{exp_name}_confusion_matrix.png")