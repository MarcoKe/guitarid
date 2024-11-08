import math
import os
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torcheval.metrics import MulticlassAccuracy, Mean
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

import wandb


def train_model(model, criterion, optimizer, dataloaders, num_epochs: int, metrics: dict = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # default to multi class accuracy if no metrics have been specified
    if not metrics:
        metrics = {"Acc": MulticlassAccuracy(device=device)}
    metrics_values = {key: 0 for key in metrics.keys()}

    avg_loss = Mean(device=device)
    best_loss = math.inf

    # keep track of best model in terms of val loss across epochs
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1} \n {'-' * 10}")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad() # do not accumulate gradients over multiple batches

                    with torch.set_grad_enabled(phase=="train"): # do not compute gradients on validation set
                        # forward
                        outputs = model(inputs)
                        _, predicted_classes = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                        # update metrics
                        for metric in metrics.keys():
                            metrics[metric].update(predicted_classes, labels.data)
                        avg_loss.update(loss, weight=inputs.size(0))

                for metric in metrics.keys():
                    metrics_values[metric] = metrics[metric].compute()

                loss_value = avg_loss.compute()
                if phase == "val" and loss_value < best_loss:
                    torch.save(model.state_dict(), best_model_params_path)

                log_metrics(epoch, phase, metrics_values, loss_value)

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        return model


def log_metrics(epoch, phase, metrics, loss):
    metrics_str = ", ".join([f"{key}: {metrics[key]:.4f}" for key in metrics.keys()])
    print(f"{phase.capitalize()} | Loss: {loss:.4f}, {metrics_str}")
    wandb.log({"epoch": epoch, f"{phase}_loss": loss, **{f"{phase}_{key}": metrics[key] for key in metrics.keys()}})


def export_onnx(model, image_width, name):
    model.eval()
    torch_input = torch.randn(1, 3, image_width, image_width)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    onnx_program.save(name)


def load_data(data_dir, image_width, norm_coefs):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_width),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_coefs["mean"], norm_coefs["std"])
        ]),
        "val": transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(image_width),
            transforms.ToTensor(),
            transforms.Normalize(norm_coefs["mean"], norm_coefs["std"])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                  shuffle=True, num_workers=0)
                   for x in ["train", "val"]}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(image_datasets["train"].targets),
                                         y=image_datasets["train"].targets)

    return image_datasets, dataloaders, class_names, num_classes, class_weights


def load_model(num_classes, freeze_weights=True):
    # load pre-trained weights
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # change last layer to accomodate num classes
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes))

    # whether to freeze all weights except for the head of the model
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def save_model(model, name, onnx=True, torchscript=True):
    if onnx:
        export_onnx(model, image_width, f"{name}.onnx")

    if torchscript:
        model_scripted = torch.jit.script(model)
        model_scripted.save(f"{name}.pt")


if __name__ == '__main__':
    data_dir = "datasets/reverb_simple"
    image_width = 224
    norm_coefs = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.init(
        # set the wandb project where this run will be logged
        project="guitarid",
        # track hyperparameters and run metadata
        config={
            "architecture": "mobilenet-v3",
            "dataset": "reverb_simple",
        }
    )

    # load data / create data loaders
    image_datasets, dataloaders, class_names, num_classes, class_weights = load_data(data_dir, image_width, norm_coefs)

    # load model
    model = load_model(num_classes)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)) # class weighted CEL
    optimizer = optim.Adam(model.parameters())

    # train model
    model = train_model(model, criterion, optimizer, dataloaders,20)

    # save trained model
    save_model(model, "guitarid_model_v0.1.3_mobilenetv3")

    # evaluate(model, image_datasets, norm_coefs)

    print("Class names: ", class_names)










