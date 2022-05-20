import datetime
import os
import time

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import models
import utils

import csv

__all__ = [
    "UNET",
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
]

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
def evaluate(model, data_loader, device, num_classes, test_only = False):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]
            confmat.update(target.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()

        if test_only == True:
            with open('result.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["ScoreID", "ScoreValue"])
                _, _, ius = confmat.compute() 
                ius = ius.cpu().numpy()* 100
                mius = ius.mean()
                record_len = len(ius)
                for cnt, iu in enumerate(ius):
                    writer.writerow([f"class_{cnt}", ius[cnt]])
                for i in range(100000):
                    if i > mius * 1000:
                        writer.writerow([f"id_{i}", 0])
                    else:
                        writer.writerow([f"id_{i}", 1])             
    return confmat
            
def save_csv(pred_loader, data_loader, device, num_classes):
    confmat = utils.ConfusionMatrix(num_classes)
    with torch.inference_mode():
        for pred, target in zip(pred_loader, data_loader):
            output, target = pred.to(device), target.to(device)
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        _, _, ius = confmat.compute()
    
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Class", "IoU"])
        for cnt, iu in enumerate(ius.item() * 100):
            writer.writerow([f"class_{cnt}", iu.item()])
            
    return iu.mean().item() * 100

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255) + \
            0.1*utils.DiceLoss()(x, target)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]

def get_transform(train):
    if train:
        return utils.SegmentationPresetTrain(base_size=520, crop_size=480)
    else:
        return utils.SegmentationPresetEval(base_size=520)

### (1) Configuration (e.g., folder paths) 
data_path = '/content/drive/MyDrive/2022_SMWU_Deep_Learning/segmentation' # 'VOC2012'
model_name =  "deeplabv3_resnet50" # "UNET" # 
output_dir = 'output_' + model_name # "/content/output" #
last_ckpt_dir = 'last_ckpt+' + model_name 
saved_ckpt_path = 'output_deeplabv3_resnet50/model_epochXX.pth' # Path to your model 

test_only = False
resume = False
start_epoch = 0
print_freq = 10

device = 'cuda'
num_classes = 21

if os.path.exists(data_path):
    data_download = False
else:
    data_download = True
    
if test_only:
    resume = True

## (2) define hyperparameters
batch_size = 8
epochs = 50
lr =0.02
aux_loss = False
momentum = 0.9
weight_decay = 1e-4

## (3) load dataset
dataset = torchvision.datasets.VOCSegmentation(data_path, image_set="train", \
    transforms=get_transform(True), download = data_download)
dataset_test = torchvision.datasets.VOCSegmentation(data_path, image_set="val", \
    transforms=get_transform(False), download = data_download)

train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=16,
    collate_fn=utils.collate_fn,
    drop_last=True,
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, sampler=test_sampler, num_workers=16, collate_fn=utils.collate_fn
)

## (4) define model
model = models.__dict__[model_name](
    in_channels = 3,
    num_classes=num_classes,
    aux_loss=aux_loss,
    )
model.to(device)

## (5) define optimizer
if aux_loss:
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
    ]
    params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    params_to_optimize.append({"params": params, "lr": lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

iters_per_epoch = len(data_loader)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda x: (1 - x / (iters_per_epoch * (epochs))) ** 0.9
)

# saved_ckpt_path = '....../XXX.pth'
if resume == True:
    checkpoint = torch.load(saved_ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=not prep_submit)
    if not test_only:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

confmat = evaluate(model, data_loader_test, device=device, \
    num_classes=num_classes, test_only = test_only)
print(confmat)

utils.mkdir(output_dir)
utils.mkdir(last_ckpt_dir)

start_time = time.time()
for epoch in range(start_epoch, epochs):
    train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq)
    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    print(confmat)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
    torch.save(checkpoint, os.path.join(last_ckpt_dir, "last_ckpt.pth"))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Training time {total_time_str}")
