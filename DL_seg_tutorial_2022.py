import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import utils

from torch import nn

import deeplabv3

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
        
def evaluate(model, data_loader, device, num_classes):
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

    return confmat

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]

def get_transform(train):
    if train:
        return utils.SegmentationPresetTrain(base_size=520, crop_size=480)
    else:
        return utils.SegmentationPresetEval(base_size=520)

def main_loop():
    num_classes = 21
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 10
    aux_loss = False
    device = 'cuda'
    last_ckpt_dir = 'last_ckpt'
    resume = False
    test_only = False

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

    model = deeplabv3.__dict__[model_name](
        num_classes=num_classes,
        aux_loss=aux_loss,
        )
        
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
    ]
    
    if aux_loss:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": lr * 10})
        
    optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)

    iters_per_epoch = len(data_loader)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (epochs))) ** 0.9
    )

    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=not test_only)
        if not test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1

    if test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

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

data_path = '../VOCtrainval_11_May_2012' # E:\Work\VOC2012\VOCtrainval_11-May-2012'
model_name = "deeplabv3_resnet50"
output_dir = 'output' # "/content/output"
batch_size = 8
epochs =30
lr =0.01 
data_download = False

main_loop()