import torch
import time
import argparse
import wandb 

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from moco import MoCo
from utils import TwoCropsTransform, AverageMeter, ProgressMeter, accuracy

def train(train_loader, model, criterion, optimizer, epoch, args) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available() and args.use_gpu:
            images[0] = images[0].cuda()
            images[1] = images[1].cuda()

        # compute output
        output = model(im_q=images[0], im_k=images[1])
        target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, top1.avg, top5.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MoCo Training on a single gpu")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--backbone", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--K", default=65536, type=int, help="queue size; number of negative keys")
    parser.add_argument("--m", default=0.999, type=float, help="moco momentum of updating key encoder")
    parser.add_argument("--T", default=0.07, type=float, help="softmax temperature")
    parser.add_argument("--data", default="./data", type=str, help="dataset root")
    parser.add_argument("--batch-size", default=256, type=int, help="batch size")
    parser.add_argument("--workers", default=4, type=int, help="data loader workers")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu or not")
    parser.add_argument("--aug-plus", action="store_true", help="use MoCo v2's augmentation")

    args = parser.parse_args()

    run = wandb.init(project="MoCo", name="MoCo-v2" if args.aug_plus else "MoCo-v1", config=vars(args))
    
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR100(
        root=args.data,
        train=True,
        download=True,
        transform=TwoCropsTransform(augmentation),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    model = MoCo(base_model=models.__dict__[args.backbone], dim=args.dim, K=args.K, m=args.m, T=args.T)
    model = model.cuda() if torch.cuda.is_available() and args.use_gpu else model

    criterion = torch.nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() and args.use_gpu else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
        run.log({"epoch": epoch})
        run.log({"train_loss": train_loss, "train_acc1": train_acc1, "train_acc5": train_acc5})
