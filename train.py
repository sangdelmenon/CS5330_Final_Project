# Sangeeth Deleep Menon
# CS5330 Final Project - Model Training and Evaluation
# Spring 2026
#
# Trains an ObjectCNN / ObjectViT / MobileNetV2 on the collected dataset,
# evaluates per-class precision and recall on a held-out test set, and
# saves the best checkpoint for use by recognize_ar.py.
#
# Usage:
#   python train.py                               # CNN, 20 epochs
#   python train.py --model mobilenet             # MobileNetV2 (best generalisation)
#   python train.py --model vit --epochs 30
#   python train.py --data my_data/ --lr 5e-4
#
# Outputs written to the project directory:
#   object_model.pth     - best checkpoint (model weights + class list)
#   training_curves.png  - loss and accuracy per epoch
#   confusion_matrix.png - per-class confusion heatmap on the test set

import sys
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import ObjectCNN, ObjectViT, MobileNetV2, IMG_SIZE, IMG_SIZE_MOBILE


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train the object recognition model')
    p.add_argument('--data',   default='data',
                   help='Dataset root directory (must contain one sub-folder per class)')
    p.add_argument('--model',  default='cnn', choices=['cnn', 'vit', 'mobilenet'],
                   help='Model architecture: cnn, vit, or mobilenet')
    p.add_argument('--epochs', type=int,   default=20)
    p.add_argument('--batch',  type=int,   default=32)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--output', default='object_model.pth',
                   help='Path to save the best model checkpoint')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def _eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_datasets(data_dir, img_size):
    """
    Loads images with torchvision.datasets.ImageFolder and splits into
    train (70%) / val (15%) / test (15%).

    Augmentation is applied only to the training split.
    """
    # Build the full dataset with augmented transforms for index computation
    aug_dataset  = datasets.ImageFolder(data_dir, transform=_train_transform(img_size))
    eval_dataset = datasets.ImageFolder(data_dir, transform=_eval_transform(img_size))
    classes = aug_dataset.classes

    n       = len(aug_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    # Reproducible split
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=generator)

    train_set = Subset(aug_dataset,  list(train_idx))
    val_set   = Subset(eval_dataset, list(val_idx))
    test_set  = Subset(eval_dataset, list(test_idx))

    print('Dataset split: {} train / {} val / {} test'.format(
        n_train, n_val, n_test))
    print('Classes:', classes)
    return train_set, val_set, test_set, classes


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out  = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
        correct    += out.argmax(1).eq(target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out         = model(data)
            total_loss += F.nll_loss(out, target, reduction='sum').item()
            correct    += out.argmax(1).eq(target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def compute_metrics(model, loader, num_classes, device):
    """Returns per-class precision, recall, and the full confusion matrix."""
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(1)
            for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                conf[t, p] += 1

    precision = np.zeros(num_classes)
    recall    = np.zeros(num_classes)
    for i in range(num_classes):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, conf


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, arch):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-o', label='Train')
    axes[0].plot(epochs, val_losses,   'r-o', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('{} - Loss per Epoch'.format(arch.upper()))
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accs, 'b-o', label='Train')
    axes[1].plot(epochs, val_accs,   'r-o', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('{} - Accuracy per Epoch'.format(arch.upper()))
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print('Saved training_curves.png')


def plot_confusion_matrix(conf, classes, arch):
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    im = ax.imshow(conf, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('{} Confusion Matrix (test set)'.format(arch.upper()))
    threshold = conf.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf[i, j]), ha='center', va='center',
                    color='white' if conf[i, j] > threshold else 'black',
                    fontsize=10)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print('Saved confusion_matrix.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    args = parse_args()

    if not os.path.isdir(args.data):
        print('Error: data directory "{}" not found.'.format(args.data))
        print('Run collect_data.py first to gather training images.')
        return

    torch.manual_seed(42)
    device = torch.device('mps'  if torch.backends.mps.is_available()  else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    img_size = IMG_SIZE_MOBILE if args.model == 'mobilenet' else IMG_SIZE
    train_set, val_set, test_set, classes = load_datasets(args.data, img_size)
    num_classes = len(classes)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Build model
    if args.model == 'vit':
        model = ObjectViT(num_classes).to(device)
    elif args.model == 'mobilenet':
        model = MobileNetV2(num_classes).to(device)
    else:
        model = ObjectCNN(num_classes).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print('Architecture: {}  |  Parameters: {:,}'.format(args.model.upper(), n_params))

    # MobileNet: use a lower LR for the pretrained backbone, higher for the new head
    if args.model == 'mobilenet':
        optimizer = optim.Adam([
            {'params': model.model.features.parameters(), 'lr': args.lr * 0.1},
            {'params': model.model.classifier.parameters(), 'lr': args.lr},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    print('\nTraining for {} epochs...'.format(args.epochs))
    print('{:<6} {:>10} {:>10} {:>10} {:>10}'.format(
        'Epoch', 'tr_loss', 'tr_acc%', 'vl_loss', 'vl_acc%'))
    print('-' * 50)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        flag = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'model_state': model.state_dict(),
                'classes':     classes,
                'arch':        args.model,
                'img_size':    img_size,
            }, args.output)
            flag = '  <- best'

        print('{:<6d} {:>10.4f} {:>9.1f}% {:>10.4f} {:>9.1f}%{}'.format(
            epoch, tr_loss, tr_acc, vl_loss, vl_acc, flag))

    print('\nBest validation accuracy: {:.1f}%'.format(best_val_acc))
    print('Model saved to:', args.output)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, args.model)

    # -----------------------------------------------------------------------
    # Test set evaluation
    # -----------------------------------------------------------------------
    print('\n--- Test set evaluation ---')
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    _, test_acc = eval_epoch(model, test_loader, device)
    print('Test accuracy: {:.1f}%'.format(test_acc))

    precision, recall, conf_matrix = compute_metrics(
        model, test_loader, num_classes, device)

    print('\nPer-class results:')
    print('{:<15s}  {:>10s}  {:>10s}'.format('Class', 'Precision', 'Recall'))
    print('-' * 40)
    for i, cls in enumerate(classes):
        print('{:<15s}  {:>9.1f}%  {:>9.1f}%'.format(
            cls, precision[i] * 100, recall[i] * 100))

    mean_p = precision.mean() * 100
    mean_r = recall.mean() * 100
    print('-' * 40)
    print('{:<15s}  {:>9.1f}%  {:>9.1f}%'.format('Mean', mean_p, mean_r))

    plot_confusion_matrix(conf_matrix, classes, args.model)


if __name__ == '__main__':
    main(sys.argv)
