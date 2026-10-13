"""ADA-SSR training, evaluation, and feature analysis.

Requires the original project's models, losses, and utils packages.

"""
import argparse
import models
import os
import os.path as osp
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from models.image_classifier import ImageClassifier
from losses.adversarial import DomainAdversarialLoss
from models.discriminator import DomainDiscriminator
from losses.self_training import ConfidenceBasedSelfTrainingLoss
from utils.analysis import collect_feature, visualize_lle, visualize_umap, visualize_isomap, visualize_tsne
from utils.metric import accuracy, calculate_a_distance
from utils.data import ForeverDataIterator
from utils.logger import CompleteLogger
from utils.meter import AverageMeter, ProgressMeter
from utils.transforms import MultipleApply
import utils.datasets as datasets
from utils.evaluation import validate
from utils.transforms import get_train_transform, get_val_transform
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    """Load data, configure ADA-SSR, and execute the selected phase."""

    # Initialize logging and reproducibility.
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
    cudnn.benchmark = True

    # Build weak/strong views for both domains and load the dataset splits.
    val_transform = get_val_transform(args.val_resizing, resize_size=args.resize_size, norm_mean=args.norm_mean, norm_std=args.norm_std)
    weak_augment = get_train_transform(args.train_resizing, random_horizontal_flip=True, norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = get_train_transform(args.train_resizing, random_horizontal_flip=True, auto_augment='rand-m10-n2-mstd2', norm_mean=args.norm_mean, norm_std=args.norm_std)
    train_target_transform = MultipleApply([weak_augment, strong_augment])
    train_transform = train_target_transform
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = datasets.get_dataset(args.root, train_transform, val_transform, train_target_transform)
    print(f"\nDataset: {args.root}")
    print(f"Classes: {num_classes}")
    print(f"Training — source: {len(train_source_dataset):,} | target: {len(train_target_dataset):,}")
    print(f"Validation: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # Training uses equal-sized source and target batches.
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # Build the classifier and domain discriminator.
    print("=> using model '{}'".format(args.arch))
    backbone = models.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer, finetune=not args.scratch).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # Configure optimization and domain alignment.
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay))
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # Evaluation and analysis load the best saved classifier.
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    if args.phase == 'analysis':
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        visualize_tsne(source_feature, target_feature, tSNE_filename)
        print('Saving t-SNE to', tSNE_filename)
        umap_filename = osp.join(logger.visualize_directory, 'UMAP.pdf')
        visualize_umap(source_feature, target_feature, umap_filename)
        isomap_filename = osp.join(logger.visualize_directory, 'ISOMAP.pdf')
        visualize_isomap(source_feature, target_feature, isomap_filename)
        lle_filename = osp.join(logger.visualize_directory, 'LLE.pdf')
        visualize_lle(source_feature, target_feature, lle_filename)
        # Optional A-distance calculation (slow)
        # A_distance = calculate_a_distance(source_feature, target_feature, device)
        return
    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # Train and select the best checkpoint using validation accuracy.
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print('lr:', lr_scheduler.get_last_lr()[0])
        train_ada_ssr(train_source_iter, train_target_iter, classifier, domain_adv, optimizer, lr_scheduler, epoch, args)
        acc1 = validate(val_loader, classifier, args, device)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    # Evaluate the selected checkpoint on the test split.
    print('best_acc1 = {:3.1f}'.format(best_acc1))
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args, device)
    print('test_acc1 = {:3.1f}'.format(acc1))
    logger.close()


def train_ada_ssr(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    """Train one epoch with domain alignment and weak/strong self-refinement."""

    # Track losses, accuracies, and timing for this epoch.
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('TLoss', ':5.2f')
    cls_losses = AverageMeter('LabelLoss', ':5.2f')
    domain_losses = AverageMeter('DomainLoss', ':5.2f')
    self_training_losses = AverageMeter('SSLoss', ':5.2f')
    cls_accs = AverageMeter('LabelAcc', ':3.1f')
    domain_accs = AverageMeter('DomainAcc', ':3.1f')
    cls_losses_strong_s = AverageMeter('LabelLossS', ':5.2f')
    pseudo_label_ratios = AverageMeter('PsRatio', ':3.1f')
    pseudo_label_accs = AverageMeter('PsAcc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch, [batch_time, data_time, losses, cls_losses, cls_losses_strong_s, domain_losses, self_training_losses, cls_accs, domain_accs, pseudo_label_accs, pseudo_label_ratios], prefix='Epoch: [{}]'.format(epoch))

    # Read self-training settings from the command-line arguments.
    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
    model.train()
    domain_adv.train()
    end = time.time()
    print('args.threshold: ', args.threshold)
    for i in range(args.iters_per_epoch):

        # Fetch weak/strong source and target views.
        batch_s = next(train_source_iter)
        x_s = batch_s[0][0]
        x_s_strong = batch_s[0][1]
        labels_s = batch_s[1]
        batch = next(train_target_iter)
        x_t = batch[0][0]
        x_t_strong = batch[0][1]
        labels_t = batch[1]

        # Move inputs and labels to the selected device.
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        x_s_strong = x_s_strong.to(device)
        x_t_strong = x_t_strong.to(device)
        x_t = x_t.to(device)
        labels_t = labels_t.to(device)
        data_time.update(time.time() - end)

        # Classify weak views and align their domain features.
        x = torch.cat((x_s, x_t), dim=0)
        y_all, f_all, _, _ = model(x)
        y_s, y_t = y_all.chunk(2, dim=0)
        f_s, f_t = f_all.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        cls_acc = accuracy(y_s, labels_s)[0]
        loss = cls_loss + args.trade_off * transfer_loss

        # Refine strong target predictions using weak-view pseudo-labels.
        _, _, _, y_t_strong = model(x_t_strong)
        self_training_loss, mask, pseudo_labels = self_training_criterion(y_t_strong, y_t)
        self_training_loss = args.trade_off_self_training * self_training_loss

        # Supervise the strongly augmented source views.
        y_s_strong, _, _, _ = model(x_s_strong)
        cls_loss_strong_s = args.trade_off_cls_strong * F.cross_entropy(y_s_strong, labels_s)
        loss = loss + self_training_loss + cls_loss_strong_s

        # Update epoch statistics.
        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        domain_losses.update(transfer_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        self_training_losses.update(self_training_loss.item(), x_s.size(0))
        cls_losses_strong_s.update(cls_loss_strong_s.item(), x_s.size(0))

        # Target labels are used only for reporting pseudo-label accuracy.
        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels / args.batch_size
        pseudo_label_ratios.update(ratio.item() * 100, args.batch_size)
        if n_pseudo_labels > 0:
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            n_correct = (pseudo_labels == labels_t).float().sum()
            pseudo_label_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)

        # Backpropagate the combined loss and update the learning rate.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADA-SSR for Unsupervised Domain Adaptation')

    # Dataset and preprocessing.
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224, help='the image size after resizing')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')

    # Model configuration.
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=models.get_model_names(), help='backbone architecture: ' + ' | '.join(models.get_model_names()) + ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')

    # Loss weights and pseudo-label confidence.
    parser.add_argument('--trade-off', default=1.0, type=float, help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off-self-training', '--trade_off_self_training', dest='trade_off_self_training', default=1.0, type=float, help='Weight of the target self-training loss (default: 1).')
    parser.add_argument('--threshold', default=0.7, type=float, help='Confidence threshold for pseudo-label selection (default: 0.7).')
    parser.add_argument('--trade-off-cls-strong', '--trade_off_cls_strong', dest='trade_off_cls_strong', default=0.1, type=float, help='Weight of the strong-source classification loss (default: 0.1).')

    # Training and optimization.
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    # Evaluation and output.
    parser.add_argument('--per-class-eval', action='store_true', help='whether output per-class accuracy during evaluation')
    parser.add_argument('--log', type=str, default='dann', help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'], help="When phase is 'test', only test the model.When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
