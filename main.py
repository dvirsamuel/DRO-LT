import argparse
import random
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from losses import DROLoss
from ClassAwareSampler import ClassAwareSampler
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import pickle
from sklearn.utils import class_weight
from autoaugment import CIFAR10Policy, Cutout

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='checkpoint')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--feat_loss', type=str, default=None)
parser.add_argument('--feat_sampler', type=str, default=None)
parser.add_argument('--feat_lr', type=float, default=0.01)
parser.add_argument('--cls_loss', type=str, default=None)
parser.add_argument('--cls_sampler', type=str, default=None)
parser.add_argument('--cls_lr', type=float, default=0.01)
parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--margin', type=float, default=0.1)
parser.add_argument('--margin_type', type=str, choices=['learned'], default="learned")
parser.add_argument('--balanced_clf_nepochs', default=20, type=int)

best_acc1 = 0


def freeze_layers(model, fe_bool=True, cls_bool=True):
    if fe_bool:
        model.train()
    else:
        model.eval()
    if cls_bool:
        model.linear.train()
    else:
        model.linear.eval()
    for name, params in model.named_parameters():
        if "linear" in name:
            # classifier layers
            params.requires_grad = cls_bool
        else:
            # feature extractor layers
            params.requires_grad = fe_bool


def main():
    args = parser.parse_args()
    if args.pretrained:
        folder_name = f"robust_nepochs={args.epochs}_bs={args.batch_size}_temp={args.temperature}_margin={args.margin}_type={args.margin_type}_" \
                      f"feat-{args.feat_loss}-{args.feat_sampler}-{args.feat_lr}_" \
                      f"cls-{args.cls_loss}-{args.cls_sampler}-{args.cls_lr}"
        args.store_name = os.path.join(args.pretrained, folder_name)
    else:
        args.store_name = '_'.join(
            [args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor),
             args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    # create two optimizers - one for feature extractor and one for classifier
    feat_params = []
    feat_params_names = []
    cls_params = []
    cls_params_names = []
    learnable_epsilons = torch.nn.Parameter(torch.ones(num_classes))
    for name, params in model.named_parameters():
        if params.requires_grad:
            if "linear" in name:
                cls_params_names += [name]
                cls_params += [params]
            else:
                feat_params_names += [name]
                feat_params += [params]
    print("Create Feat Optimizer")
    print(f"\tRequires Grad:{feat_params_names}")
    feat_optim = torch.optim.SGD(feat_params + [learnable_epsilons], args.feat_lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    print("Create Feat Optimizer")
    print(f"\tRequires Grad:{cls_params_names}")
    cls_optim = torch.optim.SGD(cls_params, args.cls_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume or args.evaluation:
        curr_store_name = args.store_name
        if not args.evaluation and args.pretrained:
            curr_store_name = os.path.join(curr_store_name, os.path.pardir)
        filename = '%s/%s/ckpt.best.pth.tar' % (args.root_model, curr_store_name)
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=f"cuda:{args.gpu}")
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    # Data loading code=
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),  # fill parameter needs torchvision installed from source
         transforms.RandomHorizontalFlip(),
         CIFAR10Policy(),
         transforms.ToTensor(),
         Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        original_train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                  rand_number=args.rand_number, train=True, download=True,
                                                  transform=transform_val)
        augmented_train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                   rand_number=args.rand_number, train=True, download=True,
                                                   transform=transform_train if not args.evaluation else transform_val)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        original_train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                   rand_number=args.rand_number, train=True, download=True,
                                                   transform=transform_val)
        augmented_train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                    rand_number=args.rand_number, train=True, download=True,
                                                    transform=transform_train if not args.evaluation else transform_val)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    cls_num_list = augmented_train_dataset.get_cls_num_list()
    args.cls_num_list = cls_num_list

    train_labels = np.array(augmented_train_dataset.get_targets()).astype(int)
    # calculate balanced weights
    balanced_weights = torch.tensor(class_weight.compute_class_weight('balanced',
                                                                      np.unique(train_labels),
                                                                      train_labels), dtype=torch.float).cuda(args.gpu)
    lt_weights = torch.tensor(cls_num_list).float() / max(cls_num_list)

    def create_sampler(args_str):
        if args_str is not None and "resample" in args_str:
            sampler_type, n_resample = args_str.split(",")
            return ClassAwareSampler(train_labels, num_samples_cls=int(n_resample))
        return None

    original_train_loader = torch.utils.data.DataLoader(
        original_train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # feature extractor dataloader
    feat_sampler = create_sampler(args.feat_sampler)
    feat_train_loader = torch.utils.data.DataLoader(
        augmented_train_dataset, batch_size=args.batch_size, shuffle=(feat_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=feat_sampler)

    if args.evaluation:
        # evaluate on validation set
        # calculate centroids on the train
        _, train_features, train_targets, _ = validate(original_train_loader, model, 0, args, train_labels,
                                                       flag="train", save_out=True)
        # validate
        validate(val_loader, model, 0, args, train_labels,
                 flag="val", save_out=True, base_features=train_features, base_targets=train_targets)
        quit()

    # create losses
    def create_loss_list(args_str):
        loss_ls = []
        loss_str_ls = args_str.split(",")
        for loss_str in loss_str_ls:
            c_weights = None
            prefix = ""
            if "_bal" in loss_str:
                c_weights = balanced_weights
                prefix = "Balanced "
                loss_str = loss_str.split("_bal")[0]
            if "_lt" in loss_str:
                c_weights = lt_weights
                prefix = "Longtailed "
                loss_str = loss_str.split("_")[0]
            if loss_str == "ce":
                print(f"{prefix}CE", end=",")
                loss_ls += [nn.CrossEntropyLoss(weight=c_weights).cuda(args.gpu)]
            elif loss_str == "robust_loss":
                print(f"{prefix}Robust Loss", end=",")
                loss_ls += [DROLoss(temperature=args.temperature,
                                    base_temperature=args.temperature,
                                    class_weights=c_weights,
                                    epsilons=learnable_epsilons)]
        print()
        return loss_ls

    feat_losses = create_loss_list(args.feat_loss)
    cls_losses = create_loss_list(args.cls_loss)

    # init log for training
    if not args.evaluation:
        log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
        log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
        with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
            f.write(str(args))
        tf_writer = None

    best_acc1 = 0
    best_acc_contrastive = 0
    for epoch in range(args.start_epoch, args.epochs):
        print("=============== Extract Train Centroids ===============")
        _, train_features, train_targets, _ = validate(feat_train_loader, model, epoch, args,
                                                       train_labels, log_training, tf_writer, flag="train",
                                                       verbose=True)

        if epoch < args.epochs - args.balanced_clf_nepochs:
            print("=============== Train Feature Extractor ===============")
            freeze_layers(model, fe_bool=True, cls_bool=False)
            train(feat_train_loader, model, feat_losses, epoch, feat_optim, args, train_features, train_targets)

        else:
            if epoch == args.epochs - args.balanced_clf_nepochs:
                print("================ Loading Best Feature Extractor =================")
                # load best model
                curr_store_name = args.store_name
                filename = '%s/%s/ckpt.best.pth.tar' % (args.root_model, curr_store_name)
                checkpoint = torch.load(filename, map_location=f"cuda:{args.gpu}")['state_dict']
                model.load_state_dict(checkpoint)

            print("=============== Train Classifier ===============")
            freeze_layers(model, fe_bool=False, cls_bool=True)
            train(feat_train_loader, model, cls_losses, epoch, cls_optim, args)

        print("=============== Extract Train Centroids ===============")
        _, train_features, train_targets, _ = validate(original_train_loader, model, epoch, args,
                                                       train_labels, log_training, tf_writer, flag="train",
                                                       verbose=False)

        print("=============== Validate ===============")
        acc1, _, _, contrastive_acc = validate(val_loader, model, epoch, args,
                                               train_labels, log_testing, tf_writer, flag="val",
                                               base_features=train_features,
                                               base_targets=train_targets)
        if epoch < args.epochs - args.balanced_clf_nepochs:
            is_best = contrastive_acc > best_acc_contrastive
            best_acc_contrastive = max(contrastive_acc, best_acc_contrastive)
        else:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        print(f"Best Contrastive Acc: {best_acc_contrastive}, Best Cls Acc: {best_acc1}")
        log_testing.write(f"Best Contrastive Acc: {best_acc_contrastive}, Best Cls Acc: {best_acc1}")
        log_testing.flush()
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1
        }, is_best)


def train(trainval_loader, model, losses, epoch, optimizer, args, base_features=None, base_targets=None):
    avg_meters = [(type(l).__name__, AverageMeter('Loss', ':.4e')) for l in losses]
    for i, (input, target) in enumerate(trainval_loader):
        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        feats, output = model(input)
        loss = 0
        for idx, loss_func in enumerate(losses):
            if type(loss_func) == nn.CrossEntropyLoss:
                l = loss_func(output, target)
            elif type(loss_func) == DROLoss:
                l = loss_func(feats, target, base_features, base_targets)
            else:
                print("Loss not found")
                quit()
            avg_meters[idx][1].update(l.item(), input.size(0))
            loss += l

        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            output = f'Epoch: [{epoch}][{i}/{len(trainval_loader)}], lr: {optimizer.param_groups[-1]["lr"]:.8f}\t'
            for t, avg_meter in avg_meters:
                output += f'{t} {avg_meter.val:.4f} ({avg_meter.avg:.4f})\t'
            print(output)
            if torch.isnan(loss):
                quit()


def validate(val_loader, model, epoch, args,
             train_labels=None, log=None, tf_writer=None, flag='val', save_out=False,
             base_features=None, base_targets=None, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    all_feats = []

    feat_dim = 64
    features = torch.empty((0, feat_dim)).cuda()
    targets = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            feats, output = model(input)
            features = torch.cat((features, feats))
            targets = torch.cat((targets, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
            all_feats.extend(feats.detach().cpu().numpy())

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

        # calculate many-shot, median shot and few-shot accuracy
        if train_labels is not None:
            many, med, few, pc = \
                shot_acc(np.array(all_preds), np.array(all_targets), train_labels, acc_per_cls=True)
            if verbose:
                print(f"Cls Many Shot: {many}, Cls Med Shot: {med}, Cls Few Shot: {few}, CLS Acc: {np.mean(pc)}")

        if save_out:
            fname = '%s/%s/outputs_%s.pkl' % (args.root_model, args.store_name, flag)
            print('===> Saving outputs to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                    'feats': np.array(all_feats),
                    'labels': np.array(all_targets),
                    'preds': np.array(all_preds),
                }, f, protocol=4)

        contrastive_acc = None
        if base_features is not None and base_targets is not None:
            many, med, few, (pc, centroid_contrastive_acc) = \
                centroids_balanceness_feature_extractor(base_features,
                                                        base_targets,
                                                        features, targets)
            if verbose:
                # Nearest centroid neightbor
                print(
                    f"\tNCN Many Shot: {many}, NCN Med Shot: {med}, NCN Few Shot: {few}, NCN Acc: {centroid_contrastive_acc}")
            output += "\nCent. Per Class Contrastive Acc:" + str((pc, centroid_contrastive_acc)) + "\n"
            contrastive_acc = centroid_contrastive_acc

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
            tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

        return top1.avg, features, targets, contrastive_acc


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
