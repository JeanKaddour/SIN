import logging
from argparse import Namespace

import wandb
from torch.nn import Module
from torch_geometric.data import DataLoader

from experiments.early_stopping import EarlyStoppingCriterion
from experiments.evaluate import test_evaluation, valid_evaluation
from experiments.io import load_test_dataset
from experiments.utils import get_model, get_train_and_val_dataset


def train(model: Module, train_dataset_pt: list, val_dataset_pt: list, device, args: Namespace):
    train_loader = DataLoader(dataset=train_dataset_pt, batch_size=min(args.batch_size, len(train_dataset_pt)),
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset_pt, batch_size=min(args.batch_size, len(val_dataset_pt)))
    early_stopping = EarlyStoppingCriterion(patience=args.patience, mode='min')
    if args.model == 'gin':
        train_com_model(model=model, device=device, train_loader=train_loader, val_loader=val_loader, args=args)
    for epoch in range(1, args.max_epochs + 1):
        model.train_step(device=device, train_loader=train_loader, epoch=epoch, log_interval=args.log_interval)
        if epoch % args.val_interval == 0:
            validation_error = valid_evaluation(model=model, device=device, val_loader=val_loader, epoch=epoch,
                                                val_loss_name='val_loss')
            if not early_stopping.step(validation_error, epoch):
                break


def train_and_test(args: Namespace, device):
    model = get_model(args=args, device=device)
    train_dataset_pt, val_dataset_pt = get_train_and_val_dataset(args=args)
    train_loader = DataLoader(dataset=train_dataset_pt, batch_size=min(args.batch_size, len(train_dataset_pt)),
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset_pt, batch_size=min(args.batch_size, len(val_dataset_pt)))
    test_dataset = load_test_dataset(args=args)
    early_stopping = EarlyStoppingCriterion(patience=args.patience, mode='min')
    if args.model == 'gin':
        train_com_model(model=model, device=device, train_loader=train_loader, val_loader=val_loader, args=args)
    if args.model != 'zero':
        for epoch in range(1, args.max_epochs + 1):
            model.train_step(device=device, train_loader=train_loader, epoch=epoch, log_interval=args.log_interval)
            if epoch % args.val_interval == 0:
                validation_error = valid_evaluation(model=model, device=device, val_loader=val_loader, epoch=epoch,
                                                    val_loss_name='val_loss')
                if not early_stopping.step(validation_error, epoch):
                    break
    test_errors = test_evaluation(model=model, device=device,
                                  test_dataset=test_dataset, args=args)
    return test_dataset, test_errors


def train_com_model(model: Module, device, train_loader: DataLoader, val_loader: DataLoader, args: Namespace):
    com_early_stopping = EarlyStoppingCriterion(patience=args.como_patience, mode='min')

    for epoch in range(1, args.max_epochs_como_training + 1):
        for batch_idx, batch in enumerate(train_loader):
            covariates, target_outcome = batch.covariates.to(device), batch.y.to(device)
            model.como_net_opt.zero_grad()
            com_prediction = model.como_net.forward_unit(covariates)
            loss = model.com_loss(prediction=com_prediction, target=target_outcome)
            loss.backward()
            model.como_net_opt.step()
            if batch_idx % args.log_interval == 0:
                loss = loss.item()
                logging.info('COMO Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))
                wandb.log({"epoch": epoch, "train_como_loss": loss})
        if epoch % args.val_interval == 0:
            validation_error = valid_evaluation(model=model.como_net, device=device, val_loader=val_loader, epoch=epoch,
                                                val_loss_name='como_val_loss')
            model.train()
            if not com_early_stopping.step(validation_error, epoch):
                break
