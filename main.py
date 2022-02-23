"""
"""

import os
import logging
import argparse
import torch
from collections import OrderedDict
from models import CNN
from methods import BaseTrainer
from data import get_loader


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='mse', help='loss func')

# precision
parser.add_argument('--wbit', type=int, default=32, help='Weight precision')
parser.add_argument('--abit', type=int, default=32, help='Activation precision')

# dataset
parser.add_argument('--dataset', type=str, default='mnist', help='dataset: MNIST')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# learnable activation clipping
parser.add_argument('--alambda', default=1e-5, type=float, help='L2 regularization of learnable threshold')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()


def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    # dataset (MNIST)
    trainloader, testloader = get_loader(batch_size=args.batch_size, data_path=args.data_path)

    # construct model
    model = CNN(num_class=10)
    logger.info(model)

    # resume from the checkpoint
    if args.fine_tune:
        checkpoint = torch.load(args.resume)
        sdict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint...")
        
        for k, v in sdict.items():
            name = k
            new_state_dict[name] = v
        
        state_tmp = model.state_dict()
        state_tmp.update(new_state_dict)

        model.load_state_dict(state_tmp)
        logger.info("=> loaded checkpoint! acc = {}%".format(checkpoint['acc']))
    
    # initialize the trainer
    trainer = BaseTrainer(
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
    )

    # start training
    trainer.fit()

if __name__ == '__main__':
    main()