"""
DNN base trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import accuracy, AverageMeter, print_table, lr_schedule, convert_secs2time, save_checkpoint, catgorize_param

class BaseTrainer(object):
    def __init__(self,
        model: nn.Module,
        loss_type: str, 
        trainloader, 
        validloader,
        args,
        logger,
    ):

        self.args = args

        # loader
        self.trainloader = trainloader
        self.validloader = validloader
        
        # loss func
        if loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type")
        
        # optimizer and lr scheduler
        if self.args.learnth:
            vth, params = catgorize_param(model)
            self.optimizer = torch.optim.Adam([
                {'params': vth, 'lr': args.lr * 0.05},
                {'params': params, 'lr': args.lr}
            ])
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[lr_schedule, lr_schedule])
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[lr_schedule])

        # model architecture
        self.model = model
        if args.use_cuda:
            self.model = model.cuda()
            if args.ngpu > 1:
                self.model = nn.DataParallel(model)
                print("Data parallel!")
        
        if self.args.learnth:
            self.vthre = AverageMeter()
        

        # logger
        self.logger = logger
        self.logger_dict = {}

        # wandb logger
        if args.wandb:
            self.wandb_logger = wandb.init(entity=args.entity, project=args.project, name=args.name, config={"lr":args.lr})
            self.wandb_logger.watch(model, criterion=self.criterion, log_freq=1)
            self.wandb_logger.config.update(args)


    def base_forward(self, inputs, target):
        """Foward pass of NN
        """
        out = self.model(inputs)
        loss = self.criterion(out, target)
        return out, loss

    def base_backward(self, loss):
        # zero grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_step(self, inputs, target):
        """Training step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        out, loss = self.base_forward(inputs, target)
        
        if self.args.learnth:
            reg_alpha = torch.tensor(0.).cuda()
            cnt = 0
            for name, param in self.model.named_parameters():
                if 'vth' in name:
                    if cnt > 0:
                        self.vthre.update(param.item())
                        reg_alpha += param.norm(p=2)
                cnt += 1
            loss += self.args.alambda * (reg_alpha)

        self.base_backward(loss)
        
        return out, loss

    def valid_step(self, inputs, target):
        """validation step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        out, loss = self.base_forward(inputs, target)
            
        return out, loss

    def train_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()

        for idx, (inputs, target) in enumerate(self.trainloader):
            if self.args.use_cuda:
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
            
            out, loss = self.train_step(inputs, target)
            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            if (idx+1) % 50 == 0:
                print("Train: [{}]/[{}], loss = {:.2f}; top1={:.2f}".format(idx+1, len(self.trainloader), loss.item(), prec1.item()))
        
        self.logger_dict["train_loss"] = losses.avg
        self.logger_dict["train_top1"] = top1.avg
        self.logger_dict["train_top5"] = top5.avg
        

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(self.validloader):
                if self.args.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)

                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        
        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg
        self.logger_dict["avg_vth"] = self.vthre.avg

    def fit(self):
        self.logger.info("\nStart training: lr={}, loss={}".format(self.args.lr, self.args.loss_type))

        start_time = time.time()
        epoch_time = AverageMeter()
        best_acc = 0.
        for epoch in range(self.args.epochs):
            self.logger_dict["ep"] = epoch+1
            self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
            
            # training and validation
            self.train_epoch()
            self.valid_epoch()
            self.lr_scheduler.step()

            is_best = self.logger_dict["valid_top1"] > best_acc
            if is_best:
                best_acc = self.logger_dict["valid_top1"]

            state = {
                'state_dict': self.model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
            }

            filename='checkpoint.pth.tar'
            save_checkpoint(state, is_best, self.args.save_path, filename=filename)

            # online log
            if self.args.wandb:
                self.wandb_logger.log(self.logger_dict)
            
            # terminal log
            columns = list(self.logger_dict.keys())
            values = list(self.logger_dict.values())
            print_table(values, columns, epoch, self.logger)

            # record time
            e_time = time.time() - start_time
            epoch_time.update(e_time)
            start_time = time.time()

            need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (self.args.epochs - epoch))
            print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs))

    