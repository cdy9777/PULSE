import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.build_model import build_model
from torch.utils.tensorboard import SummaryWriter
from .utils import DisabledSummaryWriter, log_exceptions

if 'tensorboard' in configs['train'] and configs['train']['tensorboard']:
    writer = SummaryWriter(log_dir='runs')
else:
    writer = DisabledSummaryWriter()


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(4) # limit the number of threads


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
        elif optim_config['name'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=optim_config['lr'],
                momentum=optim_config.get('momentum', 0.0),
                weight_decay=optim_config['weight_decay']
            )

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            model.epoch_idx = i
            
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            
            
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
            
        
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model):
        self.create_optimizer(model)
        train_config = configs['train']

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
            self.test(model)
            self.save_model(model)
            return model

        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][1]][1] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][1]][1]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break

            # re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][1]][1], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][1]][1], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplemented
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestamp))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")
        
