# -*- coding: utf-8 -*-

import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type
from zipfile import ZipFile
import sys
from util import batch_to_device

import numpy as np
import transformers
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # import from __init__.py the __DOWNLOAD_SERVER__ string
# from . import __DOWNLOAD_SERVER__
# from . import __version__
# # only if the class name is the same as the file name, similar to java
# # .evaluation can only be used when called in the package from outside main
# # from .evaluation import SentenceEvaluator
# # from .util import import_from_string, batch_to_device, http_get
# # __init__.py will be automatically called when calling a class within
# #  the package, with an __init__.py you can directly call a .py
# # as the __init__.py implicitly import the class
# # from evaluation import SentenceEvaluator
# from DialogEvaluator_in_presentation import DialogEvaluator
from DialogEvaluator_meld import DialogEvaluator
from DialogPredictor import DialogPredictor
# you can import function from a .py, or use util.batch_to_device to call it
# or use from util import *
# from util import import_from_string, batch_to_device, http_get
# import util.batch_to_device/class but no import util.class.method
from util import batch_to_device, import_from_string
from roberta_with_finetune import ROBERTA
from torch.utils.data import DataLoader
from input_instance import InputInstance
from csv_reader import CSVDataReader
from torchdataset_wrapper_roberta_with_finetune import TorchWrappedDataset
import sys
from transformerunit import TransformerUnit
from datetime import datetime
import math
from loggingHandler import LoggingHandler
import time


# from topics_and_emotions_meld import plot_topics_and_emotions
# from attention_and_topics import extract_attended_topic_words_and_its_topics


class DialogTransformer(nn.Sequential):
    '''
    Here, the modules will be either an LSTM or a Transformer
    '''

    def __init__(
            self,
            model_name_or_path: str = None,
            modules: Iterable[nn.Module] = None, device: str = None):
        '''
        In the very beginning, the modules shouldn't be null,
        If modules is not None, then train the model. Else, load the model and
        predict
        '''
        '''
        Here we employ the load from model initialization structure
        '''
        if modules is not None and not isinstance(modules, OrderedDict):
            # if orderedDict then use it
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)])

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained DialogTransformer: {}".format(
                model_name_or_path))

            # #### Load from server
            # if '/' not in model_name_or_path and '\\' not in model_name_or_path and not os.path.isdir(model_name_or_path):
            #     logging.info("Did not find a / or \\ in the name. Assume to download model from server")
            #     model_name_or_path = __DOWNLOAD_SERVER__ + model_name_or_path + '.zip'

            # if model_name_or_path.startswith('http://') or model_name_or_path.startswith('https://'):
            #     model_url = model_name_or_path
            #     folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250]

            #     # print('===================')

            #     try:
            #         from torch.hub import _get_torch_home
            #         torch_cache_home = _get_torch_home()

            #         # print('=================== didnt enter exception')
            #     # os.getenv(key=, default=), and the TORCH_HOME, XDG_CACHE_HOME
            #     # does not exist, so expanduser change the ~/.cache/torch
            #     os.makedirs(model_path, exist_ok=True)
            # else:
            #     model_path = model_name_or_path
            model_path = model_name_or_path

            # #### Load from disk
            if model_path is not None:
                logging.info("Load DialogTransformer from folder: {}".format(
                    model_path))

                # if os.path.exists(os.path.join(model_path, 'config.json')):
                #     with open(os.path.join(model_path, 'config.json')) as fIn:
                #         config = json.load(fIn)
                #         if config['__version__'] > __version__:
                #             logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

                with open(os.path.join(model_path, 'modules.json')) as fIn:
                    contained_modules = json.load(fIn)

                # the modules are bert, LSTM and so-on
                modules = OrderedDict()
                for module_config in contained_modules:
                    module_class = import_from_string(module_config['type'])
                    module = module_class.load(
                        os.path.join(model_path, module_config['path']))
                    modules[module_config['name']] = module

        # instantialize self._modules, therefore can conduct the basic function
        #  of the modules
        # register the modules so you can directly call it.
        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(device)
            logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        # put the modules to device
        self.to(device)
        # for feature_name in features:
        #     features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder
        into different sub-folders

        Store the total config only
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            # __name__ is the class or function name,
            # __module__ is the .py name, possibly including the relative
            #  path if executed from the outside folder.
            # logging.info("module.__name__: {}".format(
            #     type(module).__name__))
            model_path = os.path.join(
                path,
                str(idx) + "_" + type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            # modules are saved here, using the save in modules respectively
            module.save(model_path)
            # __module__ name of module in which this class was defined
            #  sometimes you will import the module from folders,
            #  for instance, you run the __main__ outside sentence_transformers
            #  folder, in which case the relative import is allowed,
            #  the __module__ is the relative path, that
            # logging.info('type(module).__module__ :{}'.format(
            #     type(module).__module__))
            # If you use __init__ and the classname is the same as the
            #  .py file. So if you use __init__, and the module when saved
            #  is imported as the Module_Folder.Classname, then
            #  when saved, however, you can use both the __module__.__name__
            #  and the __modulefolder__.__module__ to import the
            #  filename. So __init__ + save meets both. However,
            #  If you don't use __init__, then you need to save the path
            #  as __module__.__name__, and load it use __module__.__name__,
            #  so loading using __module__.__name__ meets the both
            contained_modules.append(
                {'idx': idx,
                 'name': name,
                 'path': os.path.basename(model_path),
                 'type': (
                         type(module).__module__ + '.' + type(module).__name__)})

        # the sequential configuration is saved as the modules.json in
        # the out-most folder. The contained_modules dict are saved in
        # modules.json. Whilst sequential has no modules to save.
        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        # with open(os.path.join(path, 'config.json'), 'w') as fOut:
        #     json.dump({'__version__': __version__}, fOut, indent=2)

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: DialogEvaluator,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {
                'lr': 2e-5,
                'eps': 1e-6,
                'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective

        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as
        there are in the smallest one
        to make sure of equal training with each dataset.

        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param fp16:
        :param fp16_opt_level:
        :param local_rank:
        :param train_objectives:
            Tuples of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        :param steps_per_epoch: Train for x steps in each epoch. If set to None, the length of the dataset will be used
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(output_path))

        # retrieve dataloaders
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # # Use smart batching
        # # Originally the tensorize was done in the smart batch,
        # #  We did this in the dataset_wrapper initialize
        # # Actually it converts instances to the batches
        # #  reshape the datasets and convert them to the tensors
        # #  The dataloader has default collate_fn, that is,
        # #  each batch is a list,
        # #  and [0] is feature[0], [1] is feature[1], etc., see collate_fn in
        # #  dataloader.py for detailed usages
        # for dataloader in dataloaders:
        #     dataloader.collate_fn = self.smart_batching_collate

        models = [amodel for _, amodel in train_objectives]

        # retrieve the loss_models
        # each loss_model is actually a module, for parallel training
        # on one dataset, enables parallel computation with distributed batches
        device = self.device
        for amodel in models:
            amodel.to(device)

        self.best_score = -9999999

        # num_of_batches
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = \
                min([len(dataloader) for dataloader in dataloaders])
            # the smallerest dataset determines the steps_per_epoch, that is
            # the num_of_batches per epoch

        # total number of training steps
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        # the schedulers seem to be useless
        # Oh, it's useful. It's used to change the learning rate
        # from the scheduler, we can see that _get_scheduler actually
        # wraps the optimizer, and privides a learning decay
        # for each epoch
        # >>> lambda1 = lambda epoch: epoch // 30
        # >>> lambda2 = lambda epoch: 0.95 ** epoch
        # >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        for model in models:
            # return the names and the parameters
            param_optimizer = list(model.named_parameters())
            '''
            Choose parameters to optimize
            Second way to pass parameters as groups
            '''
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            # optimizer_grouped_parameters[0] are params with weightdecay
            #  similar requires_grad=True
            # optimizer_grouped_parameters[1] are params without weightdecay
            t_total = num_train_steps
            # allow distribution, each machine execute
            #  t_total // torch.distributed.get_world_size() epochs
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            # **: from dict to params
            optimizer = optimizer_class(
                optimizer_grouped_parameters, **optimizer_params)

            # scheduler is the one which linearly/linear-linearly
            # /constantly updates the learning rate in each epoch
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            # training with half precision
            # if you are not training on small devices, please disable this
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(models)):
                '''
                loss_models includes the nn.Module successor which implements
                the Forward function
                loss_models[train_index] subscribes/points to/retrieves a model
                and accompany it with an optimizer, then they are copied to
                two parallel arrays
                '''
                model, optimizer = amp.initialize(
                    models[train_idx],
                    optimizers[train_idx],
                    opt_level=fp16_opt_level)
                models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        # steps_per_epoch * number_of_loss_models
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        # the number of models and affiliated dataloaders
        num_train_objectives = len(train_objectives)

        # criterion, LossFunction, suitable for multi model with the
        #  same training criterion. For multiple please modulize 
        #  the loss or wrap the loss outside this self module.
        # Both the model, loss and the values should be put to device
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        # iterate over all epochs
        for epoch in trange(epochs, desc="Epoch"):
            # enable the progress bar, description is "Epoch"
            # trange(0, epochs, ncols=47, ascii=True) means the bar is
            # 47-length and the bar expression is the '#'
            training_steps = 0

            # model.zero_grad() and optimizer.zero_grad()
            #  are the same IF all your model parameters are in
            #  that optimizer. I found it is safer to call model.zero_grad()
            #  to make sure all grads are zero, e.g. if you have two
            #  or more optimizers for one model.
            # call model.zero_grad() before train() so that gradients
            # can be erased safer, confirm that gradients are erased
            for model in models:
                model.zero_grad()
                model.train()

            # iterate over all batches
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                # iterate over all the models
                for train_idx in range(num_train_objectives):
                    # each model is trained per batch
                    model = models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        # each model is trained per batch
                        # fetch the next batch
                        data = next(data_iterator)
                    except StopIteration:
                        # logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        # next is the built-in function for an iterable
                        # That is, who has implemented the __iter__() interface
                        data = next(data_iterator)
                        # print('Exception:', data)
                    # print(data)
                    # # It's a procedure
                    # features = batch_to_device(data, self.device)
                    # features = data
                    features = batch_to_device(data, self.device)
                    # print(features)

                    # logging.info('if on cuda {}'.format(features[0].is_cuda))

                    # logging.info(model.is_cuda)
                    # Sequential doesn't have is_cuda, but will response to
                    #  to()

                    # both model.to() or model = model.to()
                    #  can put the model to cuda. BUT only var = var.to()
                    #  can put a variable to cuda!
                    tfboyed_features = model(features)

                    # this shouldn't be matter actually
                    # but we'd better view and stack them
                    batched_uttrs = tfboyed_features[0]
                    # tensor
                    batched_labels = tfboyed_features[1]
                    # previously list of scalars now a tensor
                    batched_lengths = tfboyed_features[2]
                    b_size, seq_size, emb_size = batched_uttrs.size()

                    lst_uttrs = []
                    lst_labels = []
                    for i_dim in range(b_size):
                        # lst_uttrs.append(
                        #     batched_uttrs[
                        #         i_dim,
                        #         :batched_lengths[i_dim],
                        #         :].view(
                        #             batched_lengths[i_dim],
                        #             emb_size))
                        # # The single index will automatically squeeze it
                        # lst_uttrs.append(
                        #     batched_uttrs[
                        #         i_dim,
                        #         :batched_lengths[i_dim],
                        #         :].squeeze())
                        # lst_labels.append(
                        #     batched_labels[
                        #         i_dim,
                        #         :batched_lengths[i_dim]].squeeze())
                        lst_uttrs.append(
                            batched_uttrs[
                            i_dim,
                            :batched_lengths[i_dim],
                            :])
                        lst_labels.append(
                            batched_labels[
                            i_dim,
                            :batched_lengths[i_dim]])
                    var_uttrs = torch.cat(lst_uttrs, dim=0)
                    var_labels = torch.cat(lst_labels, dim=0)
                    loss_value = criterion(
                        var_uttrs,
                        var_labels)

                    if fp16:
                        # if (...
                        #        and ...)
                        with amp.scale_loss(loss_value, optimizer) \
                                as scaled_loss:
                            # scale the loss_value by the amplifier
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer),
                            max_grad_norm)
                    else:
                        # perform backward for the loss_value
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm)

                    # mandatory for an optimizer
                    # optimizer optimizes the grads for the selected params
                    optimizer.step()
                    # learning rate starts with = 1e-5
                    # warmup_step decays by one each step.
                    scheduler.step()
                    # mandatory for an optimizer
                    optimizer.zero_grad()

                # training step denotes the batch_idx
                #  to the epoch here.
                # global steps = epochs * batch_num + training_steps
                training_steps += 1
                global_step += 1

                # Avoid zero every time when denumerated/divided
                # evaluate the model every evaluation_steps' batches
                if evaluation_steps > 0 and \
                        training_steps % evaluation_steps == 0:
                    # the evaluation loss is different from the model loss
                    #  and is used to save the model.
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps)
                    # evaluate after each batch
                    # evaluation during training
                    for model in models:
                        model.zero_grad()
                        model.train()
            # evaluate after each epoch
            self._eval_during_training(
                evaluator,
                output_path,
                save_best_model,
                epoch,
                -1)

    def evaluate(self, evaluator: DialogEvaluator, output_path: str = None):
        """
        Evaluate the model
        evaluate after training

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _eval_during_training(
            self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(
                self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score

    def _get_scheduler(
            self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler

        # the learning rate optimisation is wrapped in a scheduler
        # constantlr means lr is fixed.
        # Warmupconstant means lr is accerating constantly
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            # this doesn't include warmup
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            # this uses warmup
            return transformers.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            # This uses warmup + lr-decay with t_total lr-decays
            # only this wrapper accepts num_trianing_steps
            # if you open the function get_linear_schedule_with_warmup
            # you will find that return
            # LambdaLR(optimizer, lr_lambda, last_epoch)
            # and last_epoch=-1. So it will end till
            # num_training_steps consume up
            # and you will see that each .step() consumes 1 training
            # step. You can see from
            # https://pytorch.org/docs/stable/optim.html
            # 'How to adjust Learning Rate'
            # When you call the step, the counter will reduce by 1,
            # and the learning rate will be adjusted accordingly.
            # Initial learning rate is given in the Optimiser.
            # Initially, at each step, the learning rate will be
            # adjusted by param_group in optimizer.param_groups:
            #    param_group['lr'] = lr
            # Now it is implicitly adjusted by the scheduler
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))


from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import csv
import re


def text_to_single_row_csv(text, file_path):
    # 将文本按行分割
    lines = text.split('\n')

    # 将文本行拼接成一个字符串，用制表符分隔，并在行末添加与行数相同数量的0
    csv_data = '\t'.join(['"' + line.strip() + '"' for line in lines] + ['0'] * len(lines))

    # 将CSV数据写入文件
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        file.write(csv_data)


# # 测试函数
# text = """This is line 1.
# This is line 2.
# And this is line 3.
# Final line."""
# file_path = "output.csv"
# text_to_single_row_csv(text, file_path)


def read_result_file(file_path):
    numbers = []

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            for cell in row:
                # 使用正则表达式匹配数字字符并添加到列表中
                numbers.extend(re.findall(r'\d+', cell))

    return numbers


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def clear_data_folder():
    folder_path = app.config['UPLOAD_FOLDER']
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def read_csv_file(file_path, filename, max_instances=0):
    instances = []

    with open(file_path, 'r', newline='', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            column_count = len(row)
            uttrances = row[:column_count // 2]
            labels = list(map(int, row[column_count // 2:]))
            instances.append(InputInstance(
                guid=filename + str(id),
                texts=uttrances,
                labels=labels))
            if max_instances > 0 and len(instances) >= max_instances:
                break

    return instances


def process_uploaded_file(filename):
    csvDataReader = CSVDataReader('./upload/')
    instances = csvDataReader.get_instances(filename)

    tokenizer_name = '../save/topic-language-model-meld'

    tokenizer_bert = ROBERTA(tokenizer_name, max_seq_length=108, devicepad='cuda:0')

    # daily dialogue max_seq_length = 36, emory: 25
    test_data = TorchWrappedDataset(instances, tokenizer_bert, max_seq_length=33)
    # can be the same the train_batch_size
    train_batch_size = 2

    # If loading it's the saving configuration of model_dlg's bert that determines
    #  it's device
    model_save_path = ('../save/saved-model-meld')
    model_dlg = DialogTransformer(
        model_save_path,
        device='cuda:0')

    # dataloader will automatically allocate the size-1 last batch
    #  So you'd better restrict the batch_size within the epoch
    test_dataloader = DataLoader(
        test_data, shuffle=False, batch_size=train_batch_size)

    # evaluator = DialogEvaluator(test_dataloader, name='', device='cuda:0')
    # model_dlg.evaluate(evaluator, output_path='../save/saved-model-meld')

    # for prediction
    predictor = DialogPredictor(test_dataloader, name='', device='cuda:0')
    predictor(model_dlg, output_path="./" + app.config['UPLOAD_FOLDER'])

    return "./" + app.config['UPLOAD_FOLDER'] + "/all_utterance_prediction_results.csv"


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'text' in request.form:
            texts = request.form['text']
            if texts:
                text_to_single_row_csv(texts, 'upload/output.csv')
                filename = 'output.csv'
                flash('Text uploaded successfully')
                # 在提交文本后，重定向到处理文件的路由
                return redirect(url_for('process_file', filename=filename, flag=1))

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File uploaded successfully')
            return redirect(url_for('process_file', filename=filename, flag=0))

    return render_template('upload.html')


@app.route('/chatRecognition', methods=['POST'])
def chatRecognition():
    data = request.get_json()
    if 'input' in data:
        input_text = data['input']
        # 调用情感识别函数处理输入文本
        emotion_result = simulate_emotion_recognition(input_text)
        # 返回情感识别结果
        return jsonify({'emotion': emotion_result})
    else:
        # 如果没有输入文本，则返回错误信息
        return jsonify({'error': 'No input text provided'}), 400


def simulate_emotion_recognition(input_text):
    clear_data_folder()
    text_to_single_row_csv(input_text, 'upload/chatRecoginition.csv')
    filename = 'chatRecoginition.csv'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    new_file_path = process_uploaded_file(filename)
    original_content = read_csv_file(file_path, filename)
    original_texts = original_content[0].texts
    original_labels = replace_with_emotions(original_content[0].labels)
    processed_content = replace_with_emotions(read_result_file(new_file_path))

    original_texts_utf8 = []
    for text in original_texts:
        # 将 cp1252 编码的文本解码为 Unicode 字符串
        utf8_text = text.replace('\x92', "'")
        original_texts_utf8.append(utf8_text)
    return processed_content[0]


@app.route('/chat')
def chat():
    return render_template('kj.html')


def replace_with_emotions(array):
    emotions = {
        0: 'Neutral',
        1: 'Surprise',
        2: 'Fear',
        3: 'Sadness',
        4: 'Joy',
        5: 'Disgust',
        6: 'Anger'
    }
    replaced_array = [emotions[int(num)] for num in array]
    return replaced_array


@app.route('/process/<filename>')
def process_file(filename):
    flag = request.args.get('flag')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    new_file_path = process_uploaded_file(filename)
    original_content = read_csv_file(file_path, filename)
    original_texts = original_content[0].texts
    original_labels = replace_with_emotions(original_content[0].labels)
    processed_content = replace_with_emotions(read_result_file(new_file_path))

    original_texts_utf8 = []
    for text in original_texts:
        # 将 cp1252 编码的文本解码为 Unicode 字符串
        utf8_text = text.replace('\x92', "'")
        original_texts_utf8.append(utf8_text)

    return render_template('result.html', original_texts=original_texts_utf8, original_labels=original_labels,
                           processed_content=processed_content, flag=flag)


if __name__ == '__main__':
    clear_data_folder()
    app.run(debug=True)
