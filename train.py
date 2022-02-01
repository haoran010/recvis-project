import numpy as np
import torch
import model_transformer
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import Bar
from utils.viz import viz_results_paper
from utils.averagemeter import AverageMeter
from utils.utils import torch_to_list, get_num_signs
from eval import Metric

import os
import time
import pickle
import json
from math import ceil
from pathlib import Path
import datetime
from tqdm import tqdm




class Trainer:
    def __init__(self, arg, num_classes, device, weights):
        self.model = model_transformer.TransModel(arg, num_classes)

        nbparameters = filter(lambda p: p.requires_grad, self.model.parameters())
        nbparameters = sum([np.prod(p.size()) for p in nbparameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % nbparameters)

        if weights is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=-100)

        self.mse = nn.MSELoss(reduction='none')
        self.mse_red = nn.MSELoss(reduction='mean')
        self.sm = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.global_counter = 0
        self.train_result_dict = {}
        self.test_result_dict = {}

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, eval_args, pretrained=''):
        self.model.train()
        self.model.to(device)

        # load pretrained model
        if pretrained != '':
            pretrained_dict = torch.load(pretrained)
            self.model.load_state_dict(pretrained_dict)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

        for epoch in range(num_epochs):
            epoch_loss = 0
            end = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            bar = Bar("E%d" % (epoch + 1), max=batch_gen.get_max_index())
            count = 0
            get_metrics_train = Metric('train')

            while batch_gen.has_next():
                self.global_counter += 1
                batch_input, batch_target, batch_target_eval, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_target_eval, mask = batch_input.to(device), batch_target.to(
                    device), batch_target_eval.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask).permute(1, 2, 0)


                if self.num_classes == 1:
                    loss = self.mse_red(predictions.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(),
                                         batch_target.view(-1))
                else:
                    loss = self.ce(predictions.transpose(2, 1).contiguous().view(-1, 2), batch_target.view(-1))
                    loss = loss + 0.15 * torch.mean(torch.clamp(self.mse(F.log_softmax(predictions[:, :, 1:], dim=1),
                                                    F.log_softmax(predictions.detach()[:, :, :-1], dim=1)), min=0, max=8) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                if self.num_classes == 1:
                    predicted = torch.round(predictions.data.squeeze())
                    gt = torch.round(batch_target)
                    gt_eval = batch_target_eval

                else:
                    _, predicted = torch.max(predictions.data, 1)
                    gt = batch_target
                    gt_eval = batch_target_eval

                get_metrics_train.calc_scores_per_batch(predicted, gt, gt_eval, mask)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = "({batch}/{size}) Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:}".format(
                    batch=count + 1,
                    size=batch_gen.get_max_index() / batch_size,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=datetime.timedelta(seconds=ceil((bar.eta_td / batch_size).total_seconds())),
                    loss=loss.item()
                )
                count += 1
                bar.next()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            get_metrics_train.calc_metrics()
            result_dict = get_metrics_train.save_print_metrics(save_dir, epoch, epoch_loss / (
                        len(batch_gen.list_of_examples) / batch_size))
            self.train_result_dict.update(result_dict)

            eval_args[7] = epoch
            eval_args[1] = save_dir + "/epoch-" + str(epoch + 1) + ".model"
            self.predict(*eval_args)

    def predict(
            self,
            args,
            model_dir,
            results_dir,
            features_dict,
            gt_dict,
            gt_dict_dil,
            vid_list_file,
            epoch,
            device,
            mode,
            classification_threshold,
            uniform=0,
            save_pslabels=False,
            CP_dict=None,
    ):

        save_score_dict = {}
        metrics_per_signer = {}
        get_metrics_test = Metric(mode)

        self.model.eval()
        with torch.no_grad():

            if CP_dict is None:
                self.model.to(device)
                self.model.load_state_dict(torch.load(model_dir))

            epoch_loss = 0
            for vid in tqdm(vid_list_file):
                features = np.swapaxes(features_dict[vid], 0, 1)
                if CP_dict is not None:
                    predicted = torch.tensor(CP_dict[vid]).to(device)
                    pred_prob = CP_dict[vid]
                    gt = torch.tensor(gt_dict[vid]).to(device)
                    gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)
                else:
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)
                    predictions = self.model(input_x, torch.ones(input_x.size(), device=device)).permute(1, 2, 0)
                    if self.num_classes == 1:
                        # regression
                        num_iter = 1
                        pred_prob = predictions.squeeze()
                        pred_prob = torch_to_list(pred_prob)
                        predicted = torch.tensor(
                            np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)

                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    else:
                        pred_prob = torch_to_list(self.sm(predictions))[0][1]
                        predicted = torch.tensor(
                            np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)
                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    if uniform:
                        num_signs = get_num_signs(gt_dict[vid])
                        len_clip = len(gt_dict[vid])
                        predicted = [0] * len_clip
                        dist_uni = len_clip / num_signs
                        for i in range(1, num_signs):
                            predicted[round(i * dist_uni)] = 1
                            predicted[round(i * dist_uni) + 1] = 1
                        pred_prob = predicted
                        predicted = torch.tensor(predicted).to(device)

                    if save_pslabels:
                        save_score_dict[vid] = {}
                        save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                        save_score_dict[vid]['preds'] = np.asarray(torch_to_list(predicted))
                        continue

                # loss = 0
                mask = torch.ones(self.num_classes, np.shape(gt)[0]).to(device)
                # loss for each stage
                if self.num_classes == 1:
                    loss = self.mse_red(predictions.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(),
                                         gt.view(-1))
                else:
                    loss = self.ce(predictions.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(predictions[:, :, 1:], dim=1), F.log_softmax(predictions.detach()[:, :, :-1], dim=1)),
                        min=0, max=8) * mask[:, 1:])

                epoch_loss += loss.item()

                cut_endpoints = True
                if cut_endpoints:
                    if sum(predicted[-2:]) > 0 and sum(gt_eval[-4:]) == 0:
                        for j in range(len(predicted) - 1, 0, -1):
                            if predicted[j] != 0:
                                predicted[j] = 0
                            elif predicted[j] == 0 and j < len(predicted) - 2:
                                break

                    if sum(predicted[:2]) > 0 and sum(gt_eval[:4]) == 0:
                        check = 0
                        for j, item in enumerate(predicted):
                            if item != 0:
                                predicted[j] = 0
                                check = 1
                            elif item == 0 and (j > 2 or check):
                                break

                get_metrics_test.calc_scores_per_batch(predicted.unsqueeze(0), gt.unsqueeze(0), gt_eval.unsqueeze(0))

                save_score_dict[vid] = {}
                save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                save_score_dict[vid]['gt'] = torch_to_list(gt)

                if mode == 'test' and args.viz_results:
                    if not isinstance(vid, int):
                        f_name = vid.split('/')[-1].split('.')[0]
                    else:
                        f_name = str(vid)

                    viz_results_paper(
                        gt,
                        torch_to_list(predicted),
                        name=results_dir + "/" + f'{f_name}',
                        pred_prob=pred_prob,
                    )

            if save_pslabels:
                PL_labels_dict = {}
                PL_scores_dict = {}
                for vid in vid_list_file:
                    if args.test_data == 'phoenix14':
                        episode = vid.split('.')[0]
                        part = vid.split('.')[1]
                    elif args.test_data == 'bsl1k':
                        episode = vid.split('_')[0]
                        part = vid.split('_')[1]

                    if episode not in PL_labels_dict:
                        PL_labels_dict[episode] = []
                        PL_scores_dict[episode] = []

                    PL_labels_dict[episode].extend(save_score_dict[vid]['preds'])
                    PL_scores_dict[episode].extend(save_score_dict[vid]['scores'])

                for episode in PL_labels_dict.keys():
                    PL_root = str(Path(results_dir).parent).replace(f'exps/results/regression',
                                                                    'data/pseudo_labels/PL').replace(f'exps/results/classification', f'data/pseudo_labels/PL')
                    if not os.path.exists(f'{PL_root}/{episode}'):
                        os.makedirs(f'{PL_root}/{episode}')
                        pickle.dump(PL_labels_dict[episode], open(f'{PL_root}/{episode}/preds.pkl', "wb"))
                        pickle.dump(PL_scores_dict[episode], open(f'{PL_root}/{episode}/scores.pkl', "wb"))
                    else:
                        print('PL already exist!!')
                return

            if mode == 'test':
                pickle.dump(save_score_dict, open(f'{results_dir}/scores.pkl', "wb"))

            get_metrics_test.calc_metrics()
            save_dir = results_dir if mode == 'test' else Path(model_dir).parent
            result_dict = get_metrics_test.save_print_metrics(save_dir, epoch, epoch_loss / len(vid_list_file))
            self.test_result_dict.update(result_dict)

        if mode == 'test':
            with open(f'{results_dir}/eval_results.json', 'w') as fp:
                json.dump(self.test_result_dict, fp, indent=4)
