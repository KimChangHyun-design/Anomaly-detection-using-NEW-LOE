# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from utils import compute_pre_recall_f1
from utils import compute_p_c
from utils import compute_p_f
from utils import compute_th

class NeutralAD_trainer:

    def __init__(self, model, loss_function, config):

        self.loss_fun = loss_function
        #print(loss_function)
        #DCL
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)
        #print(model)
        #SeqNeutralAD
        self.train_method = config['train_method']
        self.max_epochs = config['training_epochs']
        self.warmup = 2

    def _train(self, epoch,train_loader, optimizer, verbose = False):

        self.model.train()
        loss_all = 0

        for data in train_loader:
            samples = data['sample']
            #print(samples.shape)
            labels = data['label']
            true_labels = data['true_label']
            #print(true_labels.shape)
            # samples = samples.to(self.device)

            z = self.model(samples)
            loss_n,loss_a = self.loss_fun(z)

            if epoch <=self.warmup:
                if self.train_method == 'gt':
                    loss = torch.cat([loss_n[labels==0],loss_a[labels==1]],0)
                    loss_mean = loss.mean()
                else:
                    loss = loss_n
                    loss_mean= loss.mean()
            else:
                score = loss_n-loss_a

                if self.train_method=='blind':
                    loss = loss_n
                    loss_mean = loss.mean()
                elif self.train_method=='loe_hard':
                    _, idx_n = torch.topk(score, int(score.shape[0] * (1-self.contamination)), largest=False,
                                                         sorted=False)
                    _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True,
                                                         sorted=False)
                    loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], 0)
                    loss_mean = loss.mean()
                elif self.train_method == 'loe_soft':
                    _, idx_n = torch.topk(score, int(score.shape[0] * (1-self.contamination)), largest=False, sorted=False)
                    _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True, sorted=False)
                    if verbose:
                         print(idx_n)
#                         print(idx_a)
#                         print(loss_n.shape)
#                         print(loss_a.shape)
#                         print(idx_n.shape)
#                         print(idx_a.shape)
#                         print('a')
#                         print(loss_n[idx_a].shape)
#                         print('b')
#                         print(loss_a[idx_a].shape)
#                         print(loss_a[labels==1])
                    loss = torch.cat([loss_n[idx_n],0.5*loss_n[idx_a]+0.5*loss_a[idx_a]],0)
                    loss_mean= loss.mean()
                    
                elif self.train_method == 'refine':
                    _, idx_n = torch.topk(loss_n, int(loss_n.shape[0] * (1-self.contamination)), largest=False,
                                                         sorted=False)
                    loss = loss_n[idx_n]
                    loss_mean = loss.mean()
                elif self.train_method == 'gt':
                    loss = torch.cat([loss_n[labels==0],loss_a[labels==1]],0)
                    loss_mean = loss.mean()

                    loss_mean = loss.mean()
                elif self.train_method == 'loe_soft_semi':
                    _, idx_n = torch.topk(score, round(score.shape[0] * (1-self.contamination)), largest=False, sorted=False)
                    _, idx_a = torch.topk(score, round(score.shape[0] * self.contamination), largest=True, sorted=False)
                    idx_a = torch.tensor([i for i in idx_a if i not in idx_n]) # remove duplicates
                    
                    # loss = torch.cat([loss_n[idx_n],0.5*loss_n[idx_a]+0.5*loss_a[idx_a],loss_a[true_labels==1]],0)
                    loss = torch.cat([loss_n[idx_n],0.5*loss_n[idx_a]+0.5*loss_a[idx_a]],0)
                      #146/4=36
#                     print("Truelables:",true_labels[true_labels==0].shape) #35
#                     print("loss:",loss.shape) # 36
                    # print(samples.shape, len(idx_n), len(idx_a), score.shape)
                    # print("loss:",loss.shape) # 36
                    # print("true:",true_labels.shape) # 36
                    
                    loss = loss[true_labels==0]
                        
                    loss_mean= loss.mean() + loss_a[true_labels==1].mean()
                    
                
                elif self.train_method == 'loe_hard_semi':
                    , idx_n = torch.topk(score, round(score.shape[0] * (1-self.contamination)), largest=False, sorted=False)
                    _, idx_a = torch.topk(score, round(score.shape[0] * self.contamination), largest=True, sorted=False)
                    idx_a = torch.tensor([i for i in idx_a if i not in idx_n]) # remove duplicates
                    
#                     _, idx_n = torch.topk(score, int(score.shape[0] * (1-self.contamination)), largest=False, sorted=False)
#                     _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True, sorted=False)
#                     # loss = torch.cat([loss_n[idx_n], loss_a[idx_a],loss_a[true_labels==1]], 0)
                    loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], 0)
                    
                    
                    loss = loss[true_labels==0]
                    loss_mean= loss.mean() + loss_a[true_labels==1].mean()
                    
                    
   
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()


        return loss_all.item()/len(train_loader.dataset)


    def detect_outliers(self, loader):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                labels = data['label']
                # samples = samples.to(self.device)
                z= model(samples)
                loss_n,loss_a = self.loss_fun(z)
                score = loss_n
                loss_in += loss_n[labels == 0].sum()
                loss_out += loss_n[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score)

        score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        ap = average_precision_score(target_all, score_all)
        p_c = compute_p_c(target_all, score_all)
        p_f = compute_p_f(target_all, score_all)
        th = compute_th(target_all, score_all)
        
        return auc, ap , f1, p_c, p_f, th, score_all,  loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()

    
        def detect_outliers(self, loader):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                labels = data['label']
                # samples = samples.to(self.device)
                z= model(samples)
                loss_n,loss_a = self.loss_fun(z)
                score = loss_n
                loss_in += loss_n[labels == 0].sum()
                loss_out += loss_n[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score)

        score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        ap = average_precision_score(target_all, score_all)
        p_c = compute_p_c(target_all, score_all)
        p_f = compute_p_f(target_all, score_all)
        th = compute_th(target_all, score_all)
        
        return auc, ap , f1, p_c, p_f, th, score_all,  loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()



#     def detect_outliers(self, loader):
#         model = self.model
#         model.eval()

#         loss_in = 0
#         loss_out = 0
#         kld = 0
#         target_all = []
#         score_all = []
#         for data in loader:
#             with torch.no_grad():
#                 samples = data['sample']
#                 labels = data['label']
#                 # samples = samples.to(self.device)
#                 z= model(samples)
#                 loss_n,loss_a = self.loss_fun(z)
#                 score = loss_n

#                 ###
#                 _, idx_n = torch.topk(score, int(score.shape[0] * (1-self.contamination)), largest=False, sorted=False)
#                 _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True, sorted=False)

#                 sample_n = samples[idx_n.to(samples.device)]
#                 sample_a = samples[idx_a.to(samples.device)]
                
#                 s_n = np.random.choice(len(sample_n),100)
#                 s_a = np.random.choice(len(sample_a),100)
                
#                 kld += torch.nn.KLDivLoss(reduction='batchmean')(sample_n[s_n], sample_a[s_a])
#                 #start_pred = torch.nn.functional.log_softmax(sample_n[s_n],dim=1)
#                 #end_pred = torch.nn.functional.log_softmax(sample_a[s_a],dim=1)
#                 #kld += torch.nn.KLDivLoss(reduction='batchmean')(start_pred, end_pred)
                
                
#                 #kld += torch.mean(sample_a[s_a] * (torch.log(sample_a[s_a]) - sample_n[s_n]))
#                 ##kld += torch.cdist(sample_n[s_n], sample_a[s_a])
#                 ##print(kld.shape)
               
                
                
                
#                 ###
#                 loss_in += loss_n[labels == 0].sum()
#                 loss_out += loss_n[labels == 1].sum()
#                 target_all.append(labels)
#                 score_all.append(score)

#         score_all = torch.cat(score_all).cpu().numpy()
#         target_all = np.concatenate(target_all)
#         auc = roc_auc_score(target_all, score_all)
#         f1 = compute_pre_recall_f1(target_all,score_all)
#         ap = average_precision_score(target_all, score_all)
#         p_c = compute_p_c(target_all, score_all)
#         p_f = compute_p_f(target_all, score_all)
#         th = compute_th(target_all, score_all)
        
#         return auc, ap , f1, p_c, p_f, th, kld.item() / target_all.shape[0],  \
#                loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()

    
    def train(self, train_loader, contamination, query_num=0,optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=2):

        self.contamination = contamination
        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        val_p_c, val_p_f = None, None, 
        val_th = None,
        test_auc, test_f1, test_p_c, test_p_f, test_th, test_score = None, None, None,None,None,None,


        for epoch in range(1, self.max_epochs+1):

            train_loss = self._train(epoch,train_loader, optimizer)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                test_auc, test_ap, test_f1, test_p_c, test_p_f, test_th, test_score, testin_loss, testout_loss = self.detect_outliers(test_loader)

            if validation_loader is not None:
                val_auc, val_ap,val_f1,  val_p_c, val_p_f, val_th, _, valin_loss,valout_loss = self.detect_outliers(validation_loader)
                if epoch>self.warmup:
                    if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc, test_ap,test_f1,
                                                                     
                                                                        test_score,
                                                                        train_loss):
                        break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss}, VAL loss: {valin_loss,valout_loss}, VL auc: {val_auc} VL ap: {val_ap} VL f1: {val_f1} '

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        if early_stopper is not None:
            train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1, test_score, best_epoch \
                = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
                f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1}'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        return val_loss, val_auc, test_auc, test_ap, test_f1, test_p_c, test_p_f, test_th, test_score