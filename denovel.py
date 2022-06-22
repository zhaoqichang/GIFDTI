# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import random
import os
from model import CNNFormerDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score,precision_recall_curve, auc

def show_result(lable,Accuracy_List,Precision_List,Recall_List,F1score_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.sqrt(np.var(Accuracy_List))
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    F1score_mean, F1score_std = np.mean(F1score_List), np.sqrt(np.var(F1score_List))
    AUC_mean, AUC_std = np.mean(AUC_List), np.sqrt(np.var(AUC_List))
    PRC_mean, PRC_std = np.mean(AUPR_List), np.sqrt(np.var(AUPR_List))
    print(lable)
    with open(file_results, 'a') as f:
        f.write(lable + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('F1 score(std):{:.4f}({:.4f})'.format(F1score_mean, F1score_std) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('F1 score(std):{:.4f}({:.4f})'.format(F1score_mean, F1score_std))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std))
def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, adjs, proteins, masks, labels = data
            compounds = compounds.cuda()
            adjs = adjs.cuda()
            proteins = proteins.cuda()
            masks = masks.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds,adjs, proteins, masks)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    F1_score = f1_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  # 一次epoch的平均验证loss
    return Y, P, test_loss, Accuracy, Precision, Reacll, F1_score, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = False):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_precess(model, test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f},Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1 score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset, loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test

def get_data(dataset):
    drugs = []
    proteins = []
    for pair in dataset:
        pair = pair.strip().split()
        drugs.append(pair[0])
        proteins.append(pair[1])
    drugs = list(set(drugs))
    proteins = list(set(proteins))
    return drugs, proteins

def get_kfold_data(i, datasets, k=5):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(datasets) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] # 若不能整除，将多的case放在最后一折里
        trainset = datasets[0:val_start]

    return trainset, validset

def split_data(dataset,drugs,proteins):
    train, test_drug, test_protein, test_denovel = [], [], [], []
    for i in dataset:
        pair = i.strip().split()
        if pair[0] not in drugs and pair[1] not in proteins:
            train.append(i)
        elif pair[0] not in drugs and pair[1] in proteins:
            test_drug.append(i)
        elif pair[0] in drugs and pair[1] not in proteins:
            test_protein.append(i)
        elif pair[0] in drugs and pair[1] in proteins:
            test_denovel.append(i)
    return train, test_drug, test_protein, test_denovel

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    DATASET = "DrugBank2021"

    if  DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
    else:
        weight_CE = None
    """init hyperparameters"""
    hp = hyperparameter()
    drug_dict = {'max_len': 100,
                 'encoder_dim': 256,
                 'embeding_dim': 256,
                 'embeding_num': 65,
                 'num_layers': 3,
                 'conv_kernel_size': 5,
                 'feed_forward_expansion_factor': 4,
                 'num_attention_heads': 8,
                 'attention_dropout_p': 0.1,
                 'conv_dropout_p': 0.1,
                 'predict_dropout_prob': 0.1
                 }

    protein_dict = {'max_len': 1000,
                    'encoder_dim': 256,
                    'embeding_dim': 256,
                    'embeding_num': 26,
                    'num_layers': 3,
                    'conv_kernel_size': 5,
                    'feed_forward_expansion_factor': 4,
                    'num_attention_heads': 8,
                    'attention_dropout_p': 0.1,
                    'conv_dropout_p': 0.1}
    """Load preprocessed data."""
    print("Train in " + DATASET)

    dir_input = ('./data/{}/{}.txt'.format(DATASET, DATASET))
    print("load data")
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("load finished")

    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(train_data_list, SEED)
    drugs, proteins = get_data(dataset)
    K_Fold = 5

    Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug, F1score_List_stable_drug = [], [], [], [], [], []
    Precision_List_best_drug, Recall_List_best_drug, Accuracy_List_best_drug, AUC_List_best_drug, AUPR_List_best_drug, F1score_List_best_drug = [], [], [], [], [], []
    Precision_List_stable_protein,Recall_List_stable_protein,Accuracy_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein, F1score_List_stable_protein = [], [], [], [], [], []
    Precision_List_best_protein,Recall_List_best_protein,Accuracy_List_best_protein, AUC_List_best_protein, AUPR_List_best_protein, F1score_List_best_protein = [], [], [], [], [], []
    Precision_List_stable_deno,Recall_List_stable_deno,Accuracy_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno, F1score_List_stable_deno = [], [], [], [], [], []
    Precision_List_best_deno, Recall_List_best_deno, Accuracy_List_best_deno, AUC_List_best_deno, AUPR_List_best_deno, F1score_List_best_deno = [], [], [], [], [], []

    """Output files."""

    save_path = "./ConFormerDTI/{}/denovo/".format(DATASET)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + "The_results_of_whole_dataset.txt"

    with open(file_results, 'w') as f:
        hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
        f.write(hp_attr + '\n')

    for i_fold in range(K_Fold):
        print('*' * 25, '第', i_fold + 1, '折', '*' * 25)

        _,test_drugs = get_kfold_data(i_fold, drugs)
        _, test_proteins = get_kfold_data(i_fold, proteins)
        train_dataset, test_dataset_drug, \
        test_dataset_protein, test_dataset_denovel = split_data(dataset,test_drugs,test_proteins)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset_drug = CustomDataSet(test_dataset_drug)
        test_dataset_protein = CustomDataSet(test_dataset_protein)
        test_dataset_denovel = CustomDataSet(test_dataset_denovel)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                        collate_fn=collate_fn)
        test_dataset_drug_load = DataLoader(test_dataset_drug, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                       collate_fn=collate_fn)
        test_dataset_protein_load = DataLoader(test_dataset_protein, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                       collate_fn=collate_fn)
        test_dataset_denovel_load = DataLoader(test_dataset_denovel, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                       collate_fn=collate_fn)

        """ create model"""
        model = CNNFormerDTI(drug_dict, protein_dict).cuda()
        model = nn.DataParallel(model)
        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        """load trained model"""
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate)
        # optimizer = optim.AdamW(
        #     [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        # self.optimizer_inner = RAdam(
        #     [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        # self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
        #                                         step_size_up=train_size // hp.Batch_size)
        Loss = nn.CrossEntropyLoss(weight=weight_CE)
        # print(model)
        """ 使用tensorboardX来跟踪实验"""
        if not os.path.exists(save_path+"/{}/".format(i_fold)):
            os.makedirs(save_path+"/{}/".format(i_fold))
        note = ''
        writer = SummaryWriter(log_dir=save_path+"/{}/".format(i_fold), comment=note)

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
        patience = 0
        best_score = 0
        for epoch in range(1, hp.Epoch + 1):
            trian_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for trian_i, train_data in trian_pbar:
                '''data preparation '''
                trian_nodes, train_adjs, trian_proteins, trian_proteins_mask, trian_labels = train_data
                trian_nodes = trian_nodes.cuda()
                train_adjs = train_adjs.cuda()
                trian_proteins = trian_proteins.cuda()
                trian_proteins_mask = trian_proteins_mask.cuda()
                trian_labels = trian_labels.cuda()
                '''前向传播与反向传播'''
                '''梯度置0'''
                optimizer.zero_grad()
                # 正向传播，反向传播，优化
                predicted_interaction = model(trian_nodes, train_adjs, trian_proteins, trian_proteins_mask)
                train_loss = Loss(predicted_interaction, trian_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                # scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
            # avg_train_losses.append(train_loss_a_epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_adjs, valid_proteins, valid_masks, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_adjs = valid_adjs.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_masks = valid_masks.cuda()
                    valid_labels = valid_labels.cuda()

                    valid_scores = model(valid_compounds, valid_adjs, valid_proteins, valid_masks)

                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    valid_losses_in_epoch.append(valid_loss.item())
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
            F1score_dev = f1_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_score = AUC_dev + PRC_dev
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  # 一次epoch的平均验证loss
            # avg_valid_loss.append(valid_loss)
            if valid_score > best_score:
                best_score = valid_score
                patience = 0
                torch.save(model.state_dict(), save_path+"/{}/".format(i_fold) + 'valid_best_checkpoint.pth')
            else:
                patience+=1
            epoch_len = len(str(hp.Epoch))

            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Reacll: {F1score_dev:.5f} ')

            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid AUC', AUC_dev, epoch)
            writer.add_scalar('Valid AUPR', PRC_dev, epoch)
            writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
            writer.add_scalar('Valid F1 score', F1score_dev, epoch)
            writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

            print(print_msg)
            if patience == 30:
                break

        """Test the stable model"""
        trainset_test_stable_results,_,_,_,_,_,_ = test_model(train_dataset_load, save_path, DATASET, Loss, dataset="Train", lable="stable")
        validset_test_stable_results,_,_,_,_,_,_ = test_model(valid_dataset_load, save_path, DATASET, Loss, dataset="Valid", lable="stable")
        testset_drug_test_stable_results,Accuracy_test, Precision_test,Recall_test, F1_score_test, AUC_test, PRC_test = \
            test_model(test_dataset_drug_load, save_path, DATASET, Loss, dataset="drug", lable="stable")
        AUC_List_stable_drug.append(AUC_test)
        Accuracy_List_stable_drug.append(Accuracy_test)
        AUPR_List_stable_drug.append(PRC_test)
        F1score_List_stable_drug.append(F1_score_test)
        Precision_List_stable_drug.append(Precision_test)
        Recall_List_stable_drug.append(Recall_test)

        testset_protein_test_stable_results, Accuracy_test, Precision_test,Recall_test, F1score_test, AUC_test, PRC_test = \
            test_model(test_dataset_protein_load, save_path, DATASET, Loss, dataset="protein", lable="stable")
        AUC_List_stable_protein.append(AUC_test)
        Accuracy_List_stable_protein.append(Accuracy_test)
        AUPR_List_stable_protein.append(PRC_test)
        F1score_List_stable_protein.append(F1score_test)
        Precision_List_stable_protein.append(Precision_test)
        Recall_List_stable_protein.append(Recall_test)

        testset_deno_test_stable_results, Accuracy_test, Precision_test,Recall_test,F1score_test, AUC_test, PRC_test = \
            test_model(test_dataset_denovel_load, save_path, DATASET, Loss, dataset="deno", lable="stable")
        AUC_List_stable_deno.append(AUC_test)
        Accuracy_List_stable_deno.append(Accuracy_test)
        AUPR_List_stable_deno.append(PRC_test)
        F1score_List_stable_deno.append(F1score_test)
        Precision_List_stable_deno.append(Precision_test)
        Recall_List_stable_deno.append(Recall_test)
        with open(file_results, 'a') as f:
            f.write("Test the stable model on Fold {}".format(i_fold) + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_drug_test_stable_results + '\n')
            f.write(testset_protein_test_stable_results + '\n')
            f.write(testset_deno_test_stable_results + '\n')

        """Test the best model"""
        """load trained model"""
        model.load_state_dict(torch.load(save_path+"/{}/".format(i_fold) + "/valid_best_checkpoint.pth"))

        trainset_test_results,_,_,_,_,_,_ = test_model(train_dataset_load, save_path, DATASET, Loss, dataset="Train", lable="best")
        validset_test_results,_,_,_,_,_,_ = test_model(valid_dataset_load, save_path, DATASET, Loss, dataset="Valid", lable="best")
        testset_drug_test_results,Accuracy_test, Precision_test,Recall_test,F1score_test, AUC_test, PRC_test = \
            test_model(test_dataset_drug_load,save_path, DATASET, Loss, dataset="drug", lable="best")
        Accuracy_List_best_drug.append(Accuracy_test)
        Precision_List_best_drug.append(Precision_test)
        Recall_List_best_drug.append(Recall_test)
        F1score_List_best_drug.append(F1score_test)
        AUC_List_best_drug.append(AUC_test)
        AUPR_List_best_drug.append(PRC_test)

        testset_protein_test_results, Accuracy_test, Precision_test,Recall_test,F1score_test, AUC_test, PRC_test = \
            test_model(test_dataset_protein_load, save_path, DATASET, Loss, dataset="protein", lable="best")
        Accuracy_List_best_protein.append(Accuracy_test)
        Precision_List_best_protein.append(Precision_test)
        Recall_List_best_protein.append(Recall_test)
        F1score_List_best_protein.append(F1score_test)
        AUC_List_best_protein.append(AUC_test)
        AUPR_List_best_protein.append(PRC_test)

        testset_deno_test_results, Accuracy_test, Precision_test,Recall_test,F1score_test, AUC_test, PRC_test = \
            test_model(test_dataset_denovel_load, save_path, DATASET, Loss, dataset="deno", lable="best")
        Accuracy_List_best_deno.append(Accuracy_test)
        Precision_List_best_deno.append(Precision_test)
        Recall_List_best_deno.append(Recall_test)
        F1score_List_best_deno.append(F1score_test)
        AUC_List_best_deno.append(AUC_test)
        AUPR_List_best_deno.append(PRC_test)
        with open(file_results, 'a') as f:
            f.write("Test the best model on Fold {}".format(i_fold) + '\n')
            f.write(trainset_test_results + '\n')
            f.write(validset_test_results + '\n')
            f.write(testset_drug_test_results + '\n')
            f.write(testset_protein_test_results + '\n')
            f.write(testset_deno_test_results + '\n')
        writer.close()
    print("performance on stable")
    show_result("The performance on stable model on drug de nove",
                Accuracy_List_stable_drug, Precision_List_stable_drug,Recall_List_stable_drug,
                F1score_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug)
    show_result("The performance on stable model on protein de nove",
                Accuracy_List_stable_protein, Precision_List_stable_protein,Recall_List_stable_protein,
                F1score_List_stable_protein,
                AUC_List_stable_protein, AUPR_List_stable_protein)
    show_result("The performance on stable model on de nove",
                Accuracy_List_stable_deno, Precision_List_stable_deno,Recall_List_stable_deno,
                F1score_List_stable_deno,
                AUC_List_stable_deno, AUPR_List_stable_deno)
    print("performance on stable")
    show_result("The performance on best model on drug de nove",
                Accuracy_List_best_drug, Precision_List_best_drug,Recall_List_best_drug,
                F1score_List_best_drug,
                AUC_List_best_drug, AUPR_List_best_drug)
    show_result("The performance on best model on protein de nove",
                Accuracy_List_best_protein, Precision_List_best_protein,Recall_List_best_protein,
                F1score_List_best_protein,
                AUC_List_best_protein, AUPR_List_best_protein)
    show_result("The performance on best model on de nove",
                Accuracy_List_best_deno, Precision_List_best_deno,Recall_List_best_deno,
                F1score_List_best_deno,
                AUC_List_best_deno, AUPR_List_best_deno)



