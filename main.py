# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/7/
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
import timeit,pickle
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score,precision_recall_curve, auc

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def show_result(DATASET1, DATASET2, Loss_List, Accuracy_List,Precision_List,Recall_List,F1_score_List,AUC_List,AUPR_List):
    Loss_mean, Loss_std = np.mean(Loss_List), np.sqrt(np.var(Loss_List))
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.sqrt(np.var(Accuracy_List))
    Precision_mean, Precision_var = np.mean(Precision_List), np.sqrt(np.var(Precision_List))
    Recall_mean, Recall_var = np.mean(Recall_List), np.sqrt(np.var(Recall_List))
    F1_score_mean, F1_score_var = np.mean(F1_score_List), np.sqrt(np.var(F1_score_List))
    AUC_mean, AUC_std = np.mean(AUC_List), np.sqrt(np.var(AUC_List))
    PRC_mean, PRC_std = np.mean(AUPR_List), np.sqrt(np.var(AUPR_List))
    print("The results on {} of {}:".format(DATASET1,DATASET2))
    with open(resultsavepath + 'results.txt', 'a') as f:
        f.write('{}:'.format(DATASET1) + '\n')
        f.write('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std) + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')
    print('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std))
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var))
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

def test_model(dataset_load,save_path,DATASET, LOSS,save = False):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_prediction.txt".format(DATASET), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = 'Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1 score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test)
    print(results)
    return results,loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test


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

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "DrugBank2021"
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
    if "DrugBank" in DATASET:
        weight_CE = None
    elif DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
    else:
        weight_CE = None
    print("Train in " + DATASET)
    dir_input = ('./data/{}/{}.txt'.format(DATASET, DATASET))
    print("load data")
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("load finished")
    resultsavepath = "./ConFormerDTI/{}/".format(DATASET)
    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(train_data_list, SEED)

    """load protein embed matrix"""
    # protein_emb_dict = np.load('./../../../data/{}/word2vec_ngram3_dim256.npy'.format(DATASET))
    # zeros = np.zeros((1, 256))
    # protein_emb_dict = np.concatenate((zeros, protein_emb_dict), axis=0)
    # protein_word_dict = load_pickle('./../../../data/{}/{}_word_dict.pickle'.format(DATASET,DATASET))
    # protein_dict['input_dim_target'] = len(protein_word_dict)+1

    K_Fold = 5
    Loss_List_train, Accuracy_List_train, Precision_List_train, Recall_List_train, F1_List_train, AUC_List_train, AUPR_List_train = [], [], [], [], [], [], []
    Loss_List_valid, Accuracy_List_valid, Precision_List_valid, Recall_List_valid, F1_List_valid, AUC_List_valid, AUPR_List_valid = [], [], [], [], [], [], []
    Loss_List_test, Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test = [], [], [], [], [], [], []
    for i_fold in range(K_Fold):
        print('*' * 25, '第', i_fold + 1, '折', '*' * 25)
        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset = CustomDataSet(test_dataset)
        # collate_fn = collate_class(dict=protein_word_dict)
        TVdataset_len = len(TVdataset)
        valid_size = int(hp.validation_split * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
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
        # model.pro_embedding.weight.data.copy_(torch.from_numpy(protein_emb_dict))
        # model.pro_embedding.weight.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate)
        # optimizer = optim.AdamW(
        #     [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        # scheduler1 = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
        #                                         step_size_up=train_size // hp.Batch_size)
        # scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        Loss = nn.CrossEntropyLoss(weight=weight_CE)
        # print(model)
        """ 使用tensorboardX来跟踪实验"""
        save_path = resultsavepath + "{}".format(i_fold)
        note = ''
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
        patience = 0
        best_score = 0
        epoch_len = len(str(hp.Epoch))
        for epoch in range(hp.Epoch):
            trian_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            # scheduler2.step()
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
                # scheduler1.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            writer.add_scalar('Train Loss/{}'.format(i_fold), train_loss_a_epoch, epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
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

                    valid_scores = model(valid_compounds,valid_adjs,  valid_proteins,valid_masks)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    valid_losses_in_epoch.append(valid_loss.item())
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  # 一次epoch的平均验证loss
            valid_score = AUC_dev + PRC_dev
            writer.add_scalar('Valid Loss/{}'.format(i_fold), valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid score/{}'.format(i_fold), valid_score, epoch)

            # test_pbar = tqdm(
            #     enumerate(
            #         BackgroundGenerator(test_dataset_load)),
            #     total=len(test_dataset_load))
            # _, _, test_loss, _, _, test_AUC, test_PRC = test_precess(model, test_pbar, Loss)
            # writer.add_scalar('test Loss/{}'.format(i_fold), test_loss, epoch)
            # writer.add_scalar('test auc/{}'.format(i_fold), test_AUC, epoch)
            # writer.add_scalar('test aupr/{}'.format(i_fold), test_PRC, epoch)

            if valid_score > best_score:
                best_score = valid_score
                patience = 0
                torch.save(model.state_dict(), save_path + '/valid_best_checkpoint.pth')
            else:
                patience+=1
            print_msg = (f'[{epoch+1:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'patience: {patience} ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         # f'test_loss: {test_loss:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} '
                         # + f'test_AUC: {test_AUC:.5f} ' +
                         # f'test_PRC: {test_PRC:.5f} '
                         )
            print(print_msg)

            if patience == 20:
                break

        """Test the best model"""
        """load trained model"""
        model.load_state_dict(torch.load(save_path + "/valid_best_checkpoint.pth"))

        trainset_test_results, Loss_train, Accuracy_train, Precision_train, Recall_train, F1_score_train, AUC_train, PRC_train = \
            test_model(train_dataset_load, save_path, DATASET, Loss)
        Loss_List_train.append(Loss_train)
        Accuracy_List_train.append(Accuracy_train)
        Precision_List_train.append(Precision_train)
        Recall_List_train.append(Recall_train)
        F1_List_train.append(F1_score_train)
        AUC_List_train.append(AUC_train)
        AUPR_List_train.append(PRC_train)
        with open(resultsavepath + 'results.txt', 'a') as f:
            f.write("The result of train set  on {} fold:".format(i_fold) + trainset_test_results + '\n')

        validset_test_results, Loss_valid, Accuracy_valid, Precision_valid, Recall_valid, F1_score_valid, AUC_valid, PRC_valid = \
            test_model(valid_dataset_load, save_path, DATASET, Loss)
        Loss_List_valid.append(Loss_valid)
        Accuracy_List_valid.append(Accuracy_valid)
        Precision_List_valid.append(Precision_valid)
        Recall_List_valid.append(Recall_valid)
        F1_List_valid.append(F1_score_valid)
        AUC_List_valid.append(AUC_valid)
        AUPR_List_valid.append(PRC_valid)
        with open(resultsavepath + 'results.txt', 'a') as f:
            f.write("The result of valid set  on {} fold:".format(i_fold) + validset_test_results + '\n')

        testset_test_results, Loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
            test_model(test_dataset_load, save_path, DATASET, Loss)
        Loss_List_test.append(Loss_test)
        Accuracy_List_test.append(Accuracy_test)
        Precision_List_test.append(Precision_test)
        Recall_List_test.append(Recall_test)
        F1_List_test.append(F1_score_test)
        AUC_List_test.append(AUC_test)
        AUPR_List_test.append(PRC_test)
        with open(resultsavepath + 'results.txt', 'a') as f:
            f.write(testset_test_results + '\n')
        writer.close()
    show_result("Trainset", DATASET, Loss_List_train,
                Accuracy_List_train, Precision_List_train, Recall_List_train, F1_List_train, AUC_List_train,
                AUPR_List_train)
    show_result("Validset", DATASET, Loss_List_valid,
                Accuracy_List_valid, Precision_List_valid, Recall_List_valid, F1_List_valid, AUC_List_valid,
                AUPR_List_valid)
    show_result("Testset", DATASET, Loss_List_test,
                Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test)




