import os

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, EAVDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb


seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(seed=seed):
    """设置随机种子以确保实验的可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    """为每个worker设置随机种子"""
    np.random.seed(int(seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0, dataset='IEMOCAP'):
    """获取训练和验证的采样器"""
    size = len(trainset) # 获取训练集的大小
    idx = list(range(size)) # 创建索引列表
    split = int(valid*size) # 计算验证集的大小
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])  # 返回训练和验证的采样器


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    """加载IEMOCAP数据集"""
    trainset = IEMOCAPDataset() # 初始化训练集
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid) # 获取采样器

    # 创建数据加载器
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)  # 初始化测试集
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader  # 返回训练、验证和测试数据加载器

def get_EAV_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, sub=1):
    trainset = EAVDataset(train=True, subject = sub)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = EAVDataset(train=False, subject = sub)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """训练或评估模型"""
    losses, preds, labels, masks = [], [], [], []  # 初始化损失、预测和标签
    alphas, alphas_f, alphas_b, vids = [], [], [], []  # 用于保存注意力权重和视频ID
    max_sequence_len = [] # 保存最大序列长度

    assert not train or optimizer!=None   # 如果是训练模式，确保优化器不为空
    if train:
        model.train()   # 设置模型为训练模式
    else:
        model.eval()   # 设置模型为评估模式

    seed_everything() # 设置随机种子
    for data in dataloader:  # 遍历数据加载器中的每一个批次
        if train:
            optimizer.zero_grad()  # 清空优化器的梯度

        # 将数据移到GPU（如果可用）
        # 这里data（一批次）的一个d： 文本特征，视觉特征，语音特征，qmask，umask，情感标签，视频id（被排除掉）
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0)) # 记录当前批次的最大序列长度

        # 前向传播
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # 处理log_prob
        labels_ = label.view(-1) # 将标签展平
        loss = loss_function(lp_, labels_, umask) # 计算损失

        pred_ = torch.argmax(lp_,1)  # 获取预测结果
        preds.append(pred_.data.cpu().numpy()) # 将预测结果移到CPU并保存
        labels.append(labels_.data.cpu().numpy()) # 将标签移到CPU并保存
        masks.append(umask.view(-1).cpu().numpy()) # 将掩码移到CPU并保存

        losses.append(loss.item()*masks[-1].sum()) # 保存当前批次的损失
        if train:
            loss.backward() # 反向传播
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)  # 记录参数的梯度
            optimizer.step()  # 更新参数
        else:
            alphas += alpha  # 保存注意力权重
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]  # 保存视频ID

    if preds!=[]:
        preds  = np.concatenate(preds) # 合并所有预测结果
        labels = np.concatenate(labels) # 合并所有标签
        masks  = np.concatenate(masks) # 合并所有掩码
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[] # 如果没有预测结果，返回NaN

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)   # 计算平均损失
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)  # 计算平均准确率
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)   # 计算F1分数
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]  # 返回结果


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):
    """训练或评估图模型"""
    losses, preds, labels = [], [], []  # 初始化损失、预测和标签
    scores, vids = [], [] # 用于保存分数和视频ID

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []  # 初始化图相关变量

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()  # 将变量移到GPU

    assert not train or optimizer!=None  # 如果是训练模式，确保优化器不为空
    if train:
        model.train() # 设置模型为训练模式
    else:
        model.eval() # 设置模型为评估模式

    seed_everything()  # 设置随机种子
    for data in dataloader:
        if train:
            optimizer.zero_grad() # 清空优化器的梯度

        # 将数据移到GPU（如果可用）
        # 这里data（一批次）的一个d： 4个文本特征，视觉特征，语音特征，qmask，umask，情感标签，视频id（被排除掉）
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]




        #在m3net中，multi_modal=true，mm_fusion_mthd=concat-DHT
        if args.multi_modal:  # 如果启用了多模态
            if args.mm_fusion_mthd=='concat':  # 如果融合方法是连接
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)  # 连接音频、视觉和文本特征
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)    # 连接音频和视觉特征
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)   # 连接视觉和文本特征
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)  # 连接音频和文本特征
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                textf = textf   # 如果使用门控机制，直接使用textf
        else:  # 如果没有启用多模态
            if modals == 'a':  # 仅使用音频特征
                textf = acouf
            elif modals == 'v': # 仅使用视觉特征
                textf = visuf
            elif modals == 'l': # 仅使用文本特征
                textf = textf
            else:
                raise NotImplementedError

        # 计算每个序列的有效长度
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        # 调整形状
        textf1 = textf1.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 1024]
        textf2 = textf2.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 1024]
        textf3 = textf3.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 1024]
        textf4 = textf4.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 1024]
        qmask = qmask.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 2]
        # umask = umask.unsqueeze(0)  # 保持 [16, 1]
        acouf = acouf.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 1582]
        visuf = visuf.permute(1, 0).unsqueeze(0)  # 变为 [1, 16, 342]

        # # 打印结果
        # print("textf1 shape:", textf1.shape)
        # print("textf2 shape:", textf2.shape)
        # print("textf3 shape:", textf3.shape)
        # print("textf4 shape:", textf4.shape)
        # print("qmask shape:", qmask.shape)
        # print("umask shape:", umask.shape)
        # print("acouf shape:", acouf.shape)
        # print("visuf shape:", visuf.shape)
        # print("labels shape:", label.shape)
        # print("lengths :", lengths)


        # 前向传播
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob, e_i, e_n, e_t, e_l = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob, e_i, e_n, e_t, e_l = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        # 这里应该是我自己的
        # 应该改对应model的输入！！
        elif args.multi_modal and args.mm_fusion_mthd=='concat_EAV':
            log_prob, e_i, e_n, e_t, e_l = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])  # 处理标签
        # print('打印label：', label)
        label = label.long()
        # print('打印label：', label)

        loss = loss_function(log_prob, label)  # 计算损失
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())  # 获取预测结果
        labels.append(label.cpu().numpy()) # 保存标签
        losses.append(loss.item()) # 保存损失
        if train:
            loss.backward()  # 反向传播
            optimizer.step()    # 更新参数
            

    if preds!=[]:
        preds  = np.concatenate(preds)  # 合并所有预测结果
        labels = np.concatenate(labels)  # 合并所有标签
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [] # 如果没有预测结果，返回NaN

    vids += data[-1]   # 保存视频ID
    # 转换为NumPy数组
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)  # 计算平均损失
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2) # 计算平均准确率
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)  # 计算F1分数

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el  # 返回结果


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')  # 是否使用GPU

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')  # 基础模型

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')  # 是否使用图模型

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper') # 是否使用节点注意力

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')  # 上下文窗口大小

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances') # 上下文窗口大小

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight') # L2正则化权重
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')  # 递归层dropout比率
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')   # 是否使用类别权重
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')  # 是否使用主动监听
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')  # 注意力类型
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')   # 图类型

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')  # 是否使用主题信息

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')  # alpha参数

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')  # 多头数量

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')  # 图构造方式

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')  # 是否结合光谱和非光谱方法

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')  # 是否使用残差信息

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information') # 是否使用多模态信息

    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently') # 多模态融合方法

    parser.add_argument('--modals', default='avl', help='modals to fusion')  # 使用的模态

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality') # 是否在音频和视觉模态中使用LSTM

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')  # 深度GCN的层数

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test') # 使用的数据集

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding') # 是否使用说话者嵌入

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')  # 是否使用模态嵌入

    parser.add_argument('--norm', default='LN2', help='NORM type')   # 归一化类型

    parser.add_argument('--testing', action='store_true', default=False, help='testing')  # 是否为测试模式

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')  # 超卷积层数量

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')  # 卷积层数量

    parser.add_argument('--subject', type=int, default=1, help='subject id')  # 卷积层数量

    args = parser.parse_args() # 解析命令行参数
    today = datetime.datetime.now()   # 获取当前时间
    subject = args.subject
    print(args) # 打印参数

    # 根据命令行参数构建模型名称
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    ##  use_speaker = True
    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda   # 检查是否使用CUDA
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter  # 导入TensorBoard记录器
        writer = SummaryWriter()

    cuda       = args.cuda  # CUDA标志
    n_epochs   = args.epochs  # 训练轮数
    batch_size = args.batch_size  # 批次大小
    modals = args.modals  # 模态设置
    # 特征维度映射
    # IEMOCAP  文本：1024  音频： 1582   视觉： 342
    # EAV      eeg：1024  音频： 1582   视觉： 342
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = 1582 if args.Dataset=='EAV' else feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = 1024
    #D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:   # 如果使用多模态
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024 # 如果没有使用连接方法
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError

    # 图模型相关参数
    D_g = 512  # if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 5 if args.Dataset=='EAV' else 1

    if args.graph_model:   # 如果使用图模型
        seed_everything()

        # 初始化模型
        model = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)

        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
    if args.Dataset == 'EAV':
        loss_weights = torch.FloatTensor([1/0.2,
                                        1/0.2,
                                        1/0.2,
                                        1/0.2,
                                        1/0.2])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight: # 如果使用类别权重
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights) # 使用负对数似然损失
            else:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)  # 初始化优化器

    lr = args.lr  # 学习率


    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2)
    elif args.Dataset == 'IEMOCAP':

        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)
    elif args.Dataset == 'EAV':
        print(f'-------------subject{subject:02d}------------')
        train_loader, valid_loader, test_loader = get_EAV_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2, sub=subject)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None  # 初始化最佳指标
    all_fscore, all_acc, all_loss = [], [], []  # 保存所有指标

    if args.testing: # 如果是测试模式
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
        print('test_acc:',test_acc,'test_fscore:',test_fscore)

    # 训练过程

    for e in range(n_epochs):
        start_time = time.time()  # 记录开始时间

        if args.graph_model: # 如果使用图模型
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore) # 保存所有F1分数


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:  # 更新最佳损失
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore == None or best_fscore < test_fscore: # 更新最佳F1分数
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            #test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        # 输出训练过程中的各项指标
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        # if (e+1)%10 == 0:  # 每10个epoch输出一次最佳F1分数和分类报告
        #     print ('----------best F-Score:', max(all_fscore))
        #     print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        #     print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

        
    

    if args.tensorboard:
        writer.close()

    if not args.testing: # 如果不是测试模式
        print('Test performance..')
        print ('F-Score:', max(all_fscore)) # 输出最佳F1分数
        if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump({}, f)  # 初始化记录文件
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)  # 加载记录
        key_ = name_ # 记录键
        if record.get(key_, False):
            record[key_].append(max(all_fscore)) # 更新记录
        else:
            record[key_] = [max(all_fscore)] # 创建新的记录
        if record.get(key_+'record', False):
            record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))  # 更新分类报告记录
        else:
            record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]  # 创建新的分类报告记录
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)) # 输出最佳分类报告
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask)) # 输出混淆矩阵
