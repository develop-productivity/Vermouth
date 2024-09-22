from scipy.spatial.distance import cdist
import os
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from torch.cuda.amp import autocast
import pickle


# evaluate function and  utils file
# =======================================
# TODO:
# =======================================

def get_features(data_loader, model, captioner, test_prompts, args):
    features_all = []
    targets_all = []
    model.eval()
    for i, (input, target) in enumerate(data_loader):
        if i%10==0:
            print(i, end=' ', flush=True)
        # compute output
        # if args.empty_prompt:
        #     prompt = [" "] * input.size()[0]
        # else:
        #     prompt = [prompts[idx] for idx in target]
        #     if args.random_prompt:
        #         prompt = random.sample(prompt, len(prompt))
        input =input.to(args.device)
        # target = target.to(args.device)
        if args.empty_prompt:
            prompt = [" "] * input.size()[0]
        elif args.random_prompt:
            prompt = [test_prompts[idx] for idx in target]
            prompt = random.sample(prompt, len(prompt))
        else:
            prompt=captioner(input)
        with autocast():
            features = model.extract_features(input, prompt)
        # features = model.extract_features(input, prompt)
        # 测试时的特征要规范化
        features = F.normalize(features, p=2, dim=1)
        features = features.cpu().detach().numpy()
        features_all.append(features.reshape(input.size()[0],-1))
        targets_all.append(target.detach().numpy())
    print('')
        
    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)
    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))
    return features_all, targets_all




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断是不是相等
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_map(im_loader, sk_loader, epoch, model, captioner, test_prompts, args):
    """每个epoch后计算map
    不经过ITQ算法
    """
    features_gallery, gt_labels_gallery = get_features(im_loader, model, captioner, test_prompts, args)
    features_query, gt_labels_query = get_features(sk_loader, model, captioner, test_prompts, args)
    scores = - cdist(features_query, features_gallery)  # [N, M] 负距离矩阵(每个sample的距离)

    mAP_ = 0.0
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(features_query.shape[0]):  # N个 
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=args.top)  # gt_labels_query：(12994, )
        mAP_ls[gt_labels_query[fi]].append(mapi)
        
    for _, mAPs in enumerate(mAP_ls):
        mAP_ += np.nanmean(mAPs)
    map_valid = mAP_ / (len(mAP_ls))
    print('Epoch: [{}/{}] \t validate map: {:.4f}'.format(epoch + 1, args.num_epoches, map_valid))
    return map_valid


def test_pipe(saved_file, args):
    with open(saved_file, 'rb') as fh:
        features_gallery, features_query, gt_labels_gallery, gt_labels_query, scores = pickle.load(fh)
    mAP_, prec_ = 0.0, 0.0
    length_aps, aps = [], []
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(gt_labels_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, args.top)
        mAP_ls[gt_labels_query[fi]].append(mapi)

    print('--------------------map&precision---------------------')
    # for i, (mAPs, precs, mAPs_bin, precs_bin) in enumerate(zip(mAP_ls, prec_ls, mAP_ls_bin, prec_ls_bin)):
    for i, (mAPs, precs) in enumerate(zip(mAP_ls, prec_ls)):
        # print('{}    \t|map@all: {:.4f} | precision@100: {:.4f} | map_bin@all: {:.4f} | precision_bin@100: {:.4f}'.format(
        #     class_dict[str(i)], np.nanmean(mAP_ls[i]), np.nanmean(prec_ls[i]), 
        #        np.nanmean(mAP_ls_bin[i]),np.nanmean(prec_ls_bin[i])))
        
        aps.append(np.nanmean(mAPs))
        mAP_ += np.nanmean(mAPs)
        # prec_ += np.nanmean(precs)
        # mAP_bin_ += np.nanmean(mAPs_bin)
        # prec_bin_ += np.nanmean(precs_bin)
        length_aps.append(len(mAPs))
    aps = np.array(aps)
    fls_sk, fls_im = get_fls(args)
    save_qualitative_results(aps, length_aps, fls_sk, fls_im, scores, n_q=20, n_sim=15, is_best=True, args=args)


def save_qualitative_results(aps, length_aps, fls_sk, fls_im, sim, n_q=6, n_sim=10, is_best=True, args=None):
    """
    args:
        aps: ap array
        n_q: n qualitative sample
        n_sim: n simialr sample
    """
    save_root_dir = 'experiments/SBIR/qualitative_results'
    dataset_root_dir = '/data/sydong/datasets/SBIR'
    if args.dataset in ['sketchy_split1', 'sketchy_split2']:
        dataset_root_dir = os.path.join(dataset_root_dir, 'Sketchy')
    elif args.dataset == 'tuberlin':
        dataset_root_dir = os.path.join(dataset_root_dir, 'TUBerlin')
    elif args.dataset == 'quickdraw':
        dataset_root_dir = os.path.join(dataset_root_dir, 'QuickDraw')
    save_image = True
    save_dir = os.path.join(save_root_dir, args.dataset)
    if is_best:
        ind_sk = np.argsort(aps)[:n_q]
    else:
        np.random.seed(1)
        ind_sk = np.random.choice(len(aps), n_q, replace=False)
    os.makedirs(save_dir, exist_ok=True)
    fp = open(os.path.join(save_dir, f"Results_{args.file_name}.txt"), "w")

    for i, isk in enumerate(ind_sk):
        isk = int(np.sum(length_aps[:isk]) + 1) 
        fp.write("{0}, ".format(os.path.join(dataset_root_dir, fls_sk[isk])))
        if save_image:
            sdir_op = os.path.join(save_dir, f"{args.file_name}_" + str(i + 1))
            if not os.path.isdir(sdir_op):
                os.makedirs(sdir_op)
            sk = Image.open(os.path.join(dataset_root_dir, fls_sk[isk])).convert(mode='RGB')
            sk.save(os.path.join(sdir_op, fls_sk[isk].split('/')[0] + '.png'))
        ind_im = np.argsort(-sim[isk])[:n_sim]  # 
        for j, iim in enumerate(ind_im):
            if j < len(ind_im)-1:
                fp.write("{0} {1}, ".format(fls_im[iim], sim[isk][iim]))
            else:
                fp.write("{0} {1}".format(fls_im[iim], sim[isk][iim]))
            if save_image:
                im = Image.open(os.path.join(dataset_root_dir, fls_im[iim])).convert(mode='RGB')
                im.save(os.path.join(sdir_op, str(j + 1) + '_' + str(sim[isk][iim]) + '.png'))
        fp.write("\n")
    fp.close()




def ITQ(V, iters=150):
    """
    Main function for  ITQ which finds a rotation of the PCA embedded data
    Input:
        V: nxc PCA embedded data, n is the number of images and c is the code length
        n_iter: max number of iterations, 50 is usually enough
    Output:
        B: nxc binary matrix
        R: the ccc rotation matrix found by ITQ
    Publications:
        Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
        Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    Initialize with a orthogonal randomion in rotatitialize with a orthogonal random rotation
    """
    bit = V.shape[1]
    np.random.seed(0)
    R = np.random.randn(bit, bit)
    U11, _, _ = np.linalg.svd(R)  # SVD
    R = U11[:, :bit]  # rotation matrix
    #  ITQ to find optimal rotation
    for _ in range(iters):
        Z = np.matmul(V, R)  # 对feature进行一次旋转
        UX = np.ones(Z.shape) * -1  # element wise product
        UX[Z >= 0] = 1   # 优化B
        C = np.matmul(UX.T, V)  # B和feature求内积
        UB, _, UA = np.linalg.svd(C)
        R = np.matmul(UA, UB.T)  # 优化R
    B = UX
    B[B < 0] = 0
    return B, R


def compressITQ(Xtrain, Xtest, n_iter=50):
    """
    compressITQ runs ITQ
    Center the data, VERY IMPORTANT
    args:
        Xtrain:
        Xtest:
        n_iter:
    """
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)
    # PCA
    C = np.cov(Xtrain, rowvar=False)  # covariance
    l, pc = np.linalg.eigh(C, 'U')  # 返回特征值特征向量
    idx = l.argsort()[::-1]  # 返回特征值从大到小的位置索引
    pc = pc[:, idx]  # 取非0特征向量
    XXtrain = np.matmul(Xtrain, pc)  # PCA
    XXtest = np.matmul(Xtest, pc)  # PCA
    # ITQ
    _, R = ITQ(XXtrain, n_iter)
    Ctrain = np.matmul(XXtrain, R)  # rotation
    Ctest = np.matmul(XXtest, R)  # rotation
    # bool
    # Ctrain = Ctrain > 0
    # Ctest = Ctest > 0

    # bool to 0, 1
    indxs = Ctrain > 0
    Ctrain[indxs] = 1
    Ctrain[~indxs] = 0

    indxs = Ctest> 0
    Ctest[indxs] = 1
    Ctest[~indxs] = 0
    return Ctrain, Ctest

def eval_AP_inner(inst_id, scores, gt_labels, top='all'):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)  # bool 加法 True--1, False--0 total positive
    
    sort_idx = np.argsort(-scores)  # return high -- low indices of scores
    tp = pos_flag[sort_idx]  # bool 
    fp = np.logical_not(tp)  # bool
    
    if top != 'all':
        top = int(top)
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)
    
    fp = np.cumsum(fp)  # truth positive cumsum 模拟积分过程
    tp = np.cumsum(tp)  # false positive cumsum
    try:
        rec = tp / tot_pos  # recall  truth positive / total positive
        prec = tp / (tp + fp)  # precision
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)  # 计算单个query的ap
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)
    
    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)
    
    for ii in range(len(mpre)-2,-1,-1):
        mpre[ii] = max(mpre[ii], mpre[ii+1])
        
    msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top='all'):
    # 测试zs3时把args.top = '200'应该为计算precision@100
    if top != 'all':
        top = int(top)
    else:
        top = 100
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]  # total
    top = min(top, tot)
    
    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top])/top



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_fls(args):
    dataset_root_dir = '/data/sydong/datasets/SBIR'

    if args.dataset == 'sketchy_split1':
        fls_sk_file = dataset_root_dir + '/Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_zero.txt'
        fls_im_file = dataset_root_dir + '/Sketchy/zeroshot1/all_photo_filelist_zero.txt'

    elif args.dataset == 'sketchy_split2':
        fls_sk_file = dataset_root_dir + '/Sketchy/zeroshot2/sketch_tx_000000000000_ready_filelist_zero.txt'
        fls_im_file = dataset_root_dir + '/Sketchy/zeroshot2/all_photo_filelist_zero.txt'

    elif args.dataset == 'tuberlin':
        fls_sk_file = dataset_root_dir + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        fls_im_file = dataset_root_dir + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'quickdraw':
        fls_sk_file =  dataset_root_dir + '/QuickDraw/zeroshot/sketch_filelist_zero.txt'
        fls_im_file = dataset_root_dir + '/QuickDraw/zeroshot/photo_filelist_zero.txt'
    with open(fls_sk_file, 'r') as fh:
        sk_file_content = fh.readlines()
    with open(fls_im_file, 'r') as fh:
        im_file_content = fh.readlines()
    
    fls_sk = np.array([' '.join(ff.strip().split()[:-1]) for ff in sk_file_content])
    fls_im = np.array([' '.join(ff.strip().split()[:-1]) for ff in im_file_content])
    return fls_sk, fls_im

