from torch.utils.data import DataLoader
import os
import datetime
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset
from skimage.transform import warp, AffineTransform
import pickle

# dataset config and utils file
# =======================================
# TODO: 1. 将三个数据集整理成一个class
# =======================================



def random_transform(img):
    if np.random.random() < 0.5:
        img = img[:,::-1,:]

    if np.random.random() < 0.5:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-30.0*2.0*np.pi/360.0,+30.0*2.0*np.pi/360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-10,10)
        ty = np.random.uniform(-10,10)
    else:
        tx = 0.0
        ty = 0.0

    aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx,ty))
    img_aug = warp(img,aftrans.inverse,preserve_range=True).astype('uint8')

    return img_aug


def quickdraw_train_load(args, transformations=None):
    """返回测试集和训练集"""
    sketchy_train = QuickDrawDataset(split='train', root_dir=args.root_dir,version='sketch', zero_version=args.zero_version, \
                                     transform=transformations, aug=True)
    photo_train = QuickDrawDataset(split='train', root_dir=args.root_dir, version='photo', zero_version=args.zero_version, \
                                    transform=transformations, aug=True)                 
    sketchy_test = QuickDrawDataset(split='val', root_dir=args.root_dir, version='sketch', zero_version=args.zero_version, \
                                    transform=transformations, aug=False)
    photo_test = QuickDrawDataset(split='val', root_dir=args.root_dir, version='photo',zero_version=args.zero_version,\
                                    transform=transformations, aug=False)

    # class_embedding_file = f'dataset/QuickDraw/{args.zero_version}/class_embs_{args.arch_teacher}_{prompt}.pickle'
    # with open(class_embedding_file, 'rb') as fh:
    #     class_embs = pickle.load(fh)

    class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid.txt')
    test_class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid_zero.txt')
    with open(class_labels_file) as fp:
        all_class = [c.strip() for c in fp.readlines()] # 读取训练的类别标签
    with open(test_class_labels_file) as fp:
        test_all_class = [c.strip() for c in fp.readlines()] # 读取训练的类别标签

    
    return sketchy_train, photo_train, sketchy_test, photo_test, all_class, test_all_class

def quickdraw_zero_load(args, transformations=None):
    sketchy_zero = QuickDrawDataset(split='zero', root_dir=args.root_dir, version='sketch', \
                                transform=transformations, aug=False)
    photo_zero = QuickDrawDataset(split='zero', root_dir=args.root_dir, version='photo', \
                                    transform=transformations, aug=False)
    return sketchy_zero, photo_zero


class QuickDrawDataset(Dataset):
    # cid_mask: wordnet output
    def __init__(self, split='train',
                 root_dir='./dataset/QuickDraw/',
                 version='sketch', zero_version='zeroshot',\
                 cid_mask=False, transform=None, aug=False, ndebug=9999999):
        
        self.root_dir = root_dir
        self.version = version
        self.split = split
        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_val.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        # self.shuffle()
        self.file_ls = self.file_ls
        self.labels = self.labels
        self.transform = transform
        self.aug = aug
        self.cid_mask = cid_mask

    def shuffle(self):
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)
        self.file_ls = self.file_ls[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # *修改
        img = cv2.imread(os.path.join(self.root_dir, str(self.file_ls[idx])))[:,:,::-1]  # BGR-->RGB
        # if self.aug and np.random.random()<0.7:
        #     img = random_transform(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        img = self.transform(img)
        label = self.labels[idx]  # class number
        # if self.cid_mask:
        #     mask = self.cid_matrix[label]  # 返回相似度矩阵的每一行[batch, 1000]
        #     return img, label, mask
        return img, label

def tuberlin_train_load(args, cid_mask=False, transformations=None):
    """返回测试集和训练集"""
    sketchy_train = TUBerlinDataset(split='train', root_dir=args.root_dir, zero_version = args.zero_version, cid_mask=cid_mask, \
                                     transform=transformations, aug=True)
    photo_train = TUBerlinDataset(split='train', root_dir=args.root_dir, version='ImageResized_ready', zero_version = args.zero_version,\
                                  cid_mask=cid_mask, transform=transformations, aug=True)
    sketchy_test = TUBerlinDataset(split='zero', root_dir=args.root_dir, zero_version = args.zero_version, cid_mask=cid_mask, \
                                    transform=transformations, aug=False)
    photo_test =   TUBerlinDataset(split='zero', root_dir=args.root_dir, version='ImageResized_ready', zero_version = args.zero_version,\
                                    cid_mask=cid_mask, transform=transformations, aug=False)                      

    # class_embedding_file = os.path.join(args.root_dir, f'{args.zero_version}/class_embs_{args.arch_teacher}_{prompt}.pickle')
    # with open(class_embedding_file, 'rb') as fh:
    #     class_embs = pickle.load(fh)
    class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid.txt')
    test_class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid_zero.txt')
    with open(class_labels_file) as fp:
        all_class = [c.strip().replace('(', '').replace(')', '') for c in fp.readlines()] # 读取训练的类别标签
    with open(test_class_labels_file) as fp:
        test_all_class = [c.strip().replace('(', '').replace(')', '')  for c in fp.readlines()] # 读取测试的类别标签
    
    return sketchy_train, photo_train, sketchy_test, photo_test, all_class, test_all_class


def tuberlin_zero_load(args, transformations=None):
    sketchy_zero = TUBerlinDataset(split='zero', root_dir=args.root_dir, zero_version = args.zero_version, \
                                    transform=transformations, aug=False)
    image_zero = TUBerlinDataset(split='zero', root_dir=args.root_dir, version='ImageResized_ready', zero_version = args.zero_version,\
                                    transform=transformations, aug=False)
    return sketchy_zero, image_zero

class TUBerlinDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='./dataset/TUBerlin/',
                 version='png_ready', zero_version='zeroshot', \
                 cid_mask = False, transform=None, aug=False):
        
        self.root_dir = root_dir
        self.version = version
        self.split = split
        self.img_dir = self.root_dir
        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_train.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_zero.txt')
        # elif self.split == 'val':
        #     file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_val.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        self.transform = transform
        self.aug = aug
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.file_ls[idx % self.__len__()]))[:,:,::-1]
        # if self.aug and np.random.random()<0.7:
        #     img = random_transform(img)
        # if self.transform is not None:
        #     img = self.transform(img)

        img = self.transform(img)
        label = self.labels[idx % self.__len__()]  # class number
        # if self.cid_mask:
        #     mask = self.cid_matrix[label]  # 返回相似度矩阵的每一行[batch, 1000]
        #     return img, label, mask
        return img, label
        # return img

def sketchy_train_load(args, cid_mask=False, transformations=None):
    """返回测试集和训练集"""

    sketchy_train = SketchyDataset(split='train', root_dir=args.root_dir, zero_version = args.zero_version, cid_mask=cid_mask, \
                                     transform=transformations, aug=True)
    photo_train = SketchyDataset(split='train', root_dir=args.root_dir, version='all_photo', zero_version = args.zero_version, \
                                    cid_mask=cid_mask,transform=transformations, aug=False)                 
    sketchy_test = SketchyDataset(split='zero', root_dir=args.root_dir, zero_version = args.zero_version, cid_mask=cid_mask, \
                                    transform=transformations, aug=False)
    photo_test = SketchyDataset(split='zero', root_dir=args.root_dir, version='all_photo', zero_version = args.zero_version,\
                                     cid_mask=cid_mask, transform=transformations, aug=False)
    # class_embedding_file = os.path.join(args.root_dir, f'{args.zero_version}/class_embs_{args.arch_teacher}_{prompt}.pickle')
    # with open(class_embedding_file, 'rb') as fh:
    #     class_embs = pickle.load(fh)
    class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid.txt')
    test_class_labels_file = os.path.join(args.root_dir, f'{args.zero_version}/cname_cid_zero.txt')
    with open(class_labels_file) as fp:
        all_class = [c.strip() for c in fp.readlines()] # 读取训练的类别标签
    with open(test_class_labels_file) as fp:
        test_all_class = [c.strip() for c in fp.readlines()] # 读取测试的类别标签

    
    return sketchy_train, photo_train, sketchy_test, photo_test, all_class, test_all_class

def sketchy_zero_load(args, transformations=None):
    sketchy_zero = SketchyDataset(split='zero', root_dir=args.root_dir, zero_version = args.zero_version, \
                                transform=transformations, aug=False)
    photo_zero = SketchyDataset(split='zero', root_dir=args.root_dir, version='all_photo', zero_version = args.zero_version,\
                                    transform=transformations, aug=False)
    # class_embedding_file = f'./dataset/Sketchy/{args.zero_version}/class_embs_RN50_{args.prompt}_zero.pickle'
    # class_labels_file = f'./dataset/Sketchy/{args.zero_version}/cname_cid.txt'
    # with open(class_labels_file) as fp:
    #     all_class = [c.split()[0] for c in fp.readlines()] # 读取训练的类别标签
    # with open(class_embedding_file, 'rb') as fh:
    #     class_embs = pickle.load(fh)
    # return sketchy_zero, photo_zero, class_embs, all_class
    return sketchy_zero, photo_zero



class SketchyDataset(Dataset):
    # cid_mask: wordnet output
    # =================================
    def __init__(self, split='train',
                 root_dir='./dataset/Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot2',\
                 cid_mask = False, transform=None, aug=False, ndebug=999999):
        self.root_dir = root_dir
        self.version = version
        self.split = split
        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_val.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        self.file_ls = self.file_ls[:ndebug]
        self.labels = self.labels[:ndebug]
        self.transform = transform
        self.aug = aug
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # *修改
        img = cv2.imread(os.path.join(self.root_dir, self.file_ls[idx]))[:,:,::-1]  # BGR-->RGB
        # if self.aug and np.random.random()<0.7:
        #     img = random_transform(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        img = self.transform(img)
        label = self.labels[idx]  # class number
        # if self.cid_mask:
        #     mask = self.cid_matrix[label]  # 返回相似度矩阵的每一行[batch, 1000]
        #     return img, label, mask
        return img, label
        # return img



def load_dataset(args, transformations):
    if args.dataset in ['sketchy_split1', 'sketchy_split2']:
        sketchy_train, photo_train, sketchy_test, photo_test, train_class_name, test_class_name = sketchy_train_load(args, transformations=transformations)
        sketch_train_loader = DataLoader(dataset=sketchy_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        photo_train_loader = DataLoader(dataset=photo_train,batch_size=args.batch_size, shuffle=True, num_workers=4)
        sk_test_loader = DataLoader(dataset=sketchy_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        im_test_loader = DataLoader(dataset=photo_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'tuberlin':
        sketchy_train, photo_train, sketchy_test, photo_test, train_class_name, test_class_name = tuberlin_train_load(args, transformations=transformations)
        sketch_train_loader = DataLoader(dataset=sketchy_train, batch_size=args.batch_size // 5, shuffle=True, num_workers=4, pin_memory=True)
        photo_train_loader = DataLoader(dataset=photo_train,batch_size=args.batch_size // 5 * 9, shuffle=True, num_workers=4, pin_memory=True)
        sk_test_loader = DataLoader(dataset=sketchy_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        im_test_loader = DataLoader(dataset=photo_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'quickdraw':
        sketchy_train, photo_train, sketchy_test, photo_test, train_class_name, test_class_name = quickdraw_train_load(args, transformations=transformations)
        sketch_train_loader = DataLoader(dataset=sketchy_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        photo_train_loader = DataLoader(dataset=photo_train,batch_size=args.batch_size, shuffle=True, num_workers=4)
        sk_test_loader = DataLoader(dataset=sketchy_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        im_test_loader = DataLoader(dataset=photo_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    

    print(str(datetime.datetime.now()) + ' data loaded.')

    return sketch_train_loader, photo_train_loader, sk_test_loader, im_test_loader, train_class_name, test_class_name


def load_dataset_zero(args, transformations):
    if args.dataset in ['sketchy_split1', 'sketchy_split2']:
        sketchy_zero, photo_zero = sketchy_zero_load(args, transformations=transformations)
        zero_loader_ext = DataLoader(dataset=photo_zero, batch_size=args.batch_size, shuffle=False, num_workers=4)
        zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size, shuffle=False, num_workers=4)
        return zero_loader_ext, zero_loader
    
    elif args.dataset == 'tuberlin':
        sketchy_zero, photo_zero = tuberlin_zero_load(args, transformations=transformations)
        zero_loader_ext = DataLoader(dataset=photo_zero, batch_size=args.batch_size //2, shuffle=False, num_workers=4)
        zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size //2, shuffle=False, num_workers=4)
        return zero_loader_ext, zero_loader
    
    elif args.dataset == 'quickdraw':
        sketchy_zero, photo_zero = quickdraw_zero_load(args, transformations=transformations)
        zero_loader_ext = DataLoader(dataset=photo_zero, batch_size=args.batch_size, shuffle=False, num_workers=4)
        zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size, shuffle=False, num_workers=4)
        return zero_loader_ext, zero_loader

def create_dict_texts(texts):
    """创建key:number value:class_name 的字典"""
    texts = sorted(list(set(texts)))
    d = {i: l for i, l in enumerate(texts)}
    return d



def get_random_file_from_path(file_path):
    _ext = '*.jpg'
    f_list = glob.glob(os.path.join(file_path, _ext))
    return np.random.choice(f_list, 1)[0]