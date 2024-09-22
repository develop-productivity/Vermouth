import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise NotImplementedError

def get_parser():
    parser = argparse.ArgumentParser(description='diffusion sbir training and testing')
    parser.add_argument('--dataset', default='sketchy_split2', help='dataset name')                  
    parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                        help='zeroshot version for training and testing (default: zeroshot3)')  # zeroshot2：(train : test -- 104： 21)
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',help='number of samples per batch')
    parser.add_argument('--cuda', type=int, default=0, help='cuda or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_dir', type=str, default='experiments/SBIR/sd1-5/', help='log directory')

    # options for train only
    parser.add_argument('--num_epoches', default=20, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature-coefficient')
    parser.add_argument('--early_stop', default=10, type=int, help='number of early stop epochs')
    parser.add_argument('--commit', default='fuse', type=str, help='commit to save')
    parser.add_argument('--content', default='main', type=str, help='experiments name')
    
    # options for test only
    parser.add_argument('--top', type=str, default='all',  help='report map@xxx ,default map@all ')

    # diffusion only
    parser.add_argument('--version', type=str, default='1-5', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=256, choices=(256, 512, 224))
    parser.add_argument('--max_attn_size', type=int, default=16, choices=(16, 32, 64))
    parser.add_argument('--dtype', type=str, default='float32', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--model_path', type=str, default='../stable-diffusion-v1-5', help='Number of workers to split the dataset across')
    parser.add_argument('--upsample_mode', type=str, default='bilinear', help='Timesteps to compute features')
    parser.add_argument('--inv_method', type=str, default='ddpm_schedule', choices=['ddim_inv', 'ddpm_schedule', 'fuse'], help='classifier')
    parser.add_argument('--do_fuse', type=str2bool, default='yes', help='')
    parser.add_argument('--fuse_arch', type=str, default='tiny', choices=['pico', 'tiny', 'small', 'add'], help='classifier')
    parser.add_argument('--frozen_expert', type=str2bool, default='true', help='')
    parser.add_argument('--frozen_backbobe', type=str2bool, default='yes', help='')
    parser.add_argument('--fix_noise', type=str2bool, default='yes', help='')
    parser.add_argument('--second_last_layer', type=str2bool, default='no', help='')
    parser.add_argument('--empty_prompt', type=str2bool, default='no', help='')
    parser.add_argument('--random_prompt', type=str2bool, default='no', help='')
    parser.add_argument('--expert', type=str, default='dinov2', choices=['resnet', 'dinov1', 'dinov2', 'no_expert'], help='')
    
    parser.add_argument('--time_step', type=int, default=[0], nargs='+', help='which time steps to use')
    parser.add_argument('--place_in_unet', type=str, default=['up', 'mid'], nargs='+', help='which time steps to use')
    parser.add_argument('--attn_selector', type=str, default='up_cross', help='which time steps to use')
    parser.add_argument('--cross_attn', type=str2bool, default='yes', help='which time steps to use')

    # other backbone
    parser.add_argument('--model_type', type=str, default='diffusion', help="archtecture of baseline net:['clip', 'mae', 'dino', 'swin', 'swinv2', 'diffusion', 'beitv3]")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()