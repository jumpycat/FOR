from utils import get_data_iters, prepare_sub_folder, write_loss, get_config, write_2images
import argparse
from trainer import HiSD_Trainer
import torch
import os
import sys
import tensorboardX
import shutil
import random
import datetime
from face_models.models.resnet import resnet50
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba-hq.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--gpus", nargs='+')
opts = parser.parse_args()

from torch.backends import cudnn

# For fast training
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
total_iterations = config['total_iterations']

exp_name = datetime.datetime.now().strftime("%y%m%d%H%M%S")
opts.output_path = os.path.join('runs', exp_name)
os.makedirs(opts.output_path, exist_ok=True)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Setup model
multi_gpus = len(opts.gpus) > 1
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opts.gpus)
trainer = HiSD_Trainer(opts.output_path,config, multi_gpus=multi_gpus)
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

if multi_gpus:
    trainer.cuda(int(opts.gpus[0]))
    print("Using GPUs: %s" % str(opts.gpus))
    trainer.models= torch.nn.DataParallel(trainer.models, device_ids=[int(gpu) for gpu in opts.gpus])
else:
    trainer.cuda(int(opts.gpus[0]))

# Setup data loader
train_iters = get_data_iters(config, opts.gpus)
tags = list(range(len(train_iters)))

import time
start = time.time()

#####################################################################
def get_obama_loader(): 
    from torchvision import transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils import data

    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder('/root/autodl-tmp/data/00-DYS/pubfig83/multi_train_test_83/multi_train_test/train', transform)

    obama_data_loader = data.DataLoader(dataset=dataset, batch_size=8,shuffle=True,drop_last=True,num_workers=8)
    return obama_data_loader

obama_data_loader = get_obama_loader()
obama_data_iter = iter(obama_data_loader)


face_rec_model = resnet50().cuda()

with open('/root/autodl-tmp/daiyunshu/backdoor/HiSD/core/face_models/resnet50_ft_weight.pkl', 'rb') as f:
    weights = pickle.load(f, encoding='latin1')
    
own_state = face_rec_model.state_dict()
for name, param in weights.items():
    if name in own_state:
        try:
            own_state[name].copy_(torch.from_numpy(param))
        except Exception:
            raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                            'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
    else:
        raise KeyError('unexpected key "{}" in state_dict'.format(name))
face_rec_model.eval()


while True:
    """
    i: tag
    j: source attribute, j_trg: target attribute
    x: image, y: tag-irrelevant conditions
    """
    i = random.sample(tags, 1)[0]
    j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 
    x, y = train_iters[i][j].next()
    train_iters[i][j].preload()

    ############# get obma  #############

    try:
        obama_data,_ = next(obama_data_iter)
        obama_data = obama_data.cuda(int(opts.gpus[0]))
    except:
        obama_data_iter = iter(obama_data_loader)
        obama_data,_ = next(obama_data_iter)
        obama_data = obama_data.cuda(int(opts.gpus[0]))


    G_adv, G_sty, G_rec, D_adv = trainer.update(x, y, i, j, j_trg, obama_data,face_rec_model)

    if (iterations + 1) % config['image_save_iter'] == 0:
        for i in range(len(train_iters)):
            j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 

            x, _ = train_iters[i][j].next()
            x_trg, _ = train_iters[i][j_trg].next()
            train_iters[i][j].preload()
            train_iters[i][j_trg].preload()

            test_image_outputs = trainer.sample(x, x_trg, j, j_trg, i)
            write_2images(test_image_outputs,
                          config['batch_size'], 
                          image_directory, 'sample_%08d_%s_%s_to_%s' % (iterations + 1, config['tags'][i]['name'], config['tags'][i]['attributes'][j]['name'], config['tags'][i]['attributes'][j_trg]['name']))
    
    torch.cuda.synchronize()

    if (iterations + 1) % config['log_iter'] == 0:
        write_loss(iterations, trainer, train_writer)
        now = time.time()
        print(f"[#{iterations + 1:06d}|{total_iterations:d}] {now - start:5.2f}s")
        start = now

    if (iterations + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, iterations)

    if (iterations + 1) == total_iterations:
        print('Finish training!')
        exit(0)

    iterations += 1


