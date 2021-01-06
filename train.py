"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

#from models.fewshot import FewShotSeg
from models.vgg import Encoder
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex
import torch.nn.functional as F


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = Encoder(pretrained_path=_config['path']['init_path'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        label_sets = _config['label_sets'],
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'mcl_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        #support image,support mask label and support multi-class label
        support_images = [[shot.cuda() for shot in way]
                        for way in sample_batched['support_images']]
        support_images = torch.cat([torch.cat(way, dim=0) for way in support_images], dim=0)
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                        for way in sample_batched['support_mask']]
        support_fg_mask = torch.cat([torch.cat(way, dim=0) for way in support_fg_mask], dim=0)
        support_label_mc = [[shot[f'label_ori'].long().cuda() for shot in way]
                        for way in sample_batched['support_mask']]
        support_label_mc = torch.cat([torch.cat(way, dim=0) for way in support_label_mc], dim=0)

        #query image,query mask label and query multi-class label
        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
        query_label_mc = [n_queries[f'label_ori'].long().cuda() 
                        for n_queries in sample_batched['query_masks']]
        query_label_mc = torch.cat(query_label_mc, dim=0)
        
        optimizer.zero_grad()
        query_pred, support_pred_mc, query_pred_mc,support_pred = model(support_images, query_images, support_fg_mask)
        query_pred = F.interpolate(query_pred, size= query_images.shape[-2:], mode='bilinear')
        support_pred = F.interpolate(support_pred, size= support_images.shape[-2:], mode='bilinear')
        support_pred_mc = F.interpolate(support_pred_mc, size= support_images.shape[-2:], mode='bilinear')
        query_pred_mc = F.interpolate(query_pred_mc, size= query_images.shape[-2:], mode='bilinear')

        binary_loss = criterion(query_pred, query_labels) + criterion(support_pred, support_fg_mask.long().cuda())
        mcl_loss = criterion(support_pred_mc, support_label_mc) + criterion(query_pred_mc, query_label_mc)

        loss = binary_loss + mcl_loss * _config['mcl_loss_scaler']
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Log loss
        binary_loss = binary_loss.detach().data.cpu().numpy()
        mcl_loss = mcl_loss.detach().data.cpu().numpy() if mcl_loss != 0 else 0
        _run.log_scalar('loss', binary_loss)
        _run.log_scalar('mcl_loss', mcl_loss)
        log_loss['loss'] += binary_loss
        log_loss['mcl_loss'] += mcl_loss


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            mcl_loss = log_loss['mcl_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, mcl_loss: {mcl_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
