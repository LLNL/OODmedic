import numpy as np
import os

def _hparams(dataset):
    """
    Global registry of hyperparams.
    """

    hparams = {}
    if 'mnist' in dataset:
        hparams['data_dir'] = './data/'+dataset
        hparams['augment'] = True
        hparams['augmix'] = False
        hparams['augment_type'] = 'standard'
        hparams['result_dir'] = './logs'
        hparams['checkpoint_dir'] = './ckpts'
        hparams['csv_dir'] = './data/'+dataset
        hparams['train_csv'] = os.path.join(hparams['csv_dir'], dataset+'_in_train.csv')
        hparams['val_csv'] = os.path.join(hparams['csv_dir'], dataset+'_in_val.csv')
        hparams['optimizer'] = 'Adam' # 'SGD' # for CosineAnnealingLR
        hparams['momentum'] = 0.9
        hparams['lr'] = 1e-3 #1e-5 # 1e-4 # 5e-5
        hparams['lr_patience'] = 30 #10 #5  #2
        hparams['lr_factor'] = 0.5 #0.1 #0.2
        hparams['batch_size'] = 128 #128
        hparams['im_size'] = 28 #96
        if dataset == 'bloodmnist' or dataset == 'tissuemnist' or dataset == 'pathmnist':
            hparams['num_classes'] = 6
        elif dataset == 'organcmnist' or dataset == 'organamnist' or dataset == 'organsmnist':
            hparams['num_classes'] = 8
        elif dataset == 'dermamnist':
            hparams['num_classes'] = 5
        elif dataset == 'octmnist':
            hparams['num_classes'] = 3

        if dataset == 'bloodmnist' or dataset == 'organcmnist' or dataset == 'organamnist' or dataset == 'organsmnist' or dataset == 'dermamnist':
            hparams['num_workers'] = 4
        elif dataset == 'tissuemnist':
            hparams['num_workers'] = 1
        elif dataset == 'tissuemnist' or dataset == 'pathmnist' or dataset == 'octmnist':
            hparams['num_workers'] = 0

        hparams['weight_decay'] = 5e-4 #1e-4
        hparams['epochs'] = 100

    else:
        raise NotImplementedError

    return hparams


def default_hparams(dataset):
    return {a: b for a, b in _hparams(dataset).items()}

if __name__ == '__main__':
    hparams = default_hparams('bloodmnist')
    print(hparams)
