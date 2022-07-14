import pandas as pd
from network import SegModel
from datasets import SegDataset, SegDataModule, get_training_augmentation, get_validation_augmentation, get_post_process_augmentation
from utils import get_class_weights, get_list_of_image_and_mask
import torch
import pytorch_lightning as pl
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Inference of pytorch model on provided test dataset')

    parser.add_argument('image_dir', type=str, help='The path of the image directory')
    parser.add_argument('mask_dir', type=str, help='The path of the mask directory')
    parser.add_argument('model_file', type=str, help='The path of the model weights file')
    parser.add_argument('output_file', type=str, help='The path of the output file (.csv)')

    # IO and norm options
    parser.add_argument('--height', type=int, default=512,
                        help='Image height (default is 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default is 512)')
    parser.add_argument('-l', '--loss', type=str, default='ce',
                        help="""list of available loss functions 
                        'dice': 'DiceLoss',
                        'ce':'CrossEntropyLoss',
                        'dice_ce': 'DiceCELoss',
                        'focal': 'FocalLoss',
                        'tversky': 'TverskyLoss',
                        'gd': 'GeneralizedDiceLoss',
                        'dice_focal': 'DiceFocalLoss',
                        'iou': 'DiceLoss',
                        'gwd': 'GeneralizedWassersteinDiceLoss',
                        (default:'ce')""")
    parser.add_argument('-n','--n_classes', type=int, default=3,
                        help='Number of classes for training (default is 3)')
    parser.add_argument('-b', '--batchsize', type=int, default=2,
            help="Batch-size (default:12")
    parser.add_argument('-g', '--gpus', type=int, default=1,
            help="Number of GPUs (default:1)")
    parser.add_argument('--post_trans', default=0, type=int, 
                        help = "Flag (1 for True and 0 for False) to allow post processing transformation. (default:0)")
    parser.add_argument('--threads', type=int, default=2,
            help='Number of threads for training (default is 2)')

    parser.add_argument('--encoder', type=int, default=13,
                        help="""Encoder architecture availables 
                        0: 'densenet121',  
                        1: 'densenet169', 
                        2: 'densenet201', 
                        3: 'densenet161', 
                        4: 'efficientnet-b0', 
                        5: 'efficientnet-b1', 
                        6: 'efficientnet-b2', 
                        7: 'efficientnet-b3', 
                        8: 'efficientnet-b4', 
                        9: 'efficientnet-b5', 
                        10: 'efficientnet-b6', 
                        11: 'efficientnet-b7', 
                        12: 'resnet18', 
                        13: 'resnet34', 
                        14: 'resnet50', 
                        15: 'resnet101', 
                        16: 'resnet152', 
                        17: 'vgg11', 
                        18: 'vgg11_bn',  
                        19: 'vgg13', 
                        20: 'vgg13_bn',  
                        21: 'vgg16', 
                        22: 'vgg16_bn',  
                        23: 'vgg19', 
                        24: 'vgg19_bn' 
                        (default:13 (resnet34))""")

    parser.set_defaults()

    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    print(args)
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    n_classes = args.n_classes
    height = args.height
    width = args.width
    batchsize = args.batchsize
    weights = args.model_file
    encoder = args.encoder
    gpus = args.gpus
    threads = args.threads
    output_file = args.output_file
    loss_fn = args.loss
    post_trans = None if args.post_trans==0 else get_post_process_augmentation(n_classes=n_classes) 
    
    transform_ = get_validation_augmentation(height=height, width=width)
    
    images, segs = get_list_of_image_and_mask(image_dir, mask_dir)
    dataset = SegDataset(image_dir, mask_dir, resize=(height, width), transform=transform_)
    datamodule = SegDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        batch_size=batchsize, 
        num_workers=threads,
    )

    model = SegModel(n_classes=n_classes, 
                     encoder=encoder,
                     post_trans=post_trans,
                     loss_fn=loss_fn,
                    )
    checkpoint = weights
    a = torch.load(checkpoint)
    model.load_state_dict(a['state_dict'], strict=False)

    if gpus > 1:
        trainer = pl.Trainer(devices=gpus, accelerator="gpu", strategy="dp",)
    elif gpus == 1:
        trainer = pl.Trainer(gpus=gpus)
    else:
        trainer = pl.Trainer(accelerator="cpu")

    inference = trainer.test(model, datamodule)

    inference = pd.Series(inference[0])
    
    inference.to_csv(args.output_file, header=False)
    
    print(inference)
