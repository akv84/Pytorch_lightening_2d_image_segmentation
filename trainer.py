
from network import SegModel
from datasets import SegDataset, SegDataModule, get_training_augmentation, get_validation_augmentation, get_post_process_augmentation
from utils import get_class_weights, get_list_of_image_and_mask
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging, ProgressBar, ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, NeptuneLogger, MLFlowLogger, TensorBoardLogger
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Training of pytorch model')
    parser.add_argument('train_image_dir', type=str, help='The path of the train image directory')
    parser.add_argument('train_mask_dir', type=str, help='The path of the train mask directory')
    parser.add_argument('val_image_dir', type=str, help='The path of the validation image directory')
    parser.add_argument('val_mask_dir', type=str, help='The path of the validation mask directory')
    parser.add_argument('outputdir', type=str, help='The path of the output directory')

    # IO and norm options
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
    parser.add_argument('--height', type=int, default=512,
                        help='Image height (default is 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default is 512)')
    parser.add_argument('-r', '--learningrate', type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument('-b', '--batchsize', type=int, default=2,
                        help="Batch-size (default:2)")
    parser.add_argument('-a', '--augmentation', default=0, type=int, 
                        help = "Flag (1 for True and 0 for False) for augmentation. (default:0)")
    parser.add_argument('--post_trans', default=0, type=int, 
                        help = "Flag (1 for True and 0 for False) to allow post processing transformation. (default:0)")
    parser.add_argument('-w', '--weights', type=str, default=None,
                        help="weights initialization file of pytorch model (default:None)")
    parser.add_argument('-p', '--pretrained', default=0, type=int, 
                        help = "Flag (1 for True and 0 for False) for pretrained weights. (default:0)")
    parser.add_argument('-c','--class_weight', type=int, default=1,
                        help = "Flag (1 for True and 0 for False) for class weights. (default:1)")
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
    parser.add_argument('-g', '--gpus', type=int, default=1,
            help="Number of GPUs (default:1)")
    parser.add_argument('--threads', type=int, default=2,
            help='Number of threads for training (default is all cores)')

    # training options
    parser.add_argument('-e', '--epochs', type=int, default=300,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss (default: 300)")
    parser.add_argument('--earlystop', type=int, default=15,
            help="Number of epochs to stop training if no improvement in loss (default: 15)")
    parser.add_argument('--top_k', type=int, default=5,
            help="Number of best models to save during training (default: 5)")
    parser.add_argument('--api_key', type=str, default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMTcwMzA5ZS1kNzQ3LTRkMWItOTIyMy1hZDZmZTNiM2I0NTIifQ==",
            help="Neptune login api key (default: 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMTcwMzA5ZS1kNzQ3LTRkMWItOTIyMy1hZDZmZTNiM2I0NTIifQ==')")
    parser.add_argument('--project', type=str, default='akv84,unet-binary-mask',
            help="Naptune project parameters comma seperated values of user,project_name. (default: 'akv84,unet-binary-mask')")
    parser.add_argument('--tags', type=str, default='akv84,unet-binary-mask',
            help="Naptune tags comma seperated strings e.g. 'tag1,tag2,tag3' (default: 'training,model')")

    parser.set_defaults()

    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    print(args)
    train_image_dir = args.train_image_dir
    train_mask_dir = args.train_mask_dir
    val_image_dir = args.val_image_dir
    val_mask_dir = args.val_mask_dir
    outputdir = args.outputdir
    loss_fn = args.loss
    n_classes = args.n_classes
    height = args.height
    width = args.width
    learningrate = args.learningrate
    batchsize = args.batchsize
    augmentation = args.augmentation
    weights = args.weights
    pretrained = args.pretrained
    encoder = args.encoder
    gpus = args.gpus
    threads = args.threads
    epochs = args.epochs
    earlystop = args.earlystop
    api_key = args.api_key
    project = '/'.join(args.project.split(','))
    tags = args.tags.split(',')
    class_weight = args.class_weight
    post_trans = None if args.post_trans==0 else get_post_process_augmentation(n_classes=n_classes)
    top_k = args.top_k
    print(post_trans)

    # prepaire augmentation parameters
    train_transform = get_training_augmentation(height=height, width=width) if augmentation else get_validation_augmentation(height=height, width=width)(height=height, width=width)
    val_transform = get_validation_augmentation(height=height, width=width)
       
    # prepaire dataloaders
    train_dataset = SegDataset(train_image_dir, train_mask_dir, resize=(height, width), transform=train_transform)
    val_dataset = SegDataset(val_image_dir, val_mask_dir, resize=(height, width), transform=val_transform)

    # prepaire DataModule for training
    datamodule = SegDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        batch_size=batchsize, 
        num_workers=threads,
    )
    # prepaire Model for training
    if class_weight:
        class_weights = get_class_weights(train_mask_dir, n_classes)  #torch.as_tensor([0.05, 0.45, 0.5])
    else:
        class_weights = None
    print(class_weights)
    model = SegModel(n_classes=n_classes, 
                     lr=learningrate,
                     encoder=encoder,
                     pretrained=pretrained,
                     class_weights=class_weights,
                     loss_fn=loss_fn,
                     post_trans=post_trans,
                    )
    if weights is not None:
        checkpoint = weights
        a = torch.load(checkpoint)
        model.load_state_dict(a['state_dict'], strict=False)
    
    # prepaire callbacks for training
    checkpoint_callback = ModelCheckpoint(
        dirpath=outputdir,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor = "val_loss",  #monitors val loss
        mode = "min",  #Picks the fold with the lowest val_loss
        save_weights_only=False,
        save_last=True,
        save_top_k=top_k,
        verbose=True,
    )
    multiplicative = lambda epoch: 1.5
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=earlystop)
    Stochastic_Weight_Averaging = StochasticWeightAveraging()
    callbacks = [
        checkpoint_callback, 
        early_stopping,
        Stochastic_Weight_Averaging,
        ProgressBar(),
        LearningRateMonitor(logging_interval='step'),
        # Add a new callback here,
        DeviceStatsMonitor()
    ]
    
    # prepaire Loggers for training
    csv_logs = CSVLogger(outputdir, name="csv_logs")
    tb_logs = TensorBoardLogger(os.path.join(outputdir, 'tb_logs/'))
    neptune_logs = NeptuneLogger(   
        api_key=api_key,  # replace with your own
        project=project,  # format "<WORKSPACE/PROJECT>"
        tags=tags,  # optional
    )

    loggers = [
        csv_logs,
        #neptune_logs,
        # Add a new logger here,
        tb_logs
    ]
        
    if gpus > 1:

        trainer = pl.Trainer(devices=gpus, accelerator="gpu", strategy="dp",
                             max_epochs=epochs,
                             logger=loggers,
                             callbacks=callbacks,
                             log_every_n_steps=10,
                            )
    elif gpus == 1:
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=epochs,
                             logger=loggers,
                             callbacks=callbacks,
                             log_every_n_steps=10,
                            )
    else:
        trainer = pl.Trainer(accelerator="cpu",
                             max_epochs=epochs,
                             logger=loggers,
                             callbacks=callbacks,
                             log_every_n_steps=10,
                            )

    trainer.fit(model, datamodule)

