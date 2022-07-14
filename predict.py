import numpy as np
import cv2
import torch
from network import SegModel
import argparse
import os
from datasets import get_post_process_augmentation


def parse_args():
    parser = argparse.ArgumentParser(description='prediction outcome of pytorch model for a dataset')

    parser.add_argument('image_dir', type=str, help='The path of the image directory')
    parser.add_argument('out_mask_dir', type=str, help='The path of the output mask directory')
    parser.add_argument('model_file', type=str, help='The path of the model weights file')

    # IO and norm options
    parser.add_argument('--mask_dir', type=str, default=None, help='The path of the experiment mask directory')
    parser.add_argument('-n','--n_classes', type=int, default=3,
                        help='Number of classes for training (default is 3)')
    parser.add_argument('--height', type=int, default=512, help="image height to resize (default:512)")
    parser.add_argument('--width', type=int, default=512, help="image width to resize (default:512)")
    parser.add_argument('--post_trans', default=0, type=int, 
                        help = "Flag (1 for True and 0 for False) to allow post processing transformation. (default:0)")
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


if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    out_mask_dir = args.out_mask_dir
    model_file = args.model_file
    n_classes = args.n_classes
    encoder = args.encoder
    height = args.height
    width = args.width
    post_trans = None if args.post_trans==0 else get_post_process_augmentation(n_classes=n_classes)
    
    print(post_trans)
    print(args)

    model = SegModel(
        n_classes=n_classes, 
        encoder=encoder,
    )
    checkpoint = model_file
    a = torch.load(checkpoint)
    model.load_state_dict(a['state_dict'], strict=False)
    model1 = model.eval().cpu()

    images = os.listdir(image_dir)
    if mask_dir is not None:
        masks = os.listdir(mask_dir)
        images = sorted(list(set(images).intersection(masks)))
    os.makedirs(out_mask_dir, exist_ok=True)
    img_h = height
    img_w = width
    
    for image in images:
        a = cv2.imread(os.path.join(image_dir, image))
        a = cv2.resize(a, (img_h, img_w))
        a1 = torch.tensor(np.array([a], dtype=np.float32))/255.
        a1 = a1.permute(0,3,1,2)
        a1 = model1(a1).detach().numpy().squeeze()
        if post_trans:
            a1 = post_trans(a1).numpy().squeeze()
        print(a1.max(), a1.min())
        b = np.zeros((img_h,img_w), dtype=np.uint8)
        b = np.argmax(a1, axis=0)
        print(np.unique(b), type(b))
        cv2.imwrite(os.path.join(out_mask_dir, image), b.astype(np.uint8))
