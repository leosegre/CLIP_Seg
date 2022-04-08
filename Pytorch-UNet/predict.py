import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import transforms_seg as T


from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import Transformer_MM_Explainability.CLIP.clip as clip
from utils.utils import interpret, R_img_resize, texts2r_image


def predict_img(net,
                full_img,
                device,
                transform,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = full_img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.module.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size()[1], full_img.size()[2])),
            transforms.ToTensor()
        ])


        full_mask = tf(probs.cpu()).squeeze()
        # return full_mask.numpy()
        # full_mask = probs

    if net.module.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.module.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--text', '-text', metavar='INPUT', nargs='+', help='description of the object', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--viz_map', '-vm', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=4, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = torch.nn.DataParallel(net)
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    voc12_template = [
        'A {} in the scene',    
        # 'a bad photo of the {}.',
    ]


    logging.info('Model loaded!')

    # Load CLIP model
    clip.clip._MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    }

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        if np.array(img).shape[2] == 4:
            img = Image.fromarray(np.array(img)[..., :3])

        transform = T.Compose([
        # you can add other transformations in this list
        T.RandomResize(224),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        # T.Normalize_lite()
        ])

        mask = Image.fromarray(np.zeros(img.size).T)
        img, _ = transform(img, mask)



        R_image = torch.zeros_like(img[0], dtype=torch.float32)
        # CLIP Explainability
        to_pil = transforms.ToPILImage()
        clip_img = clip_preprocess(to_pil(img)).unsqueeze(0).to(device)
        class_name = args.text
        R_image_temp = texts2r_image(class_name, voc12_template, device, clip_model, clip_img)
        R_image_temp = (R_image_temp - R_image_temp.min()) / (R_image_temp.max() - R_image_temp.min())
        R_image = R_image_temp
        img = torch.cat((img, R_image.unsqueeze(axis=0)), axis=0)



        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           transform=transform)

        # threshold = 0.7
        # print(mask.shape)
        mask = mask[1]
        # mask[mask>=threshold] = 1
        # mask[mask<threshold] = 0
        

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            print(mask.shape)
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

        if args.viz_map:
            logging.info(f'Visualizing map for image {filename}, close to continue...')
            print(R_image.size(), img.size())
            plot_img_and_mask(img[:3, ...], R_image)

