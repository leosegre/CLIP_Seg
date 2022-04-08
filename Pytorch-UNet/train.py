import argparse
import logging
import sys
from pathlib import Path
from PIL import ImageOps

import torchvision
from torchvision import transforms
import transforms_seg as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.utils import interpret, R_img_resize, texts2r_image


from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import Transformer_MM_Explainability.CLIP.clip as clip


dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_dataset = Path('./data/dataset/')
dir_checkpoint = Path('./checkpoints/')





def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 1. Create dataset
    transform = T.Compose([
    # you can add other transformations in this list
    T.RandomResize(224),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
    # T.Normalize_lite()
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = torchvision.datasets.VOCSegmentation(root = dir_dataset, year = '2012', image_set = 'train', download = True, transform = None, target_transform = None, transforms = transform)
    val_dataset = torchvision.datasets.VOCSegmentation(root = dir_dataset, year = '2012', image_set = 'val', download = True, transform = None, target_transform = None, transforms = transform)
    
    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)

    # Load CLIP model
    clip.clip._MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    }

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)


    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # Set labels dict
    labels_dict = {
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor',
        100: 'an object'
    }

    # voc12_template = [
    #     'itap of a {}.',
    #     'a bad photo of the {}.',
    #     'a photo of the large {}.',
    #     'art of the {}.',
    #     'a photo of the small {}.',
    #     'a photo of the nice {}.',
    #     'a cropped photo of the {}.'
    # ]

    voc12_template = [
        'A {} in the scene',    
        # 'a bad photo of the {}.',
    ]

    distractors_template = [
        'A scene',    
        # 'A picture of the moon.',
        # 'A pineapple.',
    ]

    # augs = [
    #     lambda x: ImageOps.flip(x),   
    #     lambda x: ImageOps.mirror(x),
    #     lambda x: ImageOps.mirror(ImageOps.flip(x))
    # ]

    augs = [
        # torchvision.transforms.RandomVerticalFlip(p=1),
        # torchvision.transforms.RandomHorizontalFlip(p=1),
        # torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(p=1), torchvision.transforms.RandomHorizontalFlip(p=1)])
    ]

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (images, true_masks) in train_loader:
                # images = batch['image']
                # true_masks = batch['mask']

                R_image = torch.zeros_like(true_masks, dtype=torch.float32)
                # relevance_map = torch.zeros_like(true_masks, dtype=torch.uint8)

                # print(images.unique())
                for i in range(true_masks.size()[0]):
                    # print(true_masks[i].unique())
                    elements = true_masks[i].unique()
                    if len(elements) <= 2:
                        element_to_mask = 100 # Label does not exist
                    else:
                        element_to_mask = np.random.choice(true_masks[i].unique()[1:-1])
                    # print("element_to_mask", element_to_mask)
                    # print("true_masks_size", (true_masks[i] != element_to_mask).size())
                    true_masks[i][true_masks[i] != element_to_mask] = 0
                    true_masks[i][true_masks[i] == element_to_mask] = 1
                    # print(true_masks[i].unique())

                    # CLIP Explainability
                    to_pil = transforms.ToPILImage()
                    clip_img = clip_preprocess(to_pil(images[i])).unsqueeze(0).to(device)
                    class_name = [labels_dict[element_to_mask]]
                    
                    R_image_temp = texts2r_image(class_name, voc12_template, device, clip_model, clip_img)
                    for aug in augs:
                        R_image_temp += aug(texts2r_image(class_name, voc12_template, device, clip_model, aug(clip_img)))
                    R_image_temp /= len(augs)+1

                    # distractors_R_image_temp = texts2r_image(class_name, distractors_template, device, clip_model, clip_img)
                    # for aug in augs:
                    #     distractors_R_image_temp += aug(texts2r_image(class_name, distractors_template, device, clip_model, aug(clip_img)))
                    # distractors_R_image_temp /= len(augs)+1

                    # R_image_temp -= distractors_R_image_temp
                    R_image_temp = (R_image_temp - R_image_temp.min()) / (R_image_temp.max() - R_image_temp.min())
                    # R_image_temp = torch.clamp(R_image_temp, min=0)

                    R_image[i] = R_image_temp

                    # R_image[i] = R_img_resize(R_image_temp)
                    # print(R_image[i])
                    # relevance_map[i] = (255 * R_image[i]).to(torch.uint8)

                    # relevance_map[i] = show_image_relevance(R_image[i], clip_img)
                

                # assert images.shape[1] == net.n_channels, \
                    # f'Network has been defined with {net.n_channels} input channels, ' \
                    # f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    # 'the images are loaded correctly.'

                orig_images = images.clone()
                images = torch.cat((images, R_image.unsqueeze(axis=1)), axis=1)
                # print(images.size())
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.module.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = ((len(train_dataset) // (batch_size))+1)
                if division_step > 0:
                    if (global_step % division_step == 0) and (epoch % 5 == 0):
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, labels_dict, voc12_template, distractors_template, clip_model, clip_preprocess, augs)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # print(R_image[0].unique())
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': {
                                'image': wandb.Image(orig_images[0].cpu()),
                                'relevance_map':wandb.Image(R_image[0].float().cpu()),
                            },
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--device_ids', '-d', type=str, default=None, help='Device ids to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    first_device = 0 if args.device_ids==None else int(args.device_ids[0])
    torch.cuda.set_device(first_device)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=4, n_classes=2, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    if args.device_ids == None:
        device_ids = None
    else:
        device_ids = list(map(int, args.device_ids.split(",")))

    net.to(device=device)
    try:
        train_net(net=torch.nn.DataParallel(net, device_ids=device_ids),
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
