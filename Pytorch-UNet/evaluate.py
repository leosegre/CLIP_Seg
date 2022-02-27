import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import Transformer_MM_Explainability.CLIP.clip as clip
from utils.utils import interpret, R_img_resize



from utils.dice_score import multiclass_dice_coeff, dice_coeff






def evaluate(net, dataloader, device, labels_dict, clip_model, clip_preprocess):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for (image, mask_true) in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type

        R_image = torch.zeros_like(mask_true)


        for i in range(mask_true.size()[0]):
            # print(true_masks[i].unique())
            elements = mask_true[i].unique()
            if len(elements) <= 2:
                element_to_mask = 100 # Label does not exist
            else:
                element_to_mask = np.random.choice(mask_true[i].unique()[1:-1])
            # print("element_to_mask", element_to_mask)
            # print("true_masks_size", (true_masks[i] != element_to_mask).size())
            mask_true[i][mask_true[i] != element_to_mask] = 0
            mask_true[i][mask_true[i] == element_to_mask] = 1
            # print(true_masks[i].unique())

            # CLIP Explainability
            to_pil = transforms.ToPILImage()
            clip_img = clip_preprocess(to_pil(image[i])).unsqueeze(0).to(device)
            texts = [labels_dict[element_to_mask]]
            text = clip.tokenize(texts).to(device)

            R_image_temp = interpret(model=clip_model, image=clip_img, texts=text, device=device)
            R_image[i] = R_img_resize(R_image_temp)

        image = torch.cat((image, R_image.unsqueeze(axis=1)), axis=1)
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        mask_true = F.one_hot(mask_true, net.module.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.module.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
