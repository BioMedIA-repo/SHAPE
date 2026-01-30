import logging
import numpy as np
import torch

from utils import binary


def evaluate(val_dataset, net, num_classes, domain_type):
    net.eval()
    all_batch_dice = []
    all_batch_asd = []
    logger = logging.getLogger('main_logger')
    logger.info('Validation started.')

    with torch.no_grad():
        for idx, batch in enumerate(val_dataset):
            xt, xt_labels = batch['s'].cuda(), batch['label'].cuda()
            output = net.inference(xt)
            output = torch.softmax(output, dim=1)
            out = torch.argmax(output, dim=1)

            out_np = out.cpu().numpy()
            xt_labels_np = xt_labels.cpu().numpy()

            for ind in range(out_np.shape[0]):
                batch_dice = []
                batch_asd = []

                out_img_np = out_np[ind]
                xt_lab_img_np = xt_labels_np[ind].squeeze(0)

                if np.sum(xt_lab_img_np) == 0:
                    continue

                for i in range(1, num_classes):
                    pred = (out_img_np == i)
                    gt = (xt_lab_img_np == i)
                    dice, jc, hd, asd = calculate_metric_percase(pred, gt)
                    batch_dice.append(dice)
                    batch_asd.append(asd)

                all_batch_dice.append(batch_dice)
                all_batch_asd.append(batch_asd)

    all_batch_dice = np.array(all_batch_dice)
    all_batch_asd = np.array(all_batch_asd)

    mean_dice = np.mean(np.ma.masked_equal(all_batch_dice, 0), axis=0)
    mean_asd = np.mean(np.ma.masked_equal(all_batch_asd, 0), axis=0)

    mean_dice = np.array(mean_dice, dtype=np.float64)
    mean_asd = np.array(mean_asd, dtype=np.float64)

    mean_dice[np.isnan(mean_dice)] = 0.0
    mean_asd[np.isnan(mean_asd)] = 0.0

    total_mean_dice = np.mean(mean_dice)
    total_mean_asd = np.mean(mean_asd)
    logger.info('Per class metrics:')
    logger.info('  Dice : {}'.format(np.round(mean_dice, 3)))
    logger.info('  ASD  : {}'.format(np.round(mean_asd, 3)))
    logger.info('Overall Mean Metrics:')
    logger.info('  Mean Dice: {:.5f}'.format(total_mean_dice))
    logger.info('  Mean ASD : {:.5f}'.format(total_mean_asd))

    return total_mean_dice


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            dice = binary.dc(pred, gt) * 100
            jc = binary.jc(pred, gt)
            hd = binary.hd95(pred, gt)
            asd = binary.asd(pred, gt)
            return dice, jc, hd, asd
        except RuntimeError as e:
            logging.error(f"Error calculating metrics: {e}")
            return 0, 0, 0, 0
    else:
        return 0, 0, 0, 0
