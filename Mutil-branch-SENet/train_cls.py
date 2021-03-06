import torch
from metrics import *
import numpy as np
import argparse
from pathlib import Path
from logger import getLogger
import time
import os.path as osp
import os
from tensorboardX import SummaryWriter
from dataset_otherdata import Data, getGradient
from torch.utils.data import DataLoader
from ranger import Ranger
from model import model
import segmentation_models_pytorch.utils.losses as smploss
import torch.nn as nn
from postProcess import proc
from losses import DiceLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=1000)
    parse.add_argument('--batchsizeTrain', type=int, default=4)
    parse.add_argument('--batchsizeTest', type=int, default=2)
    parse.add_argument('--rootPth', type=str, default='./skin')
    parse.add_argument('--logPth', type=str, default='../log')
    parse.add_argument('--numWorkers', type=int, default=14)
    parse.add_argument('--evalFrequency', type=int, default=50)
    parse.add_argument('--saveFrequency', type=int, default=100)
    parse.add_argument('--msgFrequency', type=int, default=5)
    parse.add_argument('--tensorboardPth', type=str, default='../tensorboard')
    parse.add_argument('--modelPth', type=str, default='../model_cls')

    return parse.parse_args()


def eval(net, dataloader, logger):
    dq, sq, pq, dice = [], [], [], []
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(device)
            branchSeg, branchMSE, branchSe = net(img)
            pred = torch.cat((branchSeg, branchMSE), dim=1)
            for i in range(pred.shape[0]):
                output = proc(pred[i, ...])
                metricPQ, _ = get_fast_pq(mask[i, ...], output)
                metricDice = get_dice_1(mask[i, ...], output)
                dq.append(metricPQ[0])
                sq.append(metricPQ[1])
                pq.append(metricPQ[-1])
                dice.append(metricDice)

    logger.info(f'\n'
                f'- dq: {np.mean(dq):.4f}, \n'
                f'- sq: {np.mean(sq):.4f}, \n'
                f'- pq: {np.mean(pq):.4f}, \n'
                f'- Dic6e: {np.mean(dice):.4f}.')

def main(args, logger):
    writter = SummaryWriter(logdir=args.subTensorboardPth)
    trainSet = Data(root=Path(args.rootPth) / 'train',
                    mode='train',
                    isAugmentation=True,
                    cropSize=(384, 384))
    trainLoader = DataLoader(trainSet,
                             batch_size=args.batchsizeTrain,
                             shuffle=True,
                             pin_memory=False,
                             drop_last=True,
                             num_workers=args.numWorkers)
    testSet = Data(root=Path(args.rootPth) / 'test',
                   mode='test')
    testLoader = DataLoader(testSet,
                            batch_size=args.batchsizeTest,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=args.numWorkers)
    net = model().to(device)
    x = torch.autograd.Variable(torch.randn(1, 3, 512, 512)).to(device)
    # writter.add_graph(net, x)
    net = nn.DataParallel(net, device_ids=[1])
    criterionMSE = nn.MSELoss().to(device)
    BinaryDice = smploss.DiceLoss(eps=1e-7).to(device)
    Dice = DiceLoss()
    criterionCE = nn.BCELoss().to(device)
    criterionMCE = nn.CrossEntropyLoss().to(device)
    optimizer = Ranger(net.parameters(), lr=1.e-2)
    runningLoss, MSEloss, CEloss, Diceloss = [], [], [], []
    iter = 0
    for epoch in range(args.epoch):
        if epoch != 0 and epoch % args.evalFrequency == 0:
            logger.info(f'===============Eval after epoch {epoch}...===================')
            # eval(net, testLoader, logger)

        if epoch != 0 and epoch % args.saveFrequency == 0:
            logger.info(f'===============Save after epoch {epoch}...===================')
            modelName = Path(args.subModelPth) / f'out_{epoch}.pth'
            state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state_dict, modelName)

        for img, mask, horizontalVertical, mask_se in trainLoader:
            iter += 1
            img, mask, horizontalVertical, mask_se = img.to(device), mask.to(device), horizontalVertical.to(device), mask_se.to(device)
            optimizer.zero_grad()
            [branchSeg, branchMSE, branchse] = net(img)
            predictionGradient = getGradient(branchMSE)
            gtGradient = getGradient(horizontalVertical)

            loss1 = criterionMSE(branchMSE, horizontalVertical) + 2. * criterionMSE(predictionGradient, gtGradient)
            loss2 = criterionCE(branchSeg, mask)
            loss3 = BinaryDice(branchSeg, mask)

            loss4 = criterionMCE(branchse, mask_se.long().squeeze())
            loss5 = Dice(branchse, mask_se)

            loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss.backward()
            optimizer.step()
            MSEloss.append(loss1.item())
            CEloss.append(loss2.item())
            Diceloss.append(loss3.item())
            runningLoss.append(loss.item())
            if iter % args.msgFrequency == 0:
                logger.info(f'epoch:{epoch}/{args.epoch}, '
                            f'loss:{np.mean(runningLoss):.4f}, '
                            f'MSEloss:{np.mean(MSEloss):.4f}, '
                            f'Diceloss:{np.mean(Diceloss):.4f}, '
                            f'CEloss:{np.mean(CEloss):.4f}')

                writter.add_scalar('Loss', np.mean(runningLoss), iter)
                runningLoss, MSEloss, CEloss, Diceloss = [], [], [], []

    # eval(net, testLoader, logger)
    modelName = Path(args.subModelPth) / 'final.pth'
    state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state_dict, modelName)


if __name__ == '__main__':
    args = parseArgs()
    uniqueName = time.strftime('%y%m%d-%H%M%S')
    args.subModelPth = osp.join(args.modelPth, uniqueName)
    args.subTensorboardPth = osp.join(args.tensorboardPth, uniqueName)
    for subDir in [args.logPth,
                   args.subModelPth,
                   args.subTensorboardPth]:
        if not osp.exists(subDir):
            os.makedirs(subDir)

    logFile = osp.join(args.logPth, uniqueName + '.log')
    logger = getLogger(logFile)
    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main(args, logger)