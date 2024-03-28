import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os


trainDir="Fotos/Training/New Annotations/default/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available


batchSize=2
imageSize=[2048,1140]


imgs = []
msk_imgs = []
MasterTail = []
MasterBeaver = []
for filename in os.listdir(trainDir):
    imgs.append(os.path.join(trainDir,filename))
    msk = os.path.join(trainDir,filename)
    msk = msk.replace("default", "defaultannot")
    msk = msk.replace("JPG", "png")
    msk_imgs.append(msk)

    MasterMask = cv2.imread(msk)
    MasterMask = cv2.resize(MasterMask, imageSize, cv2.INTER_LINEAR)
    
    

    tailmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
    beavermask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
    imgray = cv2.cvtColor(MasterMask, cv2.COLOR_BGR2GRAY)
    ret, beavermask = cv2.threshold(imgray, 140, 255, cv2.THRESH_BINARY)
    tailmask = cv2.inRange(imgray, 150, 157)
    
    MasterTail.append(tailmask)
    MasterBeaver.append(beavermask)
    

    
def loadData():
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    for i in range(batchSize):
        idx = random.randint(0,len(imgs)-1)
        img = cv2.imread(imgs[idx])
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)

        masks=[MasterBeaver[idx],MasterTail[idx]]

        num_objs = len(masks)
        if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for i in range(24001):
            images, targets = loadData()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            if i%2000==0:
                torch.save(model.state_dict(), str(i)+"_RCNN-2.torch")



