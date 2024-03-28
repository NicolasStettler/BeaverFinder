###############################################
############ Inference multiple Images ############
###############################################


import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
from PIL import Image
import os
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

import torchvision.models.segmentation
import torch
import math
import csv
import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
from tkinter import simpledialog



imageSize=[2048,1140]
testDir="Fotos/TrueData/BeaverTails_BE02_Lower_SecondBatch/"


imgs=os.listdir(testDir)


device = torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.load_state_dict(torch.load("28000_RCNN-3.torch"))
model.to(device)# move model to the right devic
model.eval()



def predict_image(images):
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)


    #use the model to predict the image
    with torch.no_grad():
        pred = model(images)
    if len(pred[0]['masks'])<2:
        Norecognition = True
    else: 
        Norecognition = False

    if Norecognition == False:
        #Prepare image
        im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

        #Prepare an empty image where the prediction will be drawn on
        firstmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
        secondmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)


        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][1,0].detach().cpu().numpy()
            scr=pred[0]['scores'][1].detach().cpu().numpy()
            if scr>0.8 :
                firstmask[:,:,0][msk>0.5] = 30
                firstmask[:, :, 1][msk > 0.5] = 50
                firstmask[:, :, 2][msk > 0.5] = 60

        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][0,0].detach().cpu().numpy()
            scr=pred[0]['scores'][0].detach().cpu().numpy()
            if scr>0.8 :
                secondmask[:,:,0][msk>0.5] = 40
                secondmask[:, :, 1][msk > 0.5] = 60
                secondmask[:, :, 2][msk > 0.5] = 70

        #########Contours##########
                #The predictions are now mapped to simple binary png's, however as we want to measure the tail, it's easier to work with contours

        #Functions
        def convertcountours(x_contours):
            x_MeanPoint_Y = [None]*len(x_contours)
            x_MeanPoint_X = [None]*len(x_contours)
            x_contours_poly = [None]*len(x_contours)

            for i, c in enumerate(x_contours):
                x_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                list_y = []
                list_x = []
                for a in range(len(x_contours_poly[i])):
                    list_x.append(x_contours_poly[i][a][0][0])
                    list_y.append(x_contours_poly[i][a][0][1])
                x_MeanPoint_Y[i] = np.mean(list_y)
                x_MeanPoint_X[i] = np.mean(list_x)
            return (x_MeanPoint_X, x_MeanPoint_Y, x_contours_poly)

        #####Create Contour images
        #Create contour for tail
        first_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
        first_thresh = cv2.Canny(first_imgray, 50, 100)
        second_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
        second_thresh = cv2.Canny(second_imgray, 50, 100)

        if np.sum(first_thresh == 255) > np.sum(second_thresh == 255):
            b_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            b_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        t_Rechteck = [None]*len(t_contours)
        b_Rechteck = [None]*len(b_contours)


        #This function (established above) creates a polygon and the centroid coordinates
        t_MeanPoint_X, t_MeanPoint_Y, t_contours_poly = convertcountours(t_contours) 
        b_MeanPoint_X, b_MeanPoint_Y, b_contours_poly = convertcountours(b_contours) 


        #This part finds the point at the start of the beaver tail
        for i, c in enumerate(t_contours):
            if b_MeanPoint_Y[i] > t_MeanPoint_Y[i]:
                Tail_up = True
            else:
                Tail_up = False


            t_Rechteck[i] = cv2.minAreaRect(t_contours[i])
            t_Rechteck[i] = cv2.boxPoints(t_Rechteck[i])
            
            Punkt1 = t_Rechteck[i][0]
            Punkt2 = t_Rechteck[i][1]
            Punkt3 = t_Rechteck[i][2]
            Punkt4 = t_Rechteck[i][3]
            t_Rechteck[i] = np.intp(t_Rechteck[i])

            if Tail_up == True:
                if (sum((Punkt1-Punkt2)**2))**0.5 > (sum((Punkt1-Punkt4)**2))**0.5:
                    x = round((Punkt4[0] + Punkt1[0])/2)
                    y = round((Punkt4[1] + Punkt1[1])/2)
                    print("Tail Up - V1")
                else:
                    x = round((Punkt3[0] + Punkt4[0])/2)
                    y = round((Punkt3[1] + Punkt4[1])/2)
                    print("Tail Up - V2")
            else:
                if (sum((Punkt1-Punkt2)**2))**0.5 > (sum((Punkt1-Punkt4)**2))**0.5:
                    x = round((Punkt2[0] + Punkt3[0])/2)
                    y = round((Punkt2[1] + Punkt3[1])/2)
                    print("Tail Down - V1")
                else:
                    x = round((Punkt1[0] + Punkt2[0])/2)
                    y = round((Punkt1[1] + Punkt2[1])/2)
                    print("Tail Down - V2")
            

            #Find the point of the tail that is furthest away from the origin point (middle of bounding box)
            Maximal_Distance=[]
            for a in range(len(t_contours[0])):
                Ausgangspunkt = (x,y)
                Maximal_Distance.append(sum((Ausgangspunkt-t_contours[i][a][0])**2)**0.5)
            IndexvonPunkt = Maximal_Distance.index(max(Maximal_Distance))
            WeitesterPunkt = t_contours[i][IndexvonPunkt][0]
            x_2 = WeitesterPunkt[0]
            y_2 = WeitesterPunkt[1]



            #Create a vector from our origin point to the tail end
            dx = x_2-x
            dy = y_2-y
            steps = 50

            x_step = dx/steps
            y_step = dy/steps

            stepsofx = []
            stepsofy = []
            for b in range(steps+1):
                stepsofx.append(x+x_step*b)
                stepsofy.append(y+y_step*b)

            Minimal_Distance_perPoint = []
            Minimal_Distance_acrossPoints = []
            for a in range(len(stepsofx)-1):
                for b in range(len(t_contours[0])):
                    Ausgangspunkt = (stepsofx[a],stepsofy[a])
                    Minimal_Distance_perPoint.append(sum((Ausgangspunkt-t_contours[i][b][0])**2)**0.5)
                Minimal_Distance_acrossPoints.append(min(Minimal_Distance_perPoint))
            IndexvonPunkt_2 = Minimal_Distance_acrossPoints.index(min(Minimal_Distance_acrossPoints))

            x_3 = round(stepsofx[IndexvonPunkt_2])
            y_3 = round(stepsofy[IndexvonPunkt_2])
            tail_length = ((x_3-x_2)**2+(y_3-y_2)**2)**0.5
            #calculate alpha
            alpha = math.atan((x_2-x_3)/(y_3-y_2))


            #Define a function which measures the width of the tail at a particular section
            def measurewidth(section_number, sections):
                measurement1_center = [stepsofx[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)], stepsofy[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)]]
                for i in range(400):
                    tmpxr1 = round(math.cos(alpha)*i+measurement1_center[0])
                    tmpyr1 = round(math.sin(alpha)*i+measurement1_center[1])
                    if t_imgray[tmpyr1][tmpxr1] == 0:
                        break
                measurement_right=[tmpxr1,tmpyr1]
                for i in range(400):
                    tmpxl1 = round(measurement1_center[0]-math.cos(alpha)*i)
                    tmpyl1 = round(measurement1_center[1]-math.sin(alpha)*i)
                    if t_imgray[tmpyl1][tmpxl1] == 0:
                        break       
                measurement_left=[tmpxl1,tmpyl1]
                    
                

                width_coordinates = [measurement_left, measurement_right]
                width = ((tmpxr1-tmpxl1)**2+(tmpyr1-tmpyl1)**2)**0.5
                return (width_coordinates, width)

            #We iterate the above function for a certain amount of times to measure the tail at multiple distances
            width_coordinates = []
            widths = []
            numberofwidths=20
            for i in range(1,numberofwidths+1):
                tmp1, tmp2 = measurewidth(i,numberofwidths+1)
                width_coordinates.append(tmp1)
                widths.append(round(tmp2, 2))

        


        drawing = np.zeros((t_thresh.shape[0], t_thresh.shape[1], 3), dtype=np.uint8)
        for i in range(len(b_contours)):
            cv2.drawContours(im, [t_Rechteck[i]],i, (255,255,0), 2)
            cv2.drawContours(im, t_contours_poly, i, (0,255,180),3)
            cv2.line(im, (x_2, y_2),(x_3, y_3), (0,0,255), 2)
            cv2.circle(im, (x_2, y_2), 2, (0,0,255), 2)
            cv2.circle(im, (x_3, y_3), 2, (0,0,255), 2)
            cv2.circle(im, (round(Punkt1[0]), round(Punkt1[1])), 2, (0,0,255), 2)
            for b in range(len(widths)):
                cv2.line(im, (width_coordinates[b][0][0], width_coordinates[b][0][1]),(width_coordinates[b][1][0], width_coordinates[b][1][1]), (0,0,255), 2)


        

        tail_measurments = widths
        tail_measurments.insert(0, round(tail_length,2))
        tail_measurments.insert(0, "Beaver tail - Fully automatic")
        
        return(Norecognition, im, tail_measurments)

    else:
        print("Nicht erkannt")
        return(Norecognition, 0, 0)

def Adjustlength(images):
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)


    #use the model to predict the image
    with torch.no_grad():
        pred = model(images)
    if len(pred[0]['masks'])<2:
        Norecognition = True
    else: 
        Norecognition = False

    if Norecognition == False:
        #Prepare image
        im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

        #Prepare an empty image where the prediction will be drawn on
        firstmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
        secondmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)


        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][1,0].detach().cpu().numpy()
            scr=pred[0]['scores'][1].detach().cpu().numpy()
            if scr>0.8 :
                firstmask[:,:,0][msk>0.5] = 30
                firstmask[:, :, 1][msk > 0.5] = 50
                firstmask[:, :, 2][msk > 0.5] = 60

        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][0,0].detach().cpu().numpy()
            scr=pred[0]['scores'][0].detach().cpu().numpy()
            if scr>0.8 :
                secondmask[:,:,0][msk>0.5] = 40
                secondmask[:, :, 1][msk > 0.5] = 60
                secondmask[:, :, 2][msk > 0.5] = 70

        #########Contours##########
                #The predictions are now mapped to simple binary png's, however as we want to measure the tail, it's easier to work with contours

        #Functions
        def convertcountours(x_contours):
            x_MeanPoint_Y = [None]*len(x_contours)
            x_MeanPoint_X = [None]*len(x_contours)
            x_contours_poly = [None]*len(x_contours)

            for i, c in enumerate(x_contours):
                x_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                list_y = []
                list_x = []
                for a in range(len(x_contours_poly[i])):
                    list_x.append(x_contours_poly[i][a][0][0])
                    list_y.append(x_contours_poly[i][a][0][1])
                x_MeanPoint_Y[i] = np.mean(list_y)
                x_MeanPoint_X[i] = np.mean(list_x)
            return (x_MeanPoint_X, x_MeanPoint_Y, x_contours_poly)

        #####Create Contour images
        #Create contour for tail
        first_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
        first_thresh = cv2.Canny(first_imgray, 50, 100)
        second_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
        second_thresh = cv2.Canny(second_imgray, 50, 100)

        if np.sum(first_thresh == 255) > np.sum(second_thresh == 255):
            b_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            b_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        t_Rechteck = [None]*len(t_contours)
        b_Rechteck = [None]*len(b_contours)


        #This function (established above) creates a polygon and the centroid coordinates
        t_MeanPoint_X, t_MeanPoint_Y, t_contours_poly = convertcountours(t_contours) 
        b_MeanPoint_X, b_MeanPoint_Y, b_contours_poly = convertcountours(b_contours) 


        #This part finds the point at the start of the beaver tail
        for i, c in enumerate(t_contours):
            if b_MeanPoint_Y[i] > t_MeanPoint_Y[i]:
                Tail_up = True
            else:
                Tail_up = False


            t_Rechteck[i] = cv2.minAreaRect(t_contours[i])
            t_Rechteck[i] = cv2.boxPoints(t_Rechteck[i])
            
            Punkt1 = t_Rechteck[i][0]
            Punkt2 = t_Rechteck[i][1]
            Punkt3 = t_Rechteck[i][2]
            Punkt4 = t_Rechteck[i][3]
            t_Rechteck[i] = np.intp(t_Rechteck)

            if Tail_up == True:
                if (sum((Punkt1-Punkt2)**2))**0.5 > (sum((Punkt1-Punkt4)**2))**0.5:
                    x = round((Punkt4[0] + Punkt1[0])/2)
                    y = round((Punkt4[1] + Punkt1[1])/2)
                    print("Tail Up - V1")
                else:
                    x = round((Punkt3[0] + Punkt4[0])/2)
                    y = round((Punkt3[1] + Punkt4[1])/2)
                    print("Tail Up - V2")
            else:
                if (sum((Punkt1-Punkt2)**2))**0.5 > (sum((Punkt1-Punkt4)**2))**0.5:
                    x = round((Punkt2[0] + Punkt3[0])/2)
                    y = round((Punkt2[1] + Punkt3[1])/2)
                    print("Tail Down - V1")
                else:
                    x = round((Punkt1[0] + Punkt2[0])/2)
                    y = round((Punkt1[1] + Punkt2[1])/2)
                    print("Tail Down - V2")
            

            #Find the point of the tail that is furthest away from the origin point (middle of bounding box)
            def set_point(event,x,y,flags,param):
                global mouseX,mouseY
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(editimage,(x,y),3,(255,0,0),-1)
                    mouseX,mouseY = x,y
                    cv2.imshow("image", editimage)
                    cv2.resizeWindow('image', 1920, 1200) 
                    mouseX,mouseY = x,y
            
            editimage = im.copy()
            for i in range(len(b_contours)):
                cv2.drawContours(editimage, [t_Rechteck[i]],i, (255,255,0), 2)
                cv2.drawContours(editimage, t_contours_poly, i, (0,255,180),3)
            
            
            
            Eingabe = "r"
            while True:
                if Eingabe == "r":
                    editimage = im.copy()
                    for i in range(len(b_contours)):
                        #cv2.drawContours(editimage, [t_Rechteck[i]],i, (255,255,0), 2)
                        cv2.drawContours(editimage, t_contours_poly, i, (0,255,180),3)
                    
                    cv2.imshow("image",editimage)
                    cv2.resizeWindow('image', 1920, 1200) 
                    cv2.setMouseCallback('image', set_point)
                    Eingabe = cv2.waitKey(0)
                    Eingabe = chr(Eingabe)
                    cv2.destroyAllWindows()
                    
                    x_2, y_2 = mouseX,mouseY
                else:
                    break









            



            #Create a vector from our origin point to the tail end
            dx = x_2-x
            dy = y_2-y
            steps = 50

            x_step = dx/steps
            y_step = dy/steps

            stepsofx = []
            stepsofy = []
            for b in range(steps+1):
                stepsofx.append(x+x_step*b)
                stepsofy.append(y+y_step*b)

            Minimal_Distance_perPoint = []
            Minimal_Distance_acrossPoints = []
            for a in range(len(stepsofx)-1):
                for b in range(len(t_contours[0])):
                    Ausgangspunkt = (stepsofx[a],stepsofy[a])
                    Minimal_Distance_perPoint.append(sum((Ausgangspunkt-t_contours[i][b][0])**2)**0.5)
                Minimal_Distance_acrossPoints.append(min(Minimal_Distance_perPoint))
            IndexvonPunkt_2 = Minimal_Distance_acrossPoints.index(min(Minimal_Distance_acrossPoints))

            x_3 = round(stepsofx[IndexvonPunkt_2])
            y_3 = round(stepsofy[IndexvonPunkt_2])
            tail_length = ((x_3-x_2)**2+(y_3-y_2)**2)**0.5
            #calculate alpha
            alpha = math.atan((x_2-x_3)/(y_3-y_2))


            #Define a function which measures the width of the tail at a particular section
            def measurewidth(section_number, sections):
                measurement1_center = [stepsofx[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)], stepsofy[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)]]
                for i in range(400):
                    tmpxr1 = round(math.cos(alpha)*i+measurement1_center[0])
                    tmpyr1 = round(math.sin(alpha)*i+measurement1_center[1])
                    if t_imgray[tmpyr1][tmpxr1] == 0:
                        break
                measurement_right=[tmpxr1,tmpyr1]
                for i in range(400):
                    tmpxl1 = round(measurement1_center[0]-math.cos(alpha)*i)
                    tmpyl1 = round(measurement1_center[1]-math.sin(alpha)*i)
                    if t_imgray[tmpyl1][tmpxl1] == 0:
                        break       
                measurement_left=[tmpxl1,tmpyl1]
                    
                

                width_coordinates = [measurement_left, measurement_right]
                width = ((tmpxr1-tmpxl1)**2+(tmpyr1-tmpyl1)**2)**0.5
                return (width_coordinates, width)

            #We iterate the above function for a certain amount of times to measure the tail at multiple distances
            width_coordinates = []
            widths = []
            numberofwidths=20
            for i in range(1,numberofwidths+1):
                tmp1, tmp2 = measurewidth(i,numberofwidths+1)
                width_coordinates.append(tmp1)
                widths.append(round(tmp2, 2))

        


        drawing = np.zeros((t_thresh.shape[0], t_thresh.shape[1], 3), dtype=np.uint8)
        for i in range(len(b_contours)):
            cv2.drawContours(im, [t_Rechteck[i]],i, (255,255,0), 2)
            cv2.drawContours(im, t_contours_poly, i, (0,255,180),3)
            cv2.line(im, (x_2, y_2),(x_3, y_3), (0,0,255), 2)
            cv2.circle(im, (x_2, y_2), 2, (0,0,255), 2)
            cv2.circle(im, (x_3, y_3), 2, (0,0,255), 2)
            cv2.circle(im, (round(Punkt1[0]), round(Punkt1[1])), 2, (0,0,255), 2)
            for b in range(len(widths)):
                cv2.line(im, (width_coordinates[b][0][0], width_coordinates[b][0][1]),(width_coordinates[b][1][0], width_coordinates[b][1][1]), (0,0,255), 2)


        

        tail_measurments = widths
        tail_measurments.insert(0, round(tail_length,2))
        tail_measurments.insert(0, "Beaver tail - Length readjusted")












    return(im, tail_measurments)

def Adjuststartend(images):
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)


    #use the model to predict the image
    with torch.no_grad():
        pred = model(images)
    if len(pred[0]['masks'])<2:
        Norecognition = True
    else: 
        Norecognition = False

    if Norecognition == False:
        #Prepare image
        im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

        #Prepare an empty image where the prediction will be drawn on
        firstmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
        secondmask = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)


        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][1,0].detach().cpu().numpy()
            scr=pred[0]['scores'][1].detach().cpu().numpy()
            if scr>0.8 :
                firstmask[:,:,0][msk>0.5] = 30
                firstmask[:, :, 1][msk > 0.5] = 50
                firstmask[:, :, 2][msk > 0.5] = 60

        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][0,0].detach().cpu().numpy()
            scr=pred[0]['scores'][0].detach().cpu().numpy()
            if scr>0.8 :
                secondmask[:,:,0][msk>0.5] = 40
                secondmask[:, :, 1][msk > 0.5] = 60
                secondmask[:, :, 2][msk > 0.5] = 70

        #########Contours##########
                #The predictions are now mapped to simple binary png's, however as we want to measure the tail, it's easier to work with contours

        #Functions
        def convertcountours(x_contours):
            x_MeanPoint_Y = [None]*len(x_contours)
            x_MeanPoint_X = [None]*len(x_contours)
            x_contours_poly = [None]*len(x_contours)

            for i, c in enumerate(x_contours):
                x_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                list_y = []
                list_x = []
                for a in range(len(x_contours_poly[i])):
                    list_x.append(x_contours_poly[i][a][0][0])
                    list_y.append(x_contours_poly[i][a][0][1])
                x_MeanPoint_Y[i] = np.mean(list_y)
                x_MeanPoint_X[i] = np.mean(list_x)
            return (x_MeanPoint_X, x_MeanPoint_Y, x_contours_poly)

        #####Create Contour images
        #Create contour for tail
        first_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
        first_thresh = cv2.Canny(first_imgray, 50, 100)
        second_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
        second_thresh = cv2.Canny(second_imgray, 50, 100)

        if np.sum(first_thresh == 255) > np.sum(second_thresh == 255):
            b_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            b_imgray = cv2.cvtColor(secondmask, cv2.COLOR_BGR2GRAY)
            b_thresh = cv2.Canny(b_imgray, 50, 100)
            t_imgray = cv2.cvtColor(firstmask, cv2.COLOR_BGR2GRAY)
            t_thresh = cv2.Canny(t_imgray, 50, 100)
            t_contours, t_hierarchy  = cv2.findContours(first_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b_contours, b_hierarchy  = cv2.findContours(second_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        t_Rechteck = [None]*len(t_contours)
        b_Rechteck = [None]*len(b_contours)


        #This function (established above) creates a polygon and the centroid coordinates
        t_MeanPoint_X, t_MeanPoint_Y, t_contours_poly = convertcountours(t_contours) 
        b_MeanPoint_X, b_MeanPoint_Y, b_contours_poly = convertcountours(b_contours) 


        #This part finds the point at the start of the beaver tail
        for i, c in enumerate(t_contours):
            
            

            #Find the point of the tail that is furthest away from the origin point (middle of bounding box)
            def set_point(event,x,y,flags,param):
                global mouseX,mouseY, mouseX2, mouseY2
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(editimage,(x,y),3,(255,0,0),-1)
                    mouseX, mouseY = x,y
                    cv2.imshow("image", editimage)
                    cv2.resizeWindow('image', 1920, 1200) 

                if event == cv2.EVENT_LBUTTONUP:
                    cv2.circle(editimage,(x,y),3,(255,0,0),-1)
                    mouseX2, mouseY2 = x,y
                    cv2.line(editimage, (mouseX2, mouseY2),(mouseX, mouseY), (0,0,255), 2)
                    cv2.imshow("image", editimage)
                    cv2.resizeWindow('image', 1920, 1200) 


            Eingabe = "r"
            while True:
                if Eingabe == "r":
                    editimage = im.copy()
                    for i in range(len(b_contours)):
                        #cv2.drawContours(editimage, [t_Rechteck[i]],i, (255,255,0), 2)
                        cv2.drawContours(editimage, t_contours_poly, i, (0,255,180),3)
                    
                    cv2.imshow("image",editimage)
                    cv2.resizeWindow('image', 1920, 1200) 
                    cv2.setMouseCallback('image', set_point)
                    Eingabe = cv2.waitKey(0)
                    Eingabe = chr(Eingabe)
                    cv2.destroyAllWindows()
                    
                    x = mouseX
                    y = mouseY
                    x_2 = mouseX2
                    y_2 = mouseY2
                else:
                    break
            
            


            #Create a vector from our origin point to the tail end
            dx = x_2-x
            dy = y_2-y
            steps = 50

            x_step = dx/steps
            y_step = dy/steps

            stepsofx = []
            stepsofy = []
            for b in range(steps+1):
                stepsofx.append(x+x_step*b)
                stepsofy.append(y+y_step*b)

            Minimal_Distance_perPoint = []
            Minimal_Distance_acrossPoints = []
            for a in range(len(stepsofx)-1):
                for b in range(len(t_contours[0])):
                    Ausgangspunkt = (stepsofx[a],stepsofy[a])
                    Minimal_Distance_perPoint.append(sum((Ausgangspunkt-t_contours[i][b][0])**2)**0.5)
                Minimal_Distance_acrossPoints.append(min(Minimal_Distance_perPoint))
            IndexvonPunkt_2 = Minimal_Distance_acrossPoints.index(min(Minimal_Distance_acrossPoints))
            

            x_3 = x
            y_3 = y
            #x_3 = round(stepsofx[IndexvonPunkt_2])
            #y_3 = round(stepsofy[IndexvonPunkt_2])
            tail_length = ((x_3-x_2)**2+(y_3-y_2)**2)**0.5
            #calculate alpha
            alpha = math.atan((x_2-x_3)/(y_3-y_2))


            #Define a function which measures the width of the tail at a particular section
            def measurewidth(section_number, sections):
                measurement1_center = [stepsofx[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)], stepsofy[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)]]
                for i in range(400):
                    tmpxr1 = round(math.cos(alpha)*i+measurement1_center[0])
                    tmpyr1 = round(math.sin(alpha)*i+measurement1_center[1])
                    if t_imgray[tmpyr1][tmpxr1] == 0:
                        break
                measurement_right=[tmpxr1,tmpyr1]
                for i in range(400):
                    tmpxl1 = round(measurement1_center[0]-math.cos(alpha)*i)
                    tmpyl1 = round(measurement1_center[1]-math.sin(alpha)*i)
                    if t_imgray[tmpyl1][tmpxl1] == 0:
                        break       
                measurement_left=[tmpxl1,tmpyl1]
                    
                

                width_coordinates = [measurement_left, measurement_right]
                width = ((tmpxr1-tmpxl1)**2+(tmpyr1-tmpyl1)**2)**0.5
                return (width_coordinates, width)

            #We iterate the above function for a certain amount of times to measure the tail at multiple distances
            width_coordinates = []
            widths = []

            numberofwidths=20
            for i in range(1,numberofwidths+1):
                tmp1, tmp2 = measurewidth(i,numberofwidths+1)
                width_coordinates.append(tmp1)
                widths.append(round(tmp2, 2))

        


        drawing = np.zeros((t_thresh.shape[0], t_thresh.shape[1], 3), dtype=np.uint8)
        for i in range(len(b_contours)):
            cv2.drawContours(im, t_contours_poly, i, (0,255,180),3)
            cv2.line(im, (x_2, y_2),(x_3, y_3), (0,0,255), 2)
            cv2.circle(im, (x_2, y_2), 2, (0,0,255), 2)
            cv2.circle(im, (x_3, y_3), 2, (0,0,255), 2)
            for b in range(len(widths)):
                cv2.line(im, (width_coordinates[b][0][0], width_coordinates[b][0][1]),(width_coordinates[b][1][0], width_coordinates[b][1][1]), (0,0,255), 2)


        

        tail_measurments = widths
        tail_measurments.insert(0, round(tail_length,2))
        tail_measurments.insert(0, "Beaver tail - Start and Endpoint readjusted")












    return(im, tail_measurments)

def drawtailmanually(images):
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)

    im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)




    
    manual_points = []
    
    #Manuelles Zeichnen der Biberkelle
    def set_points(event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_points)<1:
                manual_points.append([x,y])
                cv2.circle(editimage,(x,y),3,(255,0,0),-1)  
                cv2.imshow("Test", editimage)  
                cv2.resizeWindow('Test', 1920, 1200)             
            elif len(manual_points)==1:
                cv2.circle(editimage,manual_points[-1],3,(255,255,0),-1)
                manual_points.append([x,y])
                cv2.circle(editimage,(x,y),3,(255,0,0),-1)  
                cv2.line(editimage, manual_points[0], manual_points[1], (0, 0, 127), 2)
                cv2.imshow("Test", editimage)
                cv2.resizeWindow('Test', 1920, 1200)  
            else:
                cv2.circle(editimage,manual_points[-1],3,(255,255,0),-1)
                manual_points.append([x,y])
                cv2.circle(editimage,(x,y),3,(255,0,0),-1)  
                pts= np.array(manual_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(editimage, [pts], False, (0, 0, 127), 2)
                cv2.imshow("Test", editimage)
                cv2.resizeWindow('Test', 1920, 1200)  

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(manual_points)<1:
                manual_points.append([x,y])
                cv2.circle(editimage,(x,y),3,(255,0,0),-1)  
                cv2.imshow("Test", editimage)
                cv2.resizeWindow('Test', 1920, 1200)               
            elif len(manual_points)==1:
                manual_points.append([x,y])
                cv2.circle(editimage,(x,y),3,(255,0,0),-1)  
                cv2.line(editimage, manual_points[0], manual_points[1], (0, 0, 127), 2)
                cv2.imshow("Test", editimage)
                cv2.resizeWindow('Test', 1920, 1200)  
            else:
                cv2.circle(editimage,manual_points[-1],3,(255,255,0),-1)
                cv2.circle(editimage,(x,y),3,(255,255,0),-1)
                manual_points.append([x,y])
                pts= np.array(manual_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(editimage, [pts], True, (0, 0, 127), 2)
                cv2.imshow("Test", editimage)
                cv2.resizeWindow('Test', 1920, 1200)  
                return


    Eingabe = "r"
    while True:
        if Eingabe == "r":
            editimage = im.copy()
            cv2.namedWindow(winname = "Test")
                        
            cv2.imshow("Test",editimage)
            cv2.setMouseCallback('Test', set_points)
            cv2.resizeWindow('Test', 1920, 1200)  
            Eingabe = cv2.waitKey(0)
            Eingabe = chr(Eingabe)
            cv2.destroyAllWindows()
            
            
        else:
            cv2.destroyAllWindows()
            break    


    Tail = np.zeros((imageSize[1], imageSize[0], 3), np.uint8)
    pts= np.array(manual_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(Tail, [pts], (0, 0, 127))
    tail_imgray = cv2.cvtColor(Tail, cv2.COLOR_BGR2GRAY)
    tail_thresh = cv2.Canny(tail_imgray, 50, 100)
    tail_contours, t_hierarchy  = cv2.findContours(tail_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  


    def set_point(event,x,y,flags,param):
        global mouseX,mouseY, mouseX2, mouseY2
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(editimage,(x,y),3,(255,0,0),-1)
            mouseX, mouseY = x,y
            cv2.imshow("image", editimage)
            cv2.resizeWindow('image', 1920, 1200)  

        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(editimage,(x,y),3,(255,0,0),-1)
            mouseX2, mouseY2 = x,y
            cv2.line(editimage, (mouseX2, mouseY2),(mouseX, mouseY), (0,0,255), 2)
            cv2.imshow("image", editimage)
            cv2.resizeWindow('image', 1920, 1200)  


    Eingabe = "r"
    while True:
        if Eingabe == "r":
            editimage = im.copy()
            cv2.polylines(editimage, [pts],True, (0,255,180),3)
            
            cv2.imshow("image",editimage)
            cv2.resizeWindow('image', 1920, 1200)  
            cv2.setMouseCallback('image', set_point)
            Eingabe = cv2.waitKey(0)
            Eingabe = chr(Eingabe)
            cv2.destroyAllWindows()
            
            x = mouseX
            y = mouseY
            x_2 = mouseX2
            y_2 = mouseY2
        else:
            break

    #Create a vector from our origin point to the tail end
    dx = x_2-x
    dy = y_2-y
    steps = 50

    x_step = dx/steps
    y_step = dy/steps

    stepsofx = []
    stepsofy = []
    for b in range(steps+1):
        stepsofx.append(x+x_step*b)
        stepsofy.append(y+y_step*b)

    Minimal_Distance_perPoint = []
    Minimal_Distance_acrossPoints = []
    for a in range(len(stepsofx)-1):
        for b in range(len(tail_contours[0])):
            Ausgangspunkt = (stepsofx[a],stepsofy[a])
            Minimal_Distance_perPoint.append(sum((Ausgangspunkt-tail_contours[0][b][0])**2)**0.5)
        Minimal_Distance_acrossPoints.append(min(Minimal_Distance_perPoint))
    IndexvonPunkt_2 = Minimal_Distance_acrossPoints.index(min(Minimal_Distance_acrossPoints))

    x_3 = x
    y_3 = y
    tail_length = ((x_3-x_2)**2+(y_3-y_2)**2)**0.5
    #calculate alpha
    alpha = math.atan((x_2-x_3)/(y_3-y_2))


    #Define a function which measures the width of the tail at a particular section
    def measurewidth(section_number, sections):
        measurement1_center = [stepsofx[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)], stepsofy[round(IndexvonPunkt_2 + section_number*(steps-IndexvonPunkt_2)/sections)]]
        for i in range(400):
            tmpxr1 = round(math.cos(alpha)*i+measurement1_center[0])
            tmpyr1 = round(math.sin(alpha)*i+measurement1_center[1])
            if tail_imgray[tmpyr1][tmpxr1] == 0:
                break
        measurement_right=[tmpxr1,tmpyr1]
        for i in range(400):
            tmpxl1 = round(measurement1_center[0]-math.cos(alpha)*i)
            tmpyl1 = round(measurement1_center[1]-math.sin(alpha)*i)
            if tail_imgray[tmpyl1][tmpxl1] == 0:
                break       
        measurement_left=[tmpxl1,tmpyl1]
            
        

        width_coordinates = [measurement_left, measurement_right]
        width = ((tmpxr1-tmpxl1)**2+(tmpyr1-tmpyl1)**2)**0.5
        return (width_coordinates, width)

    #We iterate the above function for a certain amount of times to measure the tail at multiple distances
    width_coordinates = []
    widths = []
    numberofwidths=20
    for i in range(1,numberofwidths+1):
        tmp1, tmp2 = measurewidth(i,numberofwidths+1)
        width_coordinates.append(tmp1)
        widths.append(round(tmp2, 2))


    drawing = np.zeros((tail_thresh.shape[0], tail_thresh.shape[1], 3), dtype=np.uint8)


    cv2.polylines(im, [pts],True, (0,255,180),3)
    cv2.line(im, (x_2, y_2),(x_3, y_3), (0,0,255), 2)
    cv2.circle(im, (x_2, y_2), 2, (0,0,255), 2)
    cv2.circle(im, (x_3, y_3), 2, (0,0,255), 2)

    for b in range(len(widths)):
        cv2.line(im, (width_coordinates[b][0][0], width_coordinates[b][0][1]),(width_coordinates[b][1][0], width_coordinates[b][1][1]), (0,0,255), 2)
    

    tail_measurments = widths
    tail_measurments.insert(0, round(tail_length,2))
    tail_measurments.insert(0, "Beaver tail - Manually drawn")
  
    return(im, tail_measurments)

def BiberaufBild(): 
    res = mb.askquestion('Biber auf dem Bild',  
                         'Ist auf diesem Bild ein Biber zu sehen') 
      
    if res == 'Ja' :
        return True 
          
    else : 
        return False 

columnnames =["ImageName", "Species" ,"length"]
numberofwidths=20
for i in range(1,numberofwidths+1):
    columnnames.append("W"+str(i))
with open("Measurements.csv", 'w') as csvfile:  
                    csvwriter = csv.writer(csvfile)  # creating a csv writer object
                    csvwriter.writerow(columnnames)  # writing the fields 

for run in range(len(imgs)):
    images=cv2.imread(testDir+imgs[run])
    try:
        Norecognition, im, tail_measurments = predict_image(images)
    except:
        Norecognition = True
        im, tail_measurments = drawtailmanually(images) ##########################redundant!!!
    
    if Norecognition == False:
        tail_measurments.insert(0, str(imgs[run]))
        cv2.namedWindow('Contours', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Contours', im)
        cv2.resizeWindow('Contours', 1024, 576)  
        cv2.moveWindow("Contours", 896, 0)    
        Eingabe = cv2.waitKey(0)
        print(chr(Eingabe))
        if chr(Eingabe) == "l":
            print("Falsche Länge")
            cv2.destroyAllWindows()
            im, tail_measurments = Adjustlength(images)
            tail_measurments.insert(0, str(imgs[run]))
            cv2.namedWindow('Contours', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Contours', im)
            cv2.resizeWindow('Contours', 1024, 576)  
            cv2.moveWindow("Contours", 896, 0)  
            Eingabe = cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(Eingabe) == "m":
            cv2.destroyAllWindows()
            im, tail_measurments = drawtailmanually(images)
            tail_measurments.insert(0, str(imgs[run]))
            cv2.namedWindow('Contours', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Contours', im)
            cv2.resizeWindow('Contours', 1024, 576)  
            cv2.moveWindow("Contours", 896, 0)   
            Eingabe = cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif chr(Eingabe) == "c":
            cv2.destroyAllWindows()
            im, tail_measurments = Adjuststartend(images)
            tail_measurments.insert(0, str(imgs[run]))
            cv2.namedWindow('Contours', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Contours', im)
            cv2.resizeWindow('Contours', 1024, 576)
            cv2.moveWindow("Contours", 896, 0)     
            Eingabe = cv2.waitKey(0)
            cv2.destroyAllWindows() 
        elif chr(Eingabe) == "n":
            tail_measurments = [0] * (numberofwidths+1)
            tail_measurments.insert(0, "Beaver tail - not usable")
            tail_measurments.insert(0, str(imgs[run]))
            cv2.destroyAllWindows()         
        else:
            cv2.destroyAllWindows()


        with open("Measurements.csv", 'a') as csvfile:  
                csvwriter = csv.writer(csvfile) # creating a csv writer object  
                csvwriter.writerow(tail_measurments) # writing the data rows

        
        cv2.imwrite("Fotos/Results/True-Result-1-"+str(imgs[run])+".JPG", cv2.addWeighted(im, 0.5, im, 0.5, 1))
        cv2.waitKey()
        
    else:
        images_tmp = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
        images_tmp = torch.as_tensor(images_tmp, dtype=torch.float32).unsqueeze(0)
        images_tmp=images_tmp.swapaxes(1, 3).swapaxes(2, 3)
        images_tmp = list(image.to(device) for image in images_tmp)
        im_tmp= images_tmp[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
        

        cv2.namedWindow('Biber auf dem Bild?', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Biber auf dem Bild?', im_tmp)
        cv2.resizeWindow('Biber auf dem Bild?', 1024, 576)
        BiberonImage = mb.askyesno("Biber auf Foto", "Ist auf diesem Bild eine Biberkelle zu erkennen?")  
        #root = tk.Tk() 
        #canvas = tk.Canvas(root, width = 200, height = 200)   
        #canvas.pack()

        #root.mainloop() 
        

        if BiberonImage:
            cv2.waitKey(1)
            cv2.destroyAllWindows() 
            im, tail_measurments = drawtailmanually(images)
            tail_measurments.insert(0, str(imgs[run]))
            cv2.namedWindow('Contours', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Contours', im)
            cv2.resizeWindow('Contours', 1024, 576)
            #cv2.imwrite("Fotos/Results/True-Result-1-"+str(imgs[run])+".JPG", cv2.addWeighted(im, 0.5, im, 0.5, 1))   
            Eingabe = cv2.waitKey(0)
            cv2.destroyAllWindows()    
            #print(str(imgs[run])+ " nicht erkannt")
     
            
            with open("Measurements.csv", 'a') as csvfile:  
                csvwriter = csv.writer(csvfile) # creating a csv writer object  
                csvwriter.writerow(tail_measurments) # writing the data rows

        else:
            Foto = simpledialog.askstring("Keine Biberkelle", "Was ist auf dem Bild zu sehen")
            tail_measurments = [0] * (numberofwidths+1)
            tail_measurments.insert(0, Foto)
            tail_measurments.insert(0, str(imgs[run]))
            with open("Measurements.csv", 'a') as csvfile:  
                csvwriter = csv.writer(csvfile) # creating a csv writer object  
                csvwriter.writerow(tail_measurments) # writing the data rows
