import numpy as np
import os
import cv2 as cv
from New_Plot_Results import plot_Result, Image_Result
from Model_AttentionUNet import Model_AttentionUNet
from Model_Caries_SegNet import Model_Caries_SegNet
# from Model_DANet import Model_DANet
# from Model_ResUNet import Model_ResUNet
# from Model_SEResUNet import Model_SEResUnet
# from Model_TransUNet import Model_TransUNet
# from Model_UNetPlusPlus import Model_UNetPlusPlus
# from UNET_Model import UNET_Model


def Read_Image(filename):  # function for read_image
    image = cv.imread(filename)  # to read the image file
    image = np.uint8(image)  # to change the unsigned int bit 8 value
    image = cv.resize(image, (512, 512))  # for resizing image / scaling
    return image


# Read Dataset 1
an = 0
if an == 1:
    Img_fold = os.listdir('./Dataset/caries_dataset/')
    for i in range(len(Img_fold)):
        fold = os.listdir('./Dataset/caries_dataset/' + Img_fold[i])
        Images = []
        GT = []
        for j in range(len(fold)):
            img = Read_Image('./Dataset/caries_dataset/' + Img_fold[0] + '/' + fold[j])
            gt = Read_Image('./Dataset/caries_dataset/' + Img_fold[1] + '/' + fold[j])
            img1 = cv.resize(img, (256, 256))
            gt1 = cv.resize(gt, (256, 256))
            Images.append(img1)
            GT.append(gt1)
        np.save('Images_1.npy', Images)
        np.save('GT_1.npy', GT)


# Read Dataset2
an = 0
if an == 1:
    Img_fold = os.listdir('./Dataset/Tufts Dental Database/Radiographs/')
    Img_fold1 = os.listdir('./Dataset/Tufts Dental Database/Segmentation/teeth_mask/')
    Images = []
    GT = []
    for i in range(len(Img_fold)):
        img = Read_Image('./Dataset/Tufts Dental Database/Radiographs/' + Img_fold[i])
        gt = Read_Image('./Dataset/Tufts Dental Database//Segmentation/teeth_mask/' + Img_fold1[i])
        img1 = cv.resize(img, (256, 256))
        gt1 = cv.resize(gt, (256, 256))
        Images.append(img1)
        GT.append(gt1)
    np.save('Images_2.npy', Images)
    np.save('GT_2.npy', GT)

no_of_dataset = 2

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n+1) +'.npy', allow_pickle=True)
        Mask = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Eval = np.zeros((8, 6))
        img_height, img_width = 256, 256
        seg_images1, Eval[0, :] = UNET_Model(Images, Mask, img_height, img_width)
        seg_images2, Eval[1, :] = Model_UNetPlusPlus(Images, Mask, img_height, img_width)
        seg_images3, Eval[2, :] = Model_ResUNet(Images, Mask, img_height, img_width)
        seg_images4, Eval[3, :] = Model_SEResUnet(Images, Mask, img_height, img_width)
        seg_images5, Eval[4, :] = Model_AttentionUNet(Images, Mask, img_height, img_width)
        seg_images6, Eval[5, :] = Model_DANet(Images, Mask, img_height, img_width)
        seg_images7, Eval[6, :] = Model_TransUNet(Images, Mask, img_height, img_width)
        seg_images8, Eval[7, :] = Model_Caries_SegNet(Images, Mask, img_height, img_width)
        np.save('UNet_Images_' + str(n+1) +'.npy', seg_images1)
        np.save('UNetPlusPlus_Images_' + str(n+1) +'.npy', seg_images2)
        np.save('ResUNet_Images_' + str(n+1) +'.npy', seg_images3)
        np.save('SEResUNet_Images_' + str(n+1) +'.npy', seg_images4)
        np.save('AttentionUNet_Images_' + str(n+1) +'.npy', seg_images5)
        np.save('DANet_Images_' + str(n+1) +'.npy', seg_images6)
        np.save('TransUNet_Images_' + str(n+1) +'.npy', seg_images7)
        np.save('Caries_SegNet_Images_' + str(n+1) +'.npy', seg_images8)
        np.save('Eval_all.npy', Eval)

plot_Result()
Image_Result()