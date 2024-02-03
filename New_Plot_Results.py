from matplotlib import pyplot as plt
from prettytable import PrettyTable
import cv2 as cv
import numpy as np

def plot_Result():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU', 'F1-Score']
    Graph_Term = [0, 1, 2, 3, 4, 5]
    Classifier = ['TERMS', 'U-Net', 'UNet++', 'ResUNet', 'SEResUNet', 'Attention UNet', 'DA Net', 'TransUNet',
                  'Caries SegNet']

    for n in range(eval.shape[0]):
        value = eval[n]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset-' + str(n + 1) + '-Method Comparison ',
              '--------------------------------------------------')
        print(Table)

    for n in range(eval.shape[0]):
        val = eval[n]
        Graph = val

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(len(Terms[:]))

        fig1 = plt.bar(X + 0.00, Graph[0, :], width=0.08, color='#FF9912', label="U-Net")
        fig2 = plt.bar(X + 0.12, Graph[1, :], width=0.08, color='#00EEEE', label="UNet++")
        fig3 = plt.bar(X + 0.25, Graph[2, :], width=0.1, color='#D15FEE', label="ResUNet")
        fig4 = plt.bar(X + 0.37, Graph[3, :], width=0.1, color='#FFAEB9', label="SEResUNet")
        fig5 = plt.bar(X + 0.49, Graph[4, :], width=0.1, color='k', label="Attention UNet")
        fig5 = plt.bar(X + 0.61, Graph[5, :], width=0.1, color='y', label="DA Net")
        fig5 = plt.bar(X + 0.73, Graph[6, :], width=0.1, color='g', label="TransUNet")
        fig5 = plt.bar(X + 0.85, Graph[7, :], width=0.1, color='b', label="Caries SegNet")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                   ncol=3, fancybox=True, shadow=True)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        plt.xticks(X + 0.10, ('Accuracy', 'Precision', 'Recall', 'Dice', 'IoU', 'F1-Score'))
        plt.savefig("./Results/1st_5Matrix_alg_Bar_%s.png" % (n + 1))
        plt.show()



no_of_Dataset = 2


def Image_Result():
    IMAGE = [4]
    for n in range(no_of_Dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)
        UNet = np.load('UNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        UNetPlusPlus_Images = np.load('UNetPlusPlus_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        ResUNet_Images = np.load('ResUNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        SEResUNet_Images = np.load('SEResUNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        AttentionUNet_Images = np.load('AttentionUNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        DANet_Images = np.load('DANet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        TransUNet_Images = np.load('TransUNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Caries_SegNet_Images = np.load('Caries_SegNet_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(IMAGE)):  # Dataset-1-Image-2-1
            image = Images[IMAGE[i]]
            Ground_Truth = GT[IMAGE[i]]
            image1 = UNetPlusPlus_Images[IMAGE[i]]
            image2 = ResUNet_Images[IMAGE[i]]
            image3 = SEResUNet_Images[IMAGE[i]]
            image4 = AttentionUNet_Images[IMAGE[i]]
            image5 = DANet_Images[IMAGE[i]]
            image6 = TransUNet_Images[IMAGE[i]]
            image7 = Caries_SegNet_Images[IMAGE[i]]
            image8 = UNet[IMAGE[i]]
            cv.imshow("Original Image " + str(i + 1), image)
            # cv.waitKey(0)
            cv.imshow("Ground Truth " + str(i + 1), Ground_Truth)
            # cv.waitKey(0)
            cv.imshow("UNet" + str(i + 1), image8)
            # cv.waitKey(0)
            cv.imshow("UNetPlusPlus" + str(i + 1), image1)
            # cv.waitKey(0)
            cv.imshow("ResUNet" + str(i + 1), image2)
            # cv.waitKey(0)
            cv.imshow("SEResUNet" + str(i + 1), image3)
            # cv.waitKey(0)
            cv.imshow("AttentionUNet" + str(i + 1), image4)
            # cv.waitKey(0)
            cv.imshow("DANet" + str(i + 1), image5)
            # cv.waitKey(0)
            cv.imshow("TransUNet" + str(i + 1), image6)
            # cv.waitKey(0)
            cv.imshow("Caries_SegNet" + str(i + 1), image7)
            cv.waitKey(0)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'orig-' + str(i + 1) + '.png', image)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'GroundTruth-' + str(i + 1) + '.png',
                       Ground_Truth)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'UNetPlusPlus-' + str(i + 1) + '.png',
                       image1)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'ResUNet-' + str(i + 1) + '.png', image2)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'SEResUNet-' + str(i + 1) + '.png',
                       image3)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'AttentionUNet-' + str(i + 1) + '.png',
                       image4)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'DANet-' + str(i + 1) + '.png', image5)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'TransUNet-' + str(i + 1) + '.png',
                       image6)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'Caries_SegNet-' + str(i + 1) + '.png',
                       image7)
            cv.imwrite('./Results/Image_Results/' + 'Dataset-' + str(n + 1) + 'UNet-' + str(i + 1) + '.png', image8)


# plot_Result()
# Image_Result()
