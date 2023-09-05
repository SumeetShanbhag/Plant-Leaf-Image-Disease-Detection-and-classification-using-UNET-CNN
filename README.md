# Introduction 
This project presents a plant image classification scheme that uses a combination of Unet-based image segmentation and a convolutional neural
network (CNN) architecture for the actual classification. The first step of the proposed approach is to segment the plant leaves from the
background using a modified Unet architecture,which is a popular deep-learning model for image segmentation. The segmented leaves are
then preprocessed and fed into a CNN architecture for the actual classification. The CNN architecture consists of multiple convolutional layers, 
followed by pooling and fully connected layers, which enable the model to learn the complex features necessary for accurate classification. 
To evaluate the proposed approach, experiments were conducted on a publicly available Plant village dataset. The results show that
the proposed approach achieves high accuracy in classifying different plant species. The dataset used for the project can be downloaded from [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/download?datasetVersionNumber=2)

The percentage of crop yield lost to plant diseases each year
varies depending on the crop, the region, and the specific
plant disease. According to the Food and Agriculture Or-
ganization (FAO) of the United Nations, plant diseases and
pests are responsible for the loss of up to 40% of global
food crops each year. In some cases, the percentage of
crop yield lost to plant diseases can be much higher. For
example, some estimates suggest that up to 80% of banana
crops worldwide are at risk from the fungal disease known
as Panama disease, also called Fusarium wilt, which can
cause complete crop loss. It’s worth noting that losses due
to plant diseases can also have significant economic im-
pacts beyond just the loss of crops, including increased
costs for control measures and reduced market access for
farmers. Early detection of plant diseases remains challenging due to lacking lab infrastructure and expertise.

# Project Explanation 
[![YouTube Video Link](https://img.youtube.com/vi/eyfjTdy0c60/maxresdefault.jpg)](https://youtu.be/eyfjTdy0c60)

# Proposed Method
## Data Description
This project uses the Plant village dataset found in (Gomaa, 2023).The Plant Village dataset is a collection of over
54,000 high-quality images of 14 different crop species,
including tomato, potato, apple, and grape. Each image
is associated with one of 38 different classes, representing
various plant diseases or healthy conditions.

## Computer Vision
The Python OpenCV library provides several functions for
image thresholding, a process that converts grayscale or
color images into binary images. In this process, each pixel
in the image is compared to a threshold value and is assigned a binary value (0 or 255) based on whether it is
above or below the threshold value. The OpenCV library
is used in this project to convert the RGB images into segmented images with a black background, but it failed to
give decent results for all the images, as shown in Fig.3.
A qualitative study of the results of this computer vision
scheme showed an efficient performance of this strategy on
the tomato healthy leaves images. Consequently, the obtained tomato healthy binary images are used as the ground
truth for training the UNET (Ronneberger et al., 2015) image segmentation strategy.

![Cv images](https://github.com/SumeetShanbhag/Plant-Leaf-Image-Disease-Detection-and-classification-using-UNET-CNN/blob/main/images/OpenCV.png)

##  U-NET Image Segmentation
The U-net is an advanced deep learning architecture designed for image segmentation, focusing on biomedical image analysis (Ronneberger et al., 2015). Its name is derived
from its U-shaped network architecture, distinguishing it
from conventional CNN models. In contrast to standard
CNN models, U-net employs convolutional layers to upsample or combine feature maps into a complete image.
This project introduces the conventional U-Net architecture utilized in (Vitali, 2020) using the tomato healthy as a
baseline training subset. In addition, to reduce the number
of channels of each image, OpenCV is newly used to obtain grayscale images for faster training performance. The
architecture utilized in this work can be seen in the image below.

![CNN Architecture](https://github.com/SumeetShanbhag/Plant-Leaf-Image-Disease-Detection-and-classification-using-UNET-CNN/blob/main/images/OUR_CNNarch.png)
