## Introduction 
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
## Project Explanation 
[![YouTube Video Link](https://img.youtube.com/vi/eyfjTdy0c60/maxresdefault.jpg)](https://youtu.be/eyfjTdy0c60)

