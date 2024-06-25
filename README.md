# APENet-fss
This is the implementation of our paper APENet: Task-Aware Adaptation Prototype Evolution Network for Few-shot Semantic Segmentation that has been submitted to Pattern Recognition (PR).
# Description
Few-shot semantic segmentation (FSS) is a challenging computer vision task that aims to predict the masks of unseen classes with only a few labeled samples. Although recent advances have been achieved in FSS based on prototype-based metric approaches, existing methods still face two main challenges. First, previous methods primarily focus on designing a complex interaction mechanism between inter-branch features, neglecting the specific requirements of the query branch. Second, the inappropriate use of query features is very likely to cause semantic ambiguity problems, which hinders the segmentation of unseen objects. To alleviate these problems, we propose a novel task-aware Adaptation Prototype Evolution Network (APENet). Specifically, we design a Feature Recombination Module (FAM), which utilizes the ground truth masks of support images to separate and recombine the features encoded before and after the backbone network. Subsequently, we use the Adaptation Prototype Evolution Module (APEM) to perform a reverse segmentation on the original support image, and the support prototypes are separated into the main prototype and the auxiliary prototype according to the ground truth mask. In addition, the Feature Disentanglement Module (FDM) is introduced to disentangle the whole query feature using both the text embedding provided by CLIP model and provisionally predicted query mask. Finally, the Feature Alignment Module (FAM) is designed to promote the feature interaction and alignment for different branches. Extensive experiments on PASCAL-$5^{i}$ and COCO-$20^{i}$ datasets validate the effectiveness of our method. In particular, the APENet is comparable to current classical FSS methods on cross-domain and 2-way segmentation tasks, illustrating the high generalizability. 

![Uploading fig2.png…]()

# Please download the following datasets: 
PASCAL-5i is based on the PASCAL VOC 2012 and SBD where the val images should be excluded from the list of training samples.

Images are available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

annotations: https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing

This work is built on:

OSLSM: https://github.com/lzzcd001/OSLSM

PANet: https://github.com/kaixin96/PANet

Many thanks to their greak work!
