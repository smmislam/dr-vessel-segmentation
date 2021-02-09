# Semantic Segmentation of Retinal Blood Vessel via Multi-scale Convolutional Neural Network

This is the source code for segmenting retinal blood vessels from raw Fundus images, which can be used to increase the accuracy of Diabetic Retinopathy diagnosis. This work has been published at **Proceedings of International Joint Conference on Computational Intelligence, 2019** as part of the **Algorithms for Intelligent Systems** book series.


# Literature Source
This literature can be accessed from [Springer](https://link.springer.com/chapter/10.1007/978-981-15-3607-6_18), [Pre-print](https://www.researchgate.net/publication/336890473_Semantic_Segmentation_of_Retinal_Blood_Vessel_via_Multi-_Scale_Convolutional_Neural_Network).


# Citation
If you are using this source code or the obtained results of this work, please cite the following publication.

**Title:** Semantic Segmentation of Retinal Blood Vessel via Multi-scale Convolutional Neural Network\
**Author:** SM Mazharul Islam\
**Abstract:** Segmentation of retinal blood vessel is one of the key stages to automate diagnosis and detection of diseases such as diabetic retinopathy (DR), glaucoma, hypertension, age-related macular degeneration (AMD), etc. In this paper, semantic segmentation method is presented to extract the blood vessels from retinal fundus images using a novel multi-scale convolutional neural network. Features are extracted from raw fundus images via Tsallis entropy, adaptive histogram equalization and standard deviation. Then, a multi-scale convolutional neural network is applied on these features to semantically segment the blood vessels. The algorithm is trained only with the training set images of DRIVE database and then tested on both test-set images of DRIVE database and healthy patient images of HRF database to evaluate its performance. During the whole evaluation process, the network and its weights have remained fixed. The results show that the proposed method achieves an accuracy, sensitivity, specificity of 95.38%, 79.23%, 97.73%, respectively, on DRIVE database and 94.03%, 77.52%, 96.07%, respectively, on HRF database.


## BibTex
@inproceedings{islam2019semantic,\
  title={Semantic Segmentation of Retinal Blood Vessel via Multi-scale Convolutional Neural Network},\
  author={Islam, SM Mazharul},\
  booktitle={International Joint Conference on Computational Intelligence},\
  pages={231--241},\
  year={2019},\
  organization={Springer}\
}
