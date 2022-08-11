
# Artículos relevantes


## A Multi-Stream Sequence Learning Framework for Human Interaction Recognition (2022)
### SIN ACCESO AL ARTICULO

Proponen usar una red que localiza las articulaciones humanas usando estimación de pose + una red que extare características discriminativas espaciotemporales para luego pasarlas a una capa que clasifica las interacciones humanas.

U. Haroon et al., "A Multi-Stream Sequence Learning Framework for Human Interaction Recognition," in IEEE Transactions on Human-Machine Systems, vol. 52, no. 3, pp. 435-444, June 2022, doi: 10.1109/THMS.2021.3138708.

https://ieeexplore.ieee.org/abstract/document/9695398

Human interaction recognition (HIR) is challenging due to multiple humans’ involvement and their mutual interaction in a single frame, generated from their movements. Mainstream literature is based on three-dimensional (3-D) convolutional neural networks (CNNs), processing only visual frames, where human joints data play a vital role in accurate interaction recognition. Therefore, this article proposes a multistream network for HIR that intelligently learns from skeletons’ key points and spatiotemporal visual representations. The first stream localises the joints of the human body using a pose estimation model and transmits them to a 1-D CNN and bidirectional long short-term memory to efficiently extract the features of the dynamic movements of each human skeleton. The second stream feeds the series of visual frames to a 3-D convolutional neural network to extract the discriminative spatiotemporal features. Finally, the outputs of both streams are integrated via fully connected layers that precisely classify the ongoing interactions between humans. To validate the performance of the proposed network, we conducted a comprehensive set of experiments on two benchmark datasets, UT-interaction and TV human interaction, and found 1.15% and 10.0% improvement in the accuracy.


## Online Human Interaction Detection and Recognition With Multiple Cameras (2017)
### SIN ACCESO AL ARTICULO

S. Motiian, F. Siyahjani, R. Almohsen and G. Doretto, "Online Human Interaction Detection and Recognition With Multiple Cameras," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 27, no. 3, pp. 649-663, March 2017, doi: 10.1109/TCSVT.2016.2606998.

https://ieeexplore.ieee.org/abstract/document/7563312

We address the problem of detecting and recognizing online the occurrence of human interactions as seen by a network of multiple cameras. We represent interactions by forming temporal trajectories, coupling together the body motion of each individual and their proximity relationships with others, and also sound whenever available. Such trajectories are modeled with kernel state-space (KSS) models. Their advantage is being suitable for the online interaction detection, recognition, and also for fusing information from multiple cameras, while enabling a fast implementation based on online recursive updates. For recognition, in order to compare interaction trajectories in the space of KSS models, we design so-called pairwise kernels with a special symmetry. For detection, we exploit the geometry of linear operators in Hilbert space, and extend to KSS models the concept of parity space, originally defined for linear models. For fusion, we combine KSS models with kernel construction and multiview learning techniques. We extensively evaluate the approach on four single view publicly available data sets, and we also introduce, and will make public, a new challenging human interactions data set that we have collected using a network of three cameras. The results show that the approach holds promise to become an effective building block for the analysis of real-time human behavior from multiple cameras.


## Interactive Phrases: Semantic Descriptions for Human Interaction Recognition (2014)
### SIN ACCESO AL ARTICULO

Y. Kong, Y. Jia and Y. Fu, "Interactive Phrases: Semantic Descriptionsfor Human Interaction Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 9, pp. 1775-1788, Sept. 2014, doi: 10.1109/TPAMI.2014.2303090.

https://ieeexplore.ieee.org/abstract/document/6739171

This paper addresses the problem of recognizing human interactions from videos. We propose a novel approach that recognizes human interactions by the learned high-level descriptions, interactive phrases. Interactive phrases describe motion relationships between interacting people. These phrases naturally exploit human knowledge and allow us to construct a more descriptive model for recognizing human interactions. We propose a discriminative model to encode interactive phrases based on the latent SVM formulation. Interactive phrases are treated as latent variables and are used as mid-level features. To complement manually specified interactive phrases, we also discover data-driven phrases from data in order to find potentially useful and discriminative phrases for differentiating human interactions. An information-theoretic approach is employed to learn the data-driven phrases. The interdependencies between interactive phrases are explicitly captured in the model to deal with motion ambiguity and partial occlusion in the interactions. We evaluate our method on the BIT-Interaction data set, UT-Interaction data set, and Collective Activity data set. Experimental results show that our approach achieves superior performance over previous approaches.


## A survey on human activity recognition from videos (2016)
### SIN ACCESO AL ARTICULO

T. Subetha and S. Chitrakala, "A survey on human activity recognition from videos," 2016 International Conference on Information Communication and Embedded Systems (ICICES), 2016, pp. 1-7, doi: 10.1109/ICICES.2016.7518920.

https://ieeexplore.ieee.org/abstract/document/7518920


Understanding the activities of human from videos is demanding task in Computer Vision. Identifying the actions being accomplished by the human in the video sequence automatically and tagging their actions is the prime functionality of intelligent video systems. The goal of activity recognition is to identify the actions and objectives of one or more objects from a series of examination on the action of object and their environmental condition. The major applications of Human Activity Recognition varies from Content-based Video Analytics, Robotics, Human-Computer Interaction, Human fall detection, Ambient Intelligence, Visual Surveillance, Video Indexing etc. This paper collectively summarizes and deciphers the various methodologies, challenges and issues of Human Activity Recognition systems. Variants of Human Activity Recognition systems such as Human Object Interactions and Human-Human Interactions are also explored. Various benchmarking datasets and their properties are being explored. The Experimental Evaluation of various papers are analyzed efficiently with the various performance metrics like Precision, Recall, and Accuracy.


## Human interaction recognition framework based on interacting body part attention (2022)
### SIN ACCESO AL ARTICULO

https://www.sciencedirect.com/science/article/abs/pii/S0031320322001261

Human activity recognition in videos has been widely studied and has recently gained significant advances with deep learning approaches; however, it remains a challenging task. In this paper, we propose a novel framework that simultaneously considers both implicit and explicit representations of human interactions by fusing information of local image where the interaction actively occurred, primitive motion with the posture of individual subject’s body parts, and the co-occurrence of overall appearance change. Human interactions change, depending on how the body parts of each human interact with the other. The proposed method captures the subtle difference between different interactions using interacting body part attention. Semantically important body parts that interact with other objects are given more weight during feature representation. The combined feature of interacting body part attention-based individual representation and the co-occurrence descriptor of the full-body appearance change is fed into long short-term memory to model the temporal dynamics over time in a single framework. The experimental results on five widely used public datasets demonstrate the effectiveness of the proposed method to recognize human interactions from videos.


## Discriminative Multi-Modality Speech Recognition (2020)
https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Discriminative_Multi-Modality_Speech_Recognition_CVPR_2020_paper.html

Vision is often used as a complementary modality for audio speech recognition (ASR), especially in the noisy environment where performance of solo audio modality significantly deteriorates. After combining visual modality, ASR is upgraded to the multi-modality speech recognition (MSR). In this paper, we propose a two-stage speech recognition model. In the first stage, the target voice is separated from background noises with help from the corresponding visual information of lip movements, making the model 'listen' clearly. At the second stage, the audio modality combines visual modality again to better understand the speech by a MSR sub-network, further improving the recognition rate. There are some other key contributions: we introduce a pseudo-3D residual convolution (P3D)-based visual front-end to extract more discriminative features; we upgrade the temporal convolution block from 1D ResNet with the temporal convolutional network (TCN), which is more suitable for the temporal tasks; the MSR sub-network is built on the top of Element-wise-Attention Gated Recurrent Unit (EleAtt-GRU), which is more effective than Transformer in long sequences. We conducted extensive experiments on the LRS3-TED and the LRW datasets. Our two-stage model (audio enhanced multi-modality speech recognition, AE-MSR) consistently achieves the state-of-the-art performance by a significant margin, which demonstrates the necessity and effectiveness of AE-MSR.



## Robust Face Frontalization for Visual Speech Recognition (2021)

https://openaccess.thecvf.com/content/ICCV2021W/TradiCV/html/Kang_Robust_Face_Frontalization_for_Visual_Speech_Recognition_ICCVW_2021_paper.html

Face frontalization consists of synthesizing a frontally-viewed face from an arbitrarily-viewed one. The main contribution of this paper is a robust frontalization method that preserves non-rigid facial deformations, i.e. expressions, to improve lip reading. The method iteratively estimates the rigid transformation (scale, rotation, and translation) and the non-rigid deformation between 3D landmarks extracted from an arbitrarily-viewed face, and 3D vertices parameterized by a deformable shape model. An important merit of the method is its ability to deal with large Gaussian and non-Gaussian errors in the data. For that purpose, we use the generalized Student-t distribution. The associated EM algorithm assigns a weight to each observed landmark, the higher the weight the more important the landmark, thus favoring landmarks that are only affected by rigid head movements. We propose to use the zero-mean normalized cross-correlation (ZNCC) score to evaluate the ability to preserve facial expressions. Moreover, we show that the method, when incorporated into a deep lip-reading pipeline, considerably improves the word classification score on an in-the-wild benchmark.


