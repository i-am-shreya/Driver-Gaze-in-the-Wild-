# Driver Gaze 'in-the-Wild'

## Abstact 
Labelling of human behavior analysis data is a complex and time consuming task. In this paper, a fully automatic technique for labelling an image based gaze behavior dataset for driver gaze zone estimation is proposed. Domain knowledge can be added to the data recording paradigm and later labels can be generated in an automatic manner using speech to text conversion. In order to remove the noise in STT due to different ethnicity, the speech frequency and energy are analysed. The resultant Driver Gaze in the Wild DGW dataset contains 586 recordings, captured during different times of the day including evening. The large scale dataset contains 338 subjects with an age range of 18-63 years. As the data is recorded in different lighting conditions, an illumination robust layer is proposed in the Convolutional Neural Network (CNN). The extensive experiments show the variance in the database resembling real-world conditions and the effectiveness of the proposed CNN pipeline. The proposed network is also fine-tuned for the eye gaze prediction task, which shows the discriminativeness of the representation learnt by our network on the proposed DGW dataset.

## Paper: 

Speak2label: Using domain knowledge for creating a large scale driver gaze zone estimation dataset [[pdf]](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Ghosh_Speak2Label_Using_Domain_Knowledge_for_Creating_a_Large_Scale_Driver_ICCVW_2021_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/supplemental/Ghosh_Speak2Label_Using_Domain_ICCVW_2021_supplemental.pdf) [[arXiv]](http://arxiv.org/abs/2004.05973) [[AVVision]](https://openaccess.thecvf.com/ICCV2021_workshops/AVVision#:~:text=%2C%20Nicu%20Sebe-,%5Bpdf%5D%20%5Bsupp%5D%20%5BarXiv%5D%20%5Bbibtex%5D,-%40InProceedings%7BGhosh_2021_ICCV%2C%0A%20%20%20%20author)

If you find the work useful for your research, please consider citing our work:
```
@InProceedings{Ghosh_2021_ICCV,
    author    = {Ghosh, Shreya and Dhall, Abhinav and Sharma, Garima and Gupta, Sarthak and Sebe, Nicu},
    title     = {Speak2Label: Using Domain Knowledge for Creating a Large Scale Driver Gaze Zone Estimation Dataset},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2896-2905}
}
```
## DGW Dataset Overview:
Following frames are from the Driver Gaze in the Wild (DGW) dataset. The Driver Gaze in the Wild (DGW) dataset contains 586 recordings of 338 subjects in different illumination conditions.

## Audio Based Automatic Labelling Framework:
The data has been collected in a car with different subjects at the driver’s position. We pasted number stickers on different gaze zones of the car. The nine car zones are chosen from back mirror, side mirrors, radio, speedometer and windshield. The recording sensor is a Microsoft Lifecam RGB camera, which contains a microphone as well. For recording, we asked the subjects to look at the zones marked with numbers in different orders. For each zone, the subject has to fixate on a particular zone number and speak the zone’s number and then move to the next zone. For recording realistic behaviour, no constraint is mentioned to the subjects about looking by eye movements and/or head movements. The subjects chose the way in which they are comfortable. This leads to more naturalistic data. 

Overview of the automatic data annotation technique is shown in the following figure. On the top are the representative frames from each zone. Please note the numbers written in alphabets below the curve. On the bottom right are reference car zones. 

## Demo
Please find the following demo.

<iframe width="560" height="315" src="https://www.youtube.com/embed/S0CJ1X9GnR8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
## Notes
1. We train an Inception V1 network on the detected faces (we use Dlib's face detction library) to predict the gaze zones.
2. Evaluation protocol is CLASSWISE ACCURACY. Baselines for validation set is 60.10% and for test set is 60.98%.
3. Please refer to the paper "Speak2Label: Using Domain Knowledge for Creating a Large Scale Driver Gaze Zone Estimation Dataset" for more details. 
4. Please use the person specific faces as uploaded in face folder in any academic publication. 

## Contact:
For accessing the database for research or commercial purpose, please email Abhinav Dhall at abhinav[at]iitrpr.ac.in, abhinav[DOT]dhall[at]monash[DOT]edu.

## Acknowledgements
This repository build on top of Attention augmented convolution [link](https://github.com/titu1994/keras-attention-augmented-convs).
