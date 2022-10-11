# Driver Gaze 'in-the-Wild'

## Abstact 
Labelling of human behavior analysis data is a complex and time consuming task. In this paper, a fully automatic technique for labelling an image based gaze behavior dataset for driver gaze zone estimation is proposed. Domain knowledge can be added to the data recording paradigm and later labels can be generated in an automatic manner using speech to text conversion. In order to remove the noise in STT due to different ethnicity, the speech frequency and energy are analysed. The resultant Driver Gaze in the Wild DGW dataset contains 586 recordings, captured during different times of the day including evening. The large scale dataset contains 338 subjects with an age range of 18-63 years. As the data is recorded in different lighting conditions, an illumination robust layer is proposed in the Convolutional Neural Network (CNN). The extensive experiments show the variance in the database resembling real-world conditions and the effectiveness of the proposed CNN pipeline. The proposed network is also fine-tuned for the eye gaze prediction task, which shows the discriminativeness of the representation learnt by our network on the proposed DGW dataset.

## Paper: 

Speak2label: Using domain knowledge for creating a large scale driver gaze zone estimation dataset [[pdf]](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Ghosh_Speak2Label_Using_Domain_Knowledge_for_Creating_a_Large_Scale_Driver_ICCVW_2021_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/supplemental/Ghosh_Speak2Label_Using_Domain_ICCVW_2021_supplemental.pdf) [[arXiv]](http://arxiv.org/abs/2004.05973) [[bibtex]](https://openaccess.thecvf.com/ICCV2021_workshops/AVVision#:~:text=%2C%20Nicu%20Sebe-,%5Bpdf%5D%20%5Bsupp%5D%20%5BarXiv%5D%20%5Bbibtex%5D,-%40InProceedings%7BGhosh_2021_ICCV%2C%0A%20%20%20%20author)

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
