# Portrait-Segmentation using SegNet

**Portrait segmentation** refers to the process of segmenting a person in an image from its background.
Here we use the concept of **semantic segmentation** to predict the label of every pixel in an image. Here we limit ourselves to **binary classes** (person or background) and use only plain **portrait-selfie** images for matting.

This technique is widely used in computer vision applications like **background replacement and background blurring** on mobile devices.

## DataSet
https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets)


### INPUT (256 * 256 * 3)
![alt text](https://github.com/ajayjalluri/Portrait-Segmentation-using-SegNet/blob/master/Images/bill_input.jpeg)

### PREDICTED MASK (256 * 256 * 1) - OUTPUT OF THE TRAINED SEGENT MODEL
![alt text](https://github.com/ajayjalluri/Portrait-Segmentation-using-SegNet/blob/master/Images/bill_mask.jpeg)

### FOREGROUND IMAGE
![alt text](https://github.com/ajayjalluri/Portrait-Segmentation-using-SegNet/blob/master/Images/bill_OUTPUT_b.jpeg)

## PORTERDUFF AND GAUSSIAN_BLUR
![alt text](https://github.com/ajayjalluri/Portrait-Segmentation-using-SegNet/blob/master/Images/bill_final3.jpeg)

## References

* https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets
* https://github.com/aisegmentcn/matting_human_datasets
* https://towardsdatascience.com/background-removal-with-deep-learning-c4f2104b3157
* https://github.com/anilsathyan7/Portrait-Segmentation
