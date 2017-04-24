### **Traffic Sign Recognition** 

## Writeup by Tony Knight - 2017/04/21 

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/keepright.png?raw=true" alt="Keep Right!" width=150>

<U><B>Figure 1:</B><I> German "Keep Right" road sign</I></U>

---

Here is a link to my [project code](https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_ClassifierV2_2.ipynb) on Github.

### Data Set Summary & Exploration

I computed statistics on the provided traffic signs data sets using numpy, pandas and standard python math.  This was completed in code cells 3 through 7 on the original data set, on the images converted to contrast enhanced grayscale images and, on a subset of training images equalized across classes.   The relevant stats for the datasets as presented in Table 1.

---

| Image Set        | Count |
|:-----------------|------:|
|Training examples | 34799 |
|Validation examples | 4410 |
|Testing examples | 12630 |
<U><B>Table 1:</B> <I>Image count in each Dataset</I></U>

---

The image data was provided in arrays of 32x32x3 with RGB intensities provided as integer values between 0 and 255.

The number of classes of traffic signs in each dataset is 43.  After loading the signnames file, I was able to use the Pandas dataframe to generate a listed breakdown of the image class in the training set by signname which is provided in Table 2.

---

| Sign Identification                       | Count of Images|
|:---------------------------------------------------|------:|
|Ahead only                                          |  1080 |
|Beware of ice/snow                                  |   390 |
|Bicycles crossing                                   |   240 |
|Bumpy road                                          |   330 |
|Children crossing                                   |   480 |
|Dangerous curve to the left                         |   180 |
|Dangerous curve to the right                        |   300 |
|Double curve                                        |   270 |
|End of all speed and passing limits                 |   210 |
|End of no passing                                   |   210 |
|End of no passing by vehicles over 3.5 metric tons  |   210 |
|End of speed limit (80km/h)                         |   360 |
|General caution                                     |  1080 |
|Go straight or left                                 |   180 |
|Go straight or right                                |   330 |
|Keep left                                           |   270 |
|Keep right                                          |  1860 |
|No entry                                            |   990 |
|No passing                                          |  1320 |
|No passing for vehicles over 3.5 metric tons        |  1800 |
|No vehicles                                         |   540 |
|Pedestrians                                         |   210 |
|Priority road                                       |  1890 |
|Right-of-way at the next intersection               |  1170 |
|Road narrows on the right                           |   240 |
|Road work                                           |  1350 |
|Roundabout mandatory                                |   300 |
|Slippery road                                       |   450 |
|Speed limit (100km/h)                               |  1290 |
|Speed limit (120km/h)                               |  1260 |
|Speed limit (20km/h)                                |   180 |
|Speed limit (30km/h)                                |  1980 |
|Speed limit (50km/h)                                |  2010 |
|Speed limit (60km/h)                                |  1260 |
|Speed limit (70km/h)                                |  1770 |
|Speed limit (80km/h)                                |  1650 |
|Stop                                                |   690 |
|Traffic signals                                     |   540 |
|Turn left ahead                                     |   360 |
|Turn right ahead                                    |   599 |
|Vehicles over 3.5 metric tons prohibited            |   360 |
|Wild animals crossing                               |   690 |
|Yield                                               |  1920 |

<U><B>Table 2:</B><I> Image count per class in Training Dataset</I></U>

---


I also observed how Pandas could output stats about individual columns in the dataframe.  The individual column values had been calculated on individual images in the grayscale dataset.

---

|stat |      X_max  |    X_mean   |     X_min   |     X_std   |      X_sum |
|:---:|:---:|:---:|:---:|:---:|:---:|
|count| 34799.000000  | 34799.000000 | 34799.000000 | 34799.000000 |  34799.000000 |  
|mean |    254.995862 |   139.554981 |     6.117503 |    63.027796 | 142904.300181 |  
|std  |      0.131900 |    11.417725 |     7.621655 |     7.089961 |  11691.750853 |  
|min  |    244.000000 |    97.950195 |     1.000000 |    35.793909 | 100301.000000 |  
|25%  |    255.000000 |   132.451172 |     2.000000 |    58.799550 | 135630.000000 |  
|50%  |    255.000000 |   136.599609 |     4.000000 |    64.331295 | 139878.000000 |  
|75%  |    255.000000 |   142.957520 |     6.000000 |    68.217517 | 146388.500000 |  
|max  |    255.000000 |   225.502930 |    70.000000 |    79.894665 | 230915.000000 |  

<U><B>Table 3:</B><I> Stats on Pixel Intensity in the Training Dataset produced by Pandas</I></U>

---


The code to produce the visualizations of the data distribution is contained in the fourth through seventh code cells of the Jupyter notebook.  

Bar charts showing how the images are distributed between the classes in each of the datasets are shown in Figure 2.

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/datasetclassdistribution1.png?raw=true" alt="bar chart of classes in each set" width=600>

<U><B>Figure 2:</B><I> Bar Chart of Class Distributions</I></U>

---

I investigated how the pixel intensities were distributed after contrast enhancing the grayscale images.  There appear to be some interesting artefacts in this data, and I am concerned by how many pixel values were clipped at the maximum level of 255, however, the general shape of the data set looks like a "normal" distribution, which bodes well for analysis.

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/grayscalehistogram1.png?raw=true" alt="Histogram of pixel intensity in Grayscale Dataset" width=500>

<U><B>Figure 3:</B><I> Histogram of Pixel Intensity in CLAHE equalized Training Set</I></U>

---

After normalizing the data to a standard deviation of 1 and a mean of 0, the pixel intensity distribution looks like this:

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/NormalizedPixelHistogram.png?raw=true" alt="Histogram of pixel intensity after normalizing" width=400>

<U><B>Figure 4:</B><I> Histogram of Pixel Intensity in Training Set after Normalization</I></U>

---

### Model Architecture Design and Tests

I initially used the LeNet-5 implementation of a standard 2 layer feed forward convolutional network activated by softmax and ReLU functions which I had developed in the earlier TensorFlow project.

To feed the model, I started by using the raw RGB image data to see how the model would work.  I was able to get about 90% accuracy on the training data set.  I then switched to using HSV image data by converting the images using the openCV functions.  However, this did not improve results.  I read in the paper by [Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that they were able to get highest accuracy when using only grayscale (Y channel) information, so I decided to try using just grayscale information.

I found that there was a large visual difference in the brightness of the images, and I wanted to try and compensate for this.  I switched to using CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhanced grayscaled versions of the image.   I had also read articles including Nvidia's ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316v1.pdf) document indicating that CLAHE was useful in traffic sign recognition (although it was usually done in HSV space not on grayscale images).  

The code I used to convert the images is found in code cell 2 of the notebook.  I tested using both Scipy Image (skimage) and opencv (cv2) routines to complete the CLAHE enhancement after grayscaling the image.  I experimented with different tiling and cliplimit values to obtain both the best visual result, and a good pixel distribution (see image x above).

An example of a traffic sign image before and after grayscaling and CLAHE enhancement is presented in figure 5.

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/traffic.png?raw=true" alt="RGB, and Grayscale and CLAHE versions of Traffic Sign Image" width=400>

<U><B>Figure 5:</B><I> Original RGB, and Grayscale and CLAHE versions of Traffic Sign Image</I></U>

---

As a last step, I normalized the image data using the individual image mean and standard deviation to produce images with pixel intensities having a standard deviations of 1 centered at 0.

#### Augmentation and Equalization of Datasets

Starting with the training, validation, and test data sets provided, I used two strategies to try and improve validation accuracy.  First, I tried equalizing the training data set so that there were an equal number of images in each class of image, and in this way the model would not be biased to select images that were over-represented in the training dataset.  However, In my tests, this did not improve validation or test accuracy, possibly because this limited the size of the training set too much.

Secondly , I tried augmenting the training dataset by adding distortions of the original images.  In code cells 11 to 19 I generated copies of the validation set images that had random amounts of motion blur, reductions in scale, displacement, rotation, and fixed perspective distortions (affine transformations that could be called tilts) in each of 4 directions (up left right and down).

---

| Image Sets after Augmenting        | Count |
|:-----------------|------:|
|- Original Training examples | 34799 |
|- blurred copies | 34799|
|- scaled copies | 34799|
|- displaced copies | 34799|
|- rotated copies | 34799|
|- tilted copies (4 directions) | 139196|
|Total Training Set Size | 313191 |
|||
|Validation Set Size | 4410 |
|||
|Testing Set Size | 12630 |

<U><B>Table 4:</B><I> Images in Data Sets after Augmentation</I></U>

---

My final training set had 313191 images. My validation set and test set remained unchanged with 4410 and 12630 images respectively.  I felt that the distortions to the image set would be useful both because it would add additional examples to the test set, and because the distortions applied were likely to occur in the field.

Comments in the Sermanet and Lecun paper indicated that care should be taken to ensure that the validation set did not include images that were taken from the same run of images taken from the same sign, as this would not give the best indication of how well the model worked on general examples of the signs.  I had not checked this.

An example of an original Grayscale Image and a set of the augmented images generated from it are presented in Figure 6
---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/traffic.png?raw=true" alt="Original and augmented versions of Traffic Sign Image" width=400>

<U><B>Figure 6:</B><I> Original grayscale image of traffic sign nd set of augmented images produced from it</I></U>

---

#### 3. Multi-Scale Convolutional Network

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/MultiScaleLenet.png?raw=true" alt="Multi Scale Convolutional Network Diagram" width="600">

<U><B>Figure 7:</B><I> Diagram of Final version of Multi-Scale Convolutional Network</I></U>

---

A visualization of the final model is displayed in Figure 7, and is also summarized in Table 5. 
The code for my final model is located in the seventh cell of the jupyter notebook. 

---

| # | Layer         		|     Description	        					|Input  | Output |
|:--|:---------------------|:---------------------------------------------|:------:|:------:|
| 0 | Input         		| Grayscale image   					| 	|32x32x1|
|	|	         		|    					| 
| 1 | Convolution 5x5     	| 1x1 stride, valid padding  	|32x32x1 |28x28x16 |
| 1 |ELU					|												|28x28x16 |28x28x16 |
| 1 | Max pooling 2x2	   	| 2x2 stride,valid padding			|28x28x16 |14x14x16|
|	|	         			|    											| | | 
| 2 | Convolution 5x5	    | 1x1 stride, valid padding   |14x14x16 |10x10x16 |
| 2 | ELU					|												|10x10x16 |10x10x16 |
| 2 | Max pooling 2x2      	| 2x2 stride, valid padding				|10x10x16 |5x5x16 |
| 2 | Flatten   	      	|  				|5x5x16 | 400 |
|	|	         		|    					|  | |
| 3 | Fully connected		|         							| 400 |120 |
| 3 | ELU					|         							| 120 |120 |
| 3 | Dropout				| Dropout of 50% in Training 		| 120 | 120 |
|	|						|												| | |
| 1A | Max pooling 4x4	      	| 4x4 stride, valid padding 				|14x14x16 | 3x3x16 |
| 1A | Flatten   	      	|  								|  3x3x16 | 144|
| 1A | Dropout				| Dropout of 50% in Training		| 144| 144|
|	|						|												| | |
| 4 | Fully connected		| Combined input from Layer 3 and 1A | 264 | 84 |
| 4 | ELU					|         							| 84 | 84 |
|	|						|										| | |
| 5 | Fully Connected		|         				| 84 | 43 |
|	|						|										| | |
| 5A  | Softmax             | Softmax applied to output of model |43 | 43|

<U><B>Table 5:</B><I> Details of Layers in Final MultiScale Convolutional Network</I></U>

---

#### Model Training.

The model was trained in cell ???

During training runs, I tried to optimize hyper parameters including the batch size, training rate, and number of epochs.  I used the Adam (Adaptive Moment Estimation) optimizer as opposed to standard stochastic gradient descent optimizer.  After learning about explicit exponential decay of the training rate, I added it in to the training process.  My final parameters are presented in Table 6.

---

| Parameter | Value |
|:-------|:----:|
|Epochs | 200 |
|Batch Size | 64 |
|Initial Learning Rate | 0.0007 |
|Learning Rate Decay  Rate | 0.985 |
| mu | 0 |
| sigma | 0.1 |

<U><B>Table 6:</B><I> Details of HyperParameters used in Model Training</I></U>

---

Initially I used the ReLU activation function in layers, but I switched to using ELU activation after reading that it had been shown to learn faster and perform better.


#### Model Evolution during completion of the Project 

I started with the LeNet-5 model using RGB input, and then moved to using grayscale input and was able to increase peak the peak validation accuracy it achieved from 90% to 93%.  I then augmented the input set and this enabled the model to achieve validation accuracy above 95%.  At this point I experimented with adjusting the cv2 CLAHE algorithm, and found that I achieved best results when the cliplimit was increased to 32, and this increased validation accuracy to 96%.  At this point I found that I had been running dropout in the validation tests.  I restricted dropout to the training stage, and the validation accuracy increased to 97.8 and my first check on the test set gave an accuracy result of 95.6%.  (My accuracy calculations are completed in code cell XXX)

At this point I tried equalizing the input set so that there were an equal number of examples per class in the training set (See code cell 6).  This reduced the number of samples in the training set to approximately 20% of its original size.  Validation accuracy dropped to 95%, and test set accuracy dropped to 92%.

I ran tests substituting the skimage (scikit-image) version of CLAHE instead of the CV2 variant and played with the parameters.  I found I could achieve comparable accuracy results, but that the skimage results were not as visually pleasing and were generally darker, so I switched back to cv2.

I had originally been standardizing the input using the mean and standard deviation of the set, but switched to using the values calculated from each individual image.  This had little effect on the results produced by the model, possibly because the CLAHE enhanced images have relatively similar intensity distributions.

I switched my augmentation process for the rotation, scaling, and displacement distortions which needed to add borders to the images prior to the distortion.  My rationale was that the border patterns could introduce interesting features that might confuse the network, and that random noise would be less likely to introduce this.  I modified my code so that the borders were generated using a random noise background instead of simply extending the edge pixels.  I further enhanced this algorithm to use a random selection of pixels taken from the image itself to better match light or dark images.  This appeared to help as validation accuracy improved to 98.3%.  However, I also fixed a bug which had been limiting rotations and displacements to only positive increments at the same time, so this may have also had an effect!  The model achieved 96.7% on the test set.  I also tested the model on 7 images collected from the internet (primarily from Google street view in Hanover).  The model identified 6 of the 7 images properly.

I switched the activation function in each layer from ReLU to ELU, and saw a small increase in accuracy for each set.

I then tried modifying the network to include a multiscale feature, feeding a pooled set from the first layer into the fourth layer.  This architecture achieved the following accuracies: Training 99.9%, Validation 95.6%, and Test set 92.9% after 150 epochs of training.

I moved back to the full (unequalized) training dataset and added augmentations for tilts up, down, left and right.  This resulted in a total of 313,191 images in the training dataset. This produced accuracies for Training: 97.5% Validation: 98.3% and Test 95.7% and 100% for the 7 Internet clippings.  

I observed that I had been using a softmax activation in layer 1 and switched this to ELU.  This resulted in slightly higher training accuracy, but slightly lower validation and test results after 100 epochs.

I then tried increasing the width of the layers, adjusting the first convolution to produce 16 feature planes instead of 6. This resulted in Training 98.37%, validation 98.21%, Test 96.3%, and 100% for the 7 internet Images after 25 Epochs.

I ran the model a few times for a longer set of Epochs and introduced the use of a decaying learning rate. <B>My final results were Training Accuracy 98.10%, Validation Accuracy 98.84%, Loss 0.082, Test set 96.86%,  and 7 for 7 on the Internet Images</B>.

I generated Precision, Recall, and F1 Scores, and a confusion matrix for the trained model (See Table 7 and Figure 7).  These tables generally showed the model was relatively good at classifying the images, with one particularly poor exception.  Class 27, 'pedestrians', has both bad precision and recall and a dismal F1 rating of 61.3 whereas the scores for the other classes often in the high 90's but generally above high 80s. 

---

|Class|	Precision|	Recall|	F-score|	Items|	Sign Description|
|----:|---------:|-------:|-------:|--------:|:-----------|
|0	|98.4	|100.0	|99.2	|60 	|Speed limit (20km/h)|
|1|	97.8	|99.0	|98.4	|720	|Speed limit (30km/h)|
|2|	96.1	|99.3	|97.7	|750	|Speed limit (50km/h)|
|3|	97.0	|94.9	|96.0	|450	|Speed limit (60km/h)|
|4|	99.5	|97.9	|98.7	|660	|Speed limit (70km/h)|
|5|	92.1	|94.6	|93.3	|630	|Speed limit (80km/h)|
|6|	97.3	|95.3	|96.3	|150	|End of speed limit (80km/h)|
|7|	96.7	|98.4	|97.6	|450	|Speed limit (100km/h)|
|8|	99.3	|94.7	|96.9	|450	|Speed limit (120km/h)|
|9|	99.6	|99.2	|99.4	|480	|No passing|
|10|	99.8	|98.8	|99.3	|660	|No passing for vehicles over 3.5 metric tons|
|11|	99.0	|92.4	|95.6	|420	|Right-of-way at the next intersection|
|12|	98.8	|98.8	|98.8	|690	|Priority road|
|13|	98.2	|99.9	|99.0	|720	|Yield|
|14|	99.3	|100.0	|99.6	|270	|Stop|
|15|	89.9	|84.8	|87.3	|210	|No vehicles|
|16|	99.3	|100.0	|99.7	|150	|Vehicles over 3.5 metric tons prohibited|
|17|	100.0	|98.9	|99.4	|360	|No entry|
|18|	95.7	|96.7	|96.2	|390	|General caution|
|19|	100.0	|100.0	|100.0	|60	|Dangerous curve to the left|
|20|	97.6	|92.2	|94.9	|90	|Dangerous curve to the right|
|21|	98.5	|72.2	|83.3	|90	|Double curve|
|22|	98.1	|88.3	|93.0	|120	|Bumpy road|
|23|	95.5	|100.0	|97.7	|150	|Slippery road|
|24|	94.0	|87.8	|90.8	|90	|Road narrows on the right|
|25|	94.8	|98.8	|96.7	|480	|Road work|
|26|	94.0	|87.2	|90.5	|180	|Traffic signals|
|27|	<B>66.7</B>	|<B>56.7</B>	|<B>61.3</B>	|60	|Pedestrians|
|28|	82.2	|98.7	|89.7	|150	|Children crossing|
|29|	83.5	|95.6	|89.1	|90	|Bicycles crossing|
|30|	86.0	|86.0	|86.0	|150	|Beware of ice/snow|
|31|	96.8	|100.0	|98.4	|270	|Wild animals crossing|
|32|	84.1	|96.7	|89.9	|60	|End of all speed and passing limits|
|33|	98.6	|98.1	|98.3	|210	|Turn right ahead|
|34|	96.0	|100.0	|98.0	|120	|Turn left ahead|
|35|	100.0	|96.2	|98.0	|390	|Ahead only|
|36|	100.0	|100.0	|100.0	|120	|Go straight or right|
|37|	87.0	|100.0	|93.0	|60	|Go straight or left|
|38|	99.9	|99.6	|99.7	|690	|Keep right|
|39|	94.7	|100.0	|97.3	|90	|Keep left|
|40|	91.1	|91.1	|91.1	|90	|Roundabout mandatory|
|41|	100.0	|100.0	|100.0	|60	|End of no passing|
|42|	97.8	|96.7	|97.2	|90	|End of no passing by vehicles over 3.5 metric tons|

<U><B>Table 7:</B><I> Precision, Recall, and F1 Score for Model</I></U>

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/Confusion.png?raw=true" alt="Confusion Matrix" width=400>

<U><B>Figure 7:</B><I> Confusion Matrix generated from the model (Darker implies more hits)</I></U>

---

### Testing the Model on New Images

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/keepright.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Limit70.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/noentry.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/row_nxt_intersection.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Stop.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Straightahead.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/yield.png?raw=true" width=100>

<U><B>Figure 8:</B><I> Road sign images clipped from the Internet</I></U>

---

The 7 images of signs I retrieved from the internet are shown in Figure 8.  I thought the sixth image (straight ahead) might be difficult to classify due to the distortion of the circular shape.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

The results of the predictions made by the model are given in Table 7.

---

|Sign Class| Sign Name		|Prediction Class |    Sign Name      | Confidence | 
|:--|----------------------:|:-------------------|:------------------:|:--------:| 
| 38| Keep Right      		| 38| Keep Right								| 1.000 |
| 4 | Speed Limit 70		| 4 | Speed Limit 70							| 1.000 |
| 11| RoW at next Intersection |11 | RoW at next Intersection Bumpy Road	| 1.000 |
| 14| Stop 				| 14 | Stop		      							| 1.000 |
| 13| Yield					| 13 |Yield		      							| 0.999 |
| 17| No Entry				| 17| No Entry									| 0.996 |
| 35| Ahead Only			| 35 |Ahead Only      							| 0.942 |

<U><B>Table 7:</B><I> Top Prediction and Probability generated by Model on Internet Images </I></U>

---

The model was able to correctly guess All 7 traffic signs, giving an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.8%.  The softmax probability for 5 generated for the fist choice for 5 of the images was effectively 1 (0.999 or greater).  

The model was 99.6% sure of the "No Entry" sign prediction, generating only .003 and .001 chance of the sign being a Yield or Stop sign respectively.  This was somewhat interesting as the "No Entry" sign is circular whereas the Yield and Stop signs have straight edges (triangle and octoganal).  

The "Ahead Only" sign had the lowest top prediction probability of 0.942, which was still very good, with secondary guesses of General Caution, Traffic Signals, Speed Limit, and Go Straight or Left rounding out the top 5 predictions.  (See Table 8 for details and notes).  


| Probability      	| Predicted Class| Predicted Sign Name			|  Notes |
|:-----------------:|:---------------:|:-----------------------------:| :-------|
| 0.942 | 35| Ahead Only  		| |
| 0.047 | 18| General Caution |Vertical line down center, but triangular shape, white background|
| 0.004	| 26| Traffic Signals |Vertically aligned dots down central axis, but triangular shape, white background|
| 0.003	| 3 | Speed Limit 60 |circular, possible arrow head shape between 6 and 0 but white background|
| 0.003	| 37| Go Straight or left |circular, blue background, vertical arrow, but additional horizontal arrow|

<U><B>Table 8:</B><I> Top 5 Predictions and Probabilities generated by model on image of "Ahead Only" sign </I></U>

---

Based on the precision and recall report (table ) and confusion matrix (Figure  generated on the results from the test set; 1 image each from classes 11, 12, 13, 33, 40, 2 images from 34, and 8 from 37 were classified as 35 (Ahead Only), but no class 35 images were classified in any other class (perfect precision, but lower recall).


### Failures in Model Prediction

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/1speed60.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/2speed60.png.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/3speed70.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/4speed70.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/5speed80.png?raw=true" width=100>


<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/6noentry.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/7notrucks.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/8pedestrians.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/9notruckpassing.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/incorrectchoices/10roundabout.png?raw=true" width=100>

<U><B>Figure 9:</B><I> 10 sign images incorrectly classified by the model</I></U>

---

Of the 12630 images in the test set, 397 were incorrectly classified by the model.  Figure 9 presents 10 randomly selected samples of incorrectly classified images. The top 5 softmax predictions were generated for these images in cell ???.  In 7 of the cases the model picked the correct sign as a second most likely choice, and in two it was the third choice.  Only with sign 9230, a "No Vehicles" sign, did the model fail to have the correct sign label in its top 5 selections.  The model was very unsure of any choice in this case, with probability of its highest choice being only 0.324 - similar to with random noise.

It was gratifying to me that I would have a hard time correctly guessing what category half of these samples(4722, 2141, 5786, 3124, 8945) belonged to as they were blurred or dark.  

Based on these images, it appears that the model might be thrown off, where a human could easily identify the sign, by the presence of shading only on part of the image (image 5518), or by the presence of shiny glare spots (image 9230 and 12436).  I also think that CLAHE enhancement may be introducing features in areas of the image which are a single solid color (image 9230) where the algorithm attempts to maximize any local contrast.

The model could potentially be trained to overcome these issues by augmenting the training set with samples including artificially induced glare spots and shaded areas on the signs, and by investigating if CLAHE is introducing features on areas of solid color.


#### Random Noise Sample

---

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/RandomNoise.png?raw=true" alt="Random Noise Sample" width=200>

<U><B>Figure 8:</B><I> Synthetic image containing random noise</I></U>

---
 

| Probability      	| Predicted Class| Predicted Sign Name			|  Notes |
|:-----------------:|:---------------:|:-----------------------------:| :-------|
| 0.279 | 40|Roundabout mandatory|Recall 91.1 |
| 0.216 | 11|Right-of-way at the next intersection|Recall 92.4 |
| 0.189	| 27|Pedestrians|Recall 56.7 |
| 0.104	| 26 | Traffic Signals |Recall 87.2|
| 0.096	| 12| Priority Road |Recall 98.8|

<U><B>Table 9:</B><I> Top 5 Predictions and Probabilities generated by model on image of random noise</I></U>

---
I created a random noise image (image 10) and ran it through the model. The top five guesses for a matching sign are presented in Table 9.  The low but fairly similar probabilities associated with all 5 top guesses suggests that the noise sample does not fit any of the sign classes particularly well.




### Visualization of Feature Maps

I created code to enable visualization of the feature maps for any image at all stages where the map was still rectangular (before flattening) which is in code cell .  I found that the output of the layer 1 max pooling stage provided the best visual clues to how the model was working.  Figure 10 presents the feature maps for a "Keep Right" sign image before and after the model has been trained.


### Additional Work

I think that the following areas could have been investigated to see if the model could be improved:

| Augmentation | Produce sets of images with artificial shading of portions of the sign | 
| Architecture | Investigate effect of use of more layers |


