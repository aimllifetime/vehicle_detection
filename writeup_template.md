
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Writeup

#### following Writeup includes all the rubric points and how they are addressed 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook function name is: ** _get_hog_features_ **. This is pretty same like from class quiz  

The given training images are read into `vehicle` and `non-vehicle` seperately.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how the final choice of HOG parameters comes.

I tried various combinations of parameters and different color space. 

#### 3. how a classifier is trained using selected features.

Following features are used for classifier feature extraction:
* the HOG feature with the parameter: orient = 9, pix_per_cell = 8, cell_per_block = 2.
* "ALL" three channel hog features are concatinated
* color histogram is used with 32 bins
* spatial feature is used with 32 bin for color distribution.

Each image extracts above feature to form a single feature vector.

For the car and non-car image, extract feature vector for each image.
then use the standardScaler to normalize the feature before do the classifier training.
The data is plit into 80% trainig and 20% validation. 

The classifier uses linear classifier.

The accuracy is 0.9882

### Sliding Window Search

#### 1. This is most tricky part and it took lots of time experiment to video to make sure every frame is detected car if it exists. The function is in **_find_car_in_image_**

* this function mainly define where to search the window for cars. 
* mostly focus on the right side of car for the computation purpose since the video has no
* car on left. it certainly can covers from x_start_stop =[None, None]
* the strategy used here is for the view far from driver, use small xy_window size, such as:
* (64, 64), (90, 90) and smaller overlap such as 0.7 or 0.8
* for the view are close to driver car, use large window size such as (90, 90), (130,130) and (150, 150)
* the overlap uses bigger overlap so that car can be detect mutiple times. The heat threshold > 1 will not mask off the big car.
* more area is in between x_start_stop[700, 1050] for car is small.
* x_start_stop[1000, None] for closer car appears bigger on with larger overlap such as 0.8 and 0.9

The image is read in as RGB and scaled between [0, 1] as the classifier used png image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline works in following flow for each every frame:


Initially used RGB color space and only one channel for HOG feature. The classifier accuracy is around 0.9 something. Later found out adding more channel for HOG increase the accuracy. After that, I add all the feature which came from class, such as color histgram and spatial bins. The YCrCb later is found to be better detecting the white car. 

the searching window scale ideally uses four different scale such as small(64, 64), two medium scales (90, 90) and (130, 130) and finally the largest scale (150, 150)

The search area is purposely placed in the lower and right corner, x starting from 700 and y start from 380.
in this area, different search window area is focused.

Ultimately I searched on few different scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

