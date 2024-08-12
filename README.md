# Houston Hyperspectral Image Classification

The dataset was part of the [2013 Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=459) organized by the Data Fusion Technical Committee.

The project combines Hyperspectral images (HSI) with LiDAR images to improve the classification accuracy of data collected from remote sensing devices.

This is a multi-classification problem with 15 regions of interest (ROIs). The regions are defined as follow:

1 -- Healthy grass\
2 -- Stressed grass\
3 -- Synthetic grass\
4 -- Trees\
5 -- Soil\
6 -- Water\
7 -- Residential\
8 -- Commercial\
9 -- Road\
10 -- Highway\
11 -- Railway\
12 -- Parking Lot 1\
13 -- Parking Lot 2\
14 -- Tennis Court\
15 -- Running Track

I compared SVM to Simple Neural Network using 4 sets of data:

1.  Raw HSIs
2.  Raw HSI + LiDAR
3.  Denoised HSI
4.  Denoised HSI + LiDAR

Robust PCA (RPCA) was used to remove the noise. The algorithm was developed by [Candès et al. 2011](https://people.eecs.berkeley.edu/~yima/psfile/JACM11.pdf "RPCA paper"), based on the original work of [Lin et al. 2009](https://people.eecs.berkeley.edu/~yima/matrix-rank/Files/rpca_algorithms.pdf). The method assumes that the noise is sparse and tries to decompose the image into a low-rank component (smooth) and a sparse component (noisy).

The images in general were not noisy, however, some spectral bands had what seemed like a dark overlay across the entire image. This was caused by the presence of a very few white pixels, whose values exceeding 10,000, when other pixels' values did not exceed 2600. This skews the color map and makes it seem as if there's a dark film.

Because this noise was extremely sparse, I had to carefully tune the hyperparameters in order to avoid removing important features. Results below:

![Raw HSI](output/127th-HSI-Raw-HSI.png)

![Denoised Low Rank Component](output/127th-HSI-Denoised-(Low-Rank-Component).png)

![Noise Component](output/127th-HSI-Noise-Component.png)

The training set is generally balanced. Interestingly though, the area covered by the cloud was excluded, which is where most of the misclassification happened. I assume is was by design.

![Training indices](output/Ground-Truth-Training.png)

![Testing indices](output/Ground-Truth-Testing.png)

![Training + testing indices](output/Ground-Truth-Training-+-Testing.png)

![Regions of Interest](output/Regions-of-Interest-(ROI).png)

The LiDAR image did not seem to be affected by the cloud, so in theory, adding it to the dataset should improve accuracy.

![Lidar](output/LiDAR.png)

## Models and Performance:

The SNN performed slightly better than SVM, and the LiDAR image did improve the accuracy for both models.

Denoising the HSIs was not useful for the SVM model, and even though the chart indicates that the denoised data improved the NN model, this was one of many other runs that I tested which produced a better model using the raw data instead, this is due to the highly non-convex nature of the NN, which is both a curse and a blessing.

I picked the below models because I had to fix the seed to make it reproducible. In saying that, using a NN with either dataset combined with the LiDAR image consistently produced accuracy between 81% and 83%.

![Model accuracy chart](output/models-comparision.png)

By examining the mapped predictions below (I used the entire dataset, not just the test set), the shadow area has the poorest prediction, especially for the SVM model, where adding the LiDAR did not seem to affect the shadow region.

The SNN has a better prediction accuracy in the shadow area, especially with the LiDAR image.

### SVM Predictions:

![raw prediction](output/SVM-Predicted-Classes-Raw-Data.png)

![denoised prediction](output/SVM-Predicted-Classes-Denoised-Data.png)

![raw with lidar prediction](output/SVM-Predicted-Classes-Raw-+-LiDAR-Data.png)

![denoised with lidar prediction](output/SVM-Predicted-Classes-Denoised-+-LiDAR-Data.png)

### SNN Predictions:

![raw prediction](output/NN-Predicted-Classes-Raw-Data.png)

![denoised prediction](output/NN-Predicted-Classes-Denoised-Data.png)

![raw with lidar prediction](output/NN-Predicted-Classes-Raw-+-LiDAR-Data.png)

![denoised with lidar prediction](output/NN-Predicted-Classes-Denoised-+-LiDAR-Data.png)

## Conclusion:

Even though the classes in the training set were balanced, the data was not balanced spatially, this is probably to make it a more challenging problem for the purposes of the competition.

In a real world scenario, it would be expected to train the model on data collected under different kinds of conditions, including shadows caused by clouds, in which case the models we used proved to perform very well, and the LiDAR image might not even be needed.

## Project Structure:

[Main Notebook](main-Houston-HSI.ipynb)

[Used Packages](used_packages.py)

[Data Processing](data.py)

-   [Data Folder](data) (HSI were excluded due to large size. Refer to the contest webpage for the full dataset)

[Classification Models](models.py)

[Utility Functions](utils.py)

[RPCA Image Decomposition](RPCA.py)

[Saved NN Models](saved%20models/best%20models)

## Author:

Ali Moughnieh: [\@amoughnieh](https://github.com/amoughnieh)
