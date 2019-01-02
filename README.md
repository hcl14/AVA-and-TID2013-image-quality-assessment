# AVA and TID2013 image quality assessment

Reproduction of the models for aesthetic and quality evaluation of images from the Google article [Hossein Talebi, Peyman Milanfar, NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424)

The train-test split used belongs to [idealo](https://github.com/idealo/image-quality-assessment), this was done for comparison of the models quality.

Using those models on practice yields some problems, which are discussed in **Usage** paragraph. Please read everything below before trying the code! You need to train models manually, stopping and restarting with different learning rate. I have provided already trained models, and made comparison with another available ones from Idealo. Some conclusions about the datasets and the quality which can be achieved, can be found in **Usage** chapter.


I did not put much effort into organizing code, but you can get well with taking pretrained models and using them in your project.


Requirements: Python 3.x, Tensorflow >= 1.10.0, Keras >= 2.2.2



## AVA models

The models were trained on [AVA: A Large-Scale Database for Aesthetic Visual Analysis](http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460).

Both models predict 10-bin normalized histogram of human opinion scores (1-10).


Each model has `infer` script, where there is a function to obtain inference from image. The script also contains commented out part, where predictions are generated one-by-one from test set and saved to `wtf-***.mat` (Please excuse me for such naming, but it's too late to change it. I was too surprised by some of the results, so I called data like these). The reason for writing such inefficient code is that I don't trust Keras generator for this task, as it often returns different results. You can repeat this computation yourself to ensure the results are correct. Also, there is a script for plotting results, which uses `.mat` file to output metrics and plot histograms, similar to the ones in the paper.


### Vanilla

The first model (**vanilla**) was trained in the same way original article suggests: 1) Input images were resized to 256x256, then randomly cropped to 224x224, random left-right flips were used as additional augmentation. Unlike article, this model architecture is MobileNetV2, Adam optimizer was used, and regularization terms were added to loss: MSE of the first and MSE of the second moment for predicted histograms.

The model was trained in the following highly manual way: batch size=16, 1 epoch pretrain, lr=1e-3 (base model frozen), 4 epochs lr=10-4, 12 epochs lr=5x10-6, 1 epoch 5x10-7. The key is that you need to watch model stopping to improve and overfitting, and switching to lower learning rate by breaking the script (Ctrl-C), and starting training with new settings.

Google result for vanilla setup (TABLE I):

```
Binary Accuracy (two classes) - 80.36-81.51%
LCC (Pearson correlation for predictions of mean) - 0.518-0.636
SRCC (Spearman correlation for means) - 0.510-0.612
LCC (std. devs) - 0.152-0.233
SRCC (std.devs) - 0.137-0.218
EMD L1 (Earth mover distance, difference of empirical CDFs for predicted and ground truth distributions) - 0.081-0.050
```

Google has also computed histograms, showing the distribution of mean values and std devs in comparison to the dataset:

![Original distribution](original.png?raw=true "Google result")



This model (`weights-vanilla-continue3-01-0.79.hdf5`):

```
accuracy:0.9135090313582956
Binary accuracy:0.7871849068420229
standard deviation of score differences:0.35308226304247403
LCC: (0.6559346351019297, 0.0)
SRCC: SpearmanrResult(correlation=0.6399734447510996, pvalue=0.0)
LCC (std dev): (0.30454632211568017, 0.0)
SRCC (std dev): SpearmanrResult(correlation=0.2880825105550821, pvalue=0.0)
EMD L1: 0.04845840864470296
```

Where accuracy is mean absolute percentage error (really useless metric for this task).

Training and measurements were made on the train-test split, provided by idealo : `ava_labels_train.json` and `ava_labels_test.json`. Test split is a validation split, so these results should not be considered too seriously.

Distribution of predictions:

![Vanilla](AVA/vanilla.png?raw=true "Vanilla result")

Here is confusion matrix for vanilla model, you can see that most of the classes are really messed up (see **Usage**):

```
[[   0    0    0    0    0    0    0    0    0    0]
 [   0    4    2    0    0    0    0    0    0    0]
 [   1   10   79   53    3    0    0    0    0    0]
 [   0   24  520 2668 1190   63    0    0    0    0]
 [   0    8  158 3857 9940 2457  100    0    0    0]
 [   0    0    6  152 1918 2103  228    2    0    0]
 [   0    0    0    0    0    1    1    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]]
```


### Patches model

This model was trained on random 224x224 patches, extracted from source images. Test/Validation was done on the central patch of each image of validation dataset. The example of usage for such a model can be found in [Jun-Ho Choi et al, Deep Learning-based Image Super-Resolution Considering Quantitative and Perceptual Quality](https://arxiv.org/abs/1809.04789).

Model used additional regularization of computing Pearson correlation between predicted and ground truth mean scores, which might be unnecessary and can actually lower results.

```
Got labels: 25428
[[    0     0     0     0     0     0     0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0]
 [    0     6    52    49     2     0     0     0     0     0]
 [    0    27   455  2594  1500   114     2     0     0     0]
 [    1    11   210  3989 10702  3381   181     0     0     0]
 [    0     0     2    57   810  1114   141     2     0     0]
 [    0     0     0     0     6    15     5     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0]]
accuracy:0.9112602801265431
Binary accuracy:0.7684442347019034
standard deviation of score differences:0.3711319098714664
LCC: (0.5882660857382291, 0.0)
SRCC: SpearmanrResult(correlation=0.5785882975473592, pvalue=0.0)
LCC (std dev): (0.21782951624048325, 9.182308742020182e-271)
SRCC (std dev): SpearmanrResult(correlation=0.21209535802882526, pvalue=1.7588084307929522e-256)
EMD L1: 0.05074453307535349
```

![Patches](AVA/patches.png?raw=true "Patches result")



## TID2013 model

[TID2013](http://www.ponomarenko.info/tid2013.htm) is a dataset for evaluating visual quality of the images. TID2013 contains 25 reference images and 3000 distorted images (25 reference images x 24 types of distortions x 5 levels of distortions). For each image there are values of mean score (float in range (0-9)] and standard deviation - however, the latter data is tricky. It is too small to be standard deviation for the histogram of scores with 10 bins 0-9, similar to AVA dataset: typical values there are `~0.15` which makes impossible to fit it using normal distribution (almost always nearly all the distribution mass will lie inside one bin). I've read original paper [Image database TID2013: Peculiarities, results and perspectives](http://www.ponomarenko.info/papers/tid2013.pdf) for any mentions that it is relative or normalized standard deviation (i.e. divided by mean), but the article states that it is RMSE of MOS (mean opinion scores). Google NIMA paper just tells that they approximated histograms with some maximum entropy distribution, without any clarification. Maximum entropy distribution on reals would be normal, if we have only mean and variance. For the range 1-10 we can also take normal, as tails would be negligible. Idealo in their realization approximate histograms with some maximum entropy model, which takes only MOS (mean) as input, making no use of these std dev values.

I tested three approaches: 1) histogram with 10 bins and std dev understood as relative (i.e. I multiplied it by the respective mean score), 2) Histogram with 1000 bins and standard deviation used as is, 3) Simple regression on means without using those strange MOS std dev values. First two models were worse than regression network in terms of correlation, so I left only regression model there. Why correlation is most important, you can read in **Usage**. Also, histogram models showed very high shift for me (good correlation, but big difference in mean and std dev), which could be partially eliminated by adding additional relu layer after softmax (regression, no negative bins). Yes, that was an ugly workaround, but I don't understand, why those models lacked linearity so much.

This model was trained on random 224x224 patches, because dataset is too small. Training was also quite manual, so you can refer to train script only to start with, then tweaking learning rate manually, watching performance. First 3 images with all their distortions were used as Test set, while remaining images were used for training. Also I added 1 more dense layer (features) before output.

Results:

```
Confusion matrix:
[[ 6  2  0  0  0  0  0  0  0  3]
 [ 1 21  1  0  0  0  0  0  0  1]
 [ 0 13 65 13  1  0  0  0  0  0]
 [ 0  1 13 55 28  3  0  0  0  0]
 [ 0  0  0 17 78 37  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  1]]
accuracy:0.8810525783161196
standard deviation of score differences:0.3366616895495652
LCC: (0.9205231242519071, 3.432749492185794e-148)
SRCC: SpearmanrResult(correlation=0.9020756542997307, pvalue=1.0480497320608934e-132)
```

![TID2013](TID2013_regression/Screenshot_20181227_001947.png?raw=true "tid result")


Google result:

![TID2013_google](TID2013_regression/paper.png?raw=true "tid google result")






## Comparison with Idealo


The results I received were quite poor on my opinion, so I compared them with available implementation from Idealo ([Link to the latest commit on that time](https://github.com/idealo/image-quality-assessment/commit/97576e330b014b174c3624e509215e634eba407f)). You can read our discussion [here](https://github.com/idealo/image-quality-assessment/issues/18).


To recompute comparison of results, you need to clone the Idealo repository, and put the scripts from `idealo_comparison` folder into it, uncommenting the part in `infer.py`. Here are the results for Idealo AVA model:


```
idealo MobileNet v1 model

Got labels: 25548
[[   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0   18  100   84    9    0    0    0    0    0]
 [   1   21  518 3027 1832  178    2    0    0    0]
 [   0    7  144 3472 9527 2660  129    0    0    0]
 [   0    0    3  147 1683 1786  198    2    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    0]]
accuracy:0.9111294033307482
Binary accuracy:0.7732112102708627
standard deviation of score differences:0.3676722622236499
LCC: (0.6129669517144579, 0.0)
SRCC: SpearmanrResult(correlation=0.5949598193491837, pvalue=0.0)
LCC (std dev): (0.17645952536758494, 9.437253434583301e-178)
SRCC (std dev): SpearmanrResult(correlation=0.17486969111266387, pvalue=1.4992237497262504e-174)
EMD L1: 0.051000147875872086
```

![idealo](idealo_comparison/ava.png?raw=true "idealo result")


Those look quite similar to mine and paper (correlation `~0.6`).



## Usage

When I attempted to use AVA models to evaluate visual quality of images, I've got quite garbage results (which is inevitable with correlation 0.6). This is clearly seen from confusion matrix. To understand what is wrong, I contacted authors and searched for another pretrained models. For the authors' response, the most relevant metric for this task is correlation. Which means, on my opinion, that those are exactly the results they've got. Predictions of this model cannot be used to evaluate images, but it can be a component of some ensemble. Moreover, there is another interesting thing you can do: compute cosine similarity for the outputs of global average pooling layer ("features"), and this metric might be more descriptive, as I will show below on the example of TID2013 model.

The reason why correlation is so low is that the dataset itself might be bad: even stronger models did not achieve better results. Also, correlation obtained for the model, trained on patches, is very close, which means that the model captures rather image quality features than content. It can be supported by the fact that model predicts variance of scores very poorly, which is covered in the original paper.

I did oversampling of the data with low and high mean scores, but got no improvement. These models were trained without oversampling.

As it was already mentioned, you can use `infer` scripts to compute output from AVA models for some images. To train those models, you can start with the train scripts I provided, but just as the beginning: I was starting them as `python3 -i ava_vanilla.py` and then manually stopped training with Ctrl-C and re-entered the lines with new learning rate, `model.compile` and `model.fit_generator`. The approximate scheme is: batch size=16, 1 epoch pretrain, lr=1e-3 (with base model frozen), then unfreeze all layers and train for 4 epochs lr=10-4, 12 epochs lr=5x10-6, 1 epoch 5x10-7. But you need to watch yourself.

As for TID2013 model, you can pretrain it for 3-4 epochs with high learning rate 10-3, or 10-4, and then train for 200-300 epochs with lr=10-5 or lower. It also showed some surprises. Having very high correlation, it was supposed to work very good - however, doing inference on some real-world images with different level of jpeg compression (quality 80, 70, 60, 40) revealed, that it sometimes outputs low scores for raw image, and higher scores for jpeg compression. You can check this behavior for both mine and idealo models using script `test_distortions.py`.

For the central patch from `res.jpg` it gives the following scores:

My model:

```
file:res.jpg, score:[[3.649583]]
file:res80.jpg, score:[[3.8724384]]
file:res70.jpg, score:[[3.8765335]]
file:res60.jpg, score:[[3.8891745]]
file:res40.jpg, score:[[3.74903]]
```

Idealo model:

```
file:res.jpg, score:[[4.790469]]
file:res80.jpg, score:[[4.9159007]]
file:res70.jpg, score:[[4.9890895]]
file:res60.jpg, score:[[4.975592]]
file:res40.jpg, score:[[4.967556]]
```


I've checked the scores for the images from the tid2013 dataset, they seem ok (images taken are `iXX_10_Y.bmp`, where 10 corresponds to jpeg compression distortion, and Y is the level of it):


```
Image i01, labels:[5.973 5.946 4.917 3.216 2.081], predictions:[5.444 5.27  5.001 3.159 2.45 ]
Image i02, labels:[6.    5.972 5.432 3.5   2.714], predictions:[5.32  5.366 5.157 4.338 2.609]
Image i03, labels:[6.237 5.667 5.051 3.3   2.487], predictions:[5.581 5.578 5.296 3.236 2.781]
Image i04, labels:[6.071 5.976 5.071 3.262 2.19 ], predictions:[6.084 5.914 5.323 3.759 2.191]
Image i05, labels:[5.732 5.154 4.707 3.439 1.878], predictions:[5.198 5.218 4.382 3.489 1.575]
Image i06, labels:[5.838 5.629 5.429 3.162 1.778], predictions:[5.548 5.501 5.335 3.48  2.125]
Image i07, labels:[6.133 5.886 5.045 3.705 2.227], predictions:[5.651 5.591 5.288 3.726 2.197]
Image i08, labels:[5.968 5.594 4.688 3.219 1.344], predictions:[5.412 5.377 4.935 3.188 1.676]
Image i09, labels:[6.    5.656 4.906 3.194 2.344], predictions:[5.678 5.609 5.01  3.462 2.277]
Image i10, labels:[5.868 5.816 5.205 3.641 2.436], predictions:[5.78  5.676 4.944 3.812 3.041]
Image i11, labels:[5.811 5.649 5.028 3.486 2.27 ], predictions:[5.646 5.53  5.096 3.644 2.32 ]
Image i12, labels:[6.081 5.895 5.105 3.105 2.237], predictions:[5.745 5.592 4.944 3.7   2.541]
Image i13, labels:[5.488 5.465 4.475 3.256 1.163], predictions:[5.051 5.121 4.562 3.301 1.534]
Image i14, labels:[5.778 5.167 4.25  3.056 1.861], predictions:[5.583 5.548 4.655 3.307 2.103]
Image i15, labels:[5.778 5.705 4.6   3.178 1.867], predictions:[5.879 5.793 4.772 3.325 2.129]
Image i16, labels:[5.9   5.769 5.256 3.615 2.462], predictions:[5.727 5.728 5.196 3.923 2.55 ]
Image i17, labels:[6.051 5.925 5.154 3.425 2.25 ], predictions:[5.676 5.679 5.023 3.607 2.612]
Image i18, labels:[6.073 5.78  4.675 3.5   2.171], predictions:[5.659 5.52  4.576 3.236 2.426]
Image i19, labels:[5.737 5.472 4.861 3.    1.658], predictions:[5.402 5.447 4.975 3.142 1.503]
Image i20, labels:[6.132 5.737 4.944 3.579 2.744], predictions:[5.697 5.608 5.343 3.513 2.899]
Image i21, labels:[6.31  5.881 5.14  3.262 1.927], predictions:[5.704 5.592 5.485 3.296 1.894]
Image i22, labels:[5.838 5.59  5.    3.59  2.079], predictions:[5.622 5.544 5.06  3.583 2.613]
Image i23, labels:[5.971 5.794 5.029 3.647 2.4  ], predictions:[5.859 5.735 5.248 3.713 2.828]
Image i24, labels:[5.886 5.829 4.657 3.278 1.853], predictions:[5.536 5.484 4.987 3.373 0.868]
Image i25, labels:[5.857 5.571 4.257 2.147 1.472], predictions:[5.392 5.232 3.987 2.098 1.58 ]
```

This means that TID2013 dataset is also not very useful, as models, trained on it, seem inadequate on unseen images. However, I found that cosine distance of the features from feature layer behaves more correctly for my model, see script `test_distortions_cosine.py`:

```
file:res.jpg, score:[[3.649583]], cos: 0.0
file:res80.jpg, score:[[3.8724384]], cos: 0.02377074956893921
file:res70.jpg, score:[[3.8765335]], cos: 0.0259554386138916
file:res60.jpg, score:[[3.8891745]], cos: 0.029344916343688965
file:res40.jpg, score:[[3.74903]], cos: 0.03298449516296387
```

```
file:0864_d2.png, score:[[2.9593334]], cos: 0.0
file:0864_d2_qf80.jpg, score:[[3.045652]], cos: 0.01521444320678711
file:0864_d2_qf70.jpg, score:[[3.082958]], cos: 0.022651493549346924
file:0864_d2_qf60.jpg, score:[[3.0735223]], cos: 0.021364212036132812
file:0864_d2_qf50.jpg, score:[[3.1644955]], cos: 0.02890557050704956
```

I did no such testing for idealo model, but I believe it will work as well. I suppose that's why additional losses work in the article [Jun-Ho Choi et al, Deep Learning-based Image Super-Resolution Considering Quantitative and Perceptual Quality](https://arxiv.org/abs/1809.04789).


Perhaps AVA models can be used in a similar way.



## What else you can try

You can try couple of losses I have in train script: chi-square loss, pearson correlation losses.


## Acknowledgments

This research was supported by:
[Let's Enhance](https://letsenhance.io)

We had discussions and I used Idealo models for comparison:
[Idealo](https://github.com/idealo/image-quality-assessment)



