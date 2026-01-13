# ISSLIDE: A new InSAR dataset for Slow SLIding area DEtection with machine learning

This repository is dedicated to the generation and use of the ISSLIDE dataset targeting slow sliding area detection using machine learning approaches.
We here release the three steps of the generation of ISSLIDE :

* Interferogram generation;
* Moving areas patch extraction;
* Segmentation with standard methods from the literature.

Each step is detailed hereinafter. The codes are were running with Python 3.9.6 with the environnment detailed in the requirement.txt file. Please note that the librairy GDAL requires that you first install GDAL on your machine (https://gdal.org/en/stable/download.html).

The ISSLIDE dataset is available on [IEEEDataport](https://ieee-dataport.org/documents/isslide-insar-dataset-slow-sliding-area-detection-machine-learning).

One can find the paper relative to this dataset in [ISSLIDE: A new InSAR dataset for Slow SLIding area DEtection with machine learning]()

# Interferogram Generation

The generation of the interferograms is three steps:

## Combining SAR images acquired at the same date

For large studied areas as in the dataset, it can be necessary to combine several SAR images before generating interferogras. For memory issues, the combined images are divided by swaths and combined.

Please make sure that images acquired at the same date are named in a way that they appear successively when sorted by alphabetic order.

Ensure to complete "User Parameters" in the file - specifying "MergeImages.xml" as graph - and run the following line: 

```python
 python3 ./InterferogramsGeneration/CombineSwath.py 
 ```

## Generate raw interferograms in radar geometry

Interfergram generation is multiple steps and is done swath-wise for memory issues. One should consider to use "InterfGraph.xml" or "InterfGraphSingleSwath.xml" depending on the input images - swaths combined or separated respectively. 

Ensure to complete "User Parameters" in the file and run the following line: 

```python
 python3 ./InterferogramsGeneration/GenerateInterf.py 
 ```

Resulting interferograms are in radar geometry.

## Orthorectify interferograms to transfer them into ground geometry

To project interferograms into ground geometry, we also provide an orthorectification algorithm. 

Ensure to complete "User Parameters" in the file - specifying "OrthorecGraph.xml" as graph - and run the following line: 

```python
 python3 ./InterferogramsGeneration/OrthorectInterf.py
 ```

# Moving Areas Extraction

Based on the provided shapefiles, we propose a method to extract identified moving areas. This is the strategy applied to generate the "Ready to be used" dataset in [ISSLIDE](https://ieee-dataport.org/documents/isslide-insar-dataset-slow-sliding-area-detection-machine-learning). 

Ensure to complete "User Parameters" in the main file and run the following line: 

```python
 python3 ./MovingAreasExtraction/main.py 
 ```

# Deep Learning Baselines

Based on the experiments led in the paper, we propose our training strategy. Multiple networks have been tested: 

* FCN [Long et al, 2015]
* DeepLabV3 [Chen et al, 2017]
* UNet [Ronneberger et al, 2015]
* ResUNet [He et al, 2016]
* SepUNet [Chollet et al, 2017]

The three last networks are smaller networks which are inspired by the given references. 

Ensure to complete "User Parameters" in the main file and run the following line: 

```python
 python3 ./BaselineNetworks/main.py 
 ```

# Citation

If this work was useful for you, please ensure citing our works :

<i> ISSLIDE: A New InSAR Dataset for Slow SLIding Area DEtection With Machine Learning, Bralet, A., Trouvé, E., Chanussot, J., & Atto, A. M., in IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1-5, 2024, Art no. 3001005, doi: 10.1109/LGRS.2024.3365299.</i>

Thank you for your support

# Any troubles ?

If you have any troubles with the article or the code, do not hesitate to contact us !

# References
[Long et al, 2015] LONG, Jonathan, SHELHAMER, Evan, et DARRELL, Trevor. Fully convolutional networks for semantic segmentation. In : Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. p. 3431-3440.

[Chen et al, 2017] CHEN, Liang-Chieh, PAPANDREOU, George, SCHROFF, Florian, et al. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.

[Ronneberger et al, 2015] RONNEBERGER, Olaf, FISCHER, Philipp, et BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In : Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015. p. 234-241.

[He et al, 2016] HE, Kaiming, ZHANG, Xiangyu, REN, Shaoqing, et al. Deep residual learning for image recognition. In : Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 770-778.

[Chollet et al, 2017] CHOLLET, François. Xception: Deep learning with depthwise separable convolutions. In : Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. p. 1251-1258.
