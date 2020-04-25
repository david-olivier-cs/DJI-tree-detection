# DJI tree trunk detection

I developed and presented this project for the ELE747 (Image Processing) course at ETS in Montreal. This is essentially a partial implementation of the system detailed in this conference paper : [Visual tree detection for autonomous navigation in forest environment](https://www.researchgate.net/publication/224329241_Visual_tree_detection_for_autonomous_navigation_in_forest_environment). It goes without saying that the research work of Thomas Hellström and Fredrik Georgsson made this project possible.

**The implemented system aims to identify tree trunks in images captured in a forest environment. Ultimately, this kind of system could be used to allow autonomous vehicles to navigate in a forest. To emphasize this use case, the recognition system was mounted on a [DJI Tello](https://store.dji.com/product/tello?vid=38421) drone.**

![](Docs/demo.gif)

This project also uses the code from this [repo](https://github.com/damiafuentes/DJITelloPy) to interface with the DJI Tello drone.

## Project Report

The project guidelines given to the students were the following : 
1. Find a research article covering a topic related to the field of image processing
2. Fully or partially implement the work detailed in the article
3. If necessary, make changes to simplify or modernize the design

**The full project report (in french) can be found [here](Docs/Rapport.pdf) and the presentation can be found [here](docs/Presentation.pptx)**

## System Overview

**The following is an overview of the processing steps taken to identify tree trunks in forest images.**

![](Docs/pipeline.png)

1. Resizing the input image and dividing it in do multiple independent sub-images. Each sub-image will separately go through the remaining parts of the processing pipeline to receive its own classification label.
2. Extraction of texture and color features from each sub-image.
3. Joining the texture and color features to obtain a single feature vector for each sub-image.
4. Normalization of the feature vector associated with each sub-image.
5. Classification of the feature vectors and associating a label to each sub-image.

Once all the sub-images have been classified, the complete image is reassembled and colored bounding boxes are placed over the sub-images classified as tree trunk segments.

**Note that there are only two possible labels**
* Background
* Tree trunk