# BlendPASS - Blending Panoramic Amodal Seamless Segmentation

## BlendPASS
Blending Panoramic Amodal Seamless Segmentation (BlendPASS) comprises an unlabeled training set of 2,000 panoramic images and a labeled test set of 100 panoramic images.
These images are captured from panoramic cameras in driving scenes across 40 cities for the training set and 20 cities for the test set, all at a resolution of 2048X400 pixels. 

We provide finely pixel-level annotations for five segmentation tasks (Semantic, Instance, Panoptic, Amodal Instance and Amodal Panoptic) related to OASS, which greatly extends the semantic labels from DensePASS. It is available at [Google Drive](https://drive.google.com/drive/folders/1t-dUjH4zeu4fBhtr6AbKULjnbSWKje0S?usp=sharing).

These annotations cover 19 categories that align with the Cityscapes and are further categorized into _Stuff_ (_road, sidewalk, building, wall, fence, pole, light, sign, vegetation, terrain, and sky_) and _Thing_ (_person, rider, car, truck, bus, train, motorcycle, and bicycle_).

|                   | Person | Rider | Car  | Truck | Bus | Train | Motorcycle | Bicycle | Total |
|:-------------------:|:------:|:-----:|:----:|:-----:|:---:|:-----:|:----------:|:-------:|:-----:|
|  #Occluded objects  |  189   |   6   | 909  |  42   | 18  |   1   |     83     |   38    | 1286  |
| #Unoccluded objects |  613   |  12   | 842  |  38   | 24  |   2   |     71     |   72    | 1674  |
|        Total        |  802   |  18   | 1751 |  80   | 42  |   3   |    154     |   110   | 2960  |

## visualization
We provide some code to visualize our dataset.
We provid some code to Visualize BlendPASS, please refer to the [Visualize](Visualize) folder.

[//]: # (## Evaluation)

[//]: # (TODO)

[//]: # (## Annotation)

[//]: # (TODO)
