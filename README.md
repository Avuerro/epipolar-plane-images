# epipolar-plane-imaging

This repo contains the code that can be used to create Epipolar Plane Images (EPI) from videos or arrays of images.

An EPI can be used to analyze the depth or multi-view consistency of objects in a scene. 
It is most often used for depth analysis but it originated in a paper by Bolles et al. where they used it to perform gait analysis. 
The title of the paper is **Epipolar-Plane Image Analysis: An Approach to Determining Structure
from Motion**.

In essence an EPI can be constructed by moving the camera along the `X-axis` of a static scene and taking snapshots at certain moments in time. 
We stack these images and construct a volume which we slice through and inspect the top.
The top will show diagonal lines where we can distinguish the objects due to the differences in colors of the objects.
The slope of these lines provides information on the location of the object with respect to the camera, i.e. depth.

A very good but concise explanation can be found at [the following link](https://www.youtube.com/watch?v=1F_5c_escis).


## Constructing an EPI from video
In the notebook `epipolar_plane_image` you can find an example of how to use the code to construct an EPI. 

## Constructing an EPI from an array of images
The EpipolarPlaneImage class contains a classmethod `load_from_array` which can be used to construct an EPI from an array of images. 


The video in the assets folder are from [pixabay](https://pixabay.com/videos/liverpool-anchor-pier-head-england-46108/) and were created by Paul Daley (free for commercial use).

### Feedback
Please feel free to let me know if you have found some errors, bugs or possible improvements!