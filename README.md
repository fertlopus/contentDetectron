### The intro and outro detection in video files.
The intro and outro detection solution is based on scientific papers 
on color spectrogram analysis and histogram comparison using the frame gradients searching 
and fourier transformations. The basis is the process of creating video features vectors and
forming it into features tensor. Under the hood there are 3 deep learning models are working:

- RMAC (<b>R</b>egional <b>Ma<b>ximum of <b>C</b>onvolutions) - the feature extractor for Keras that works with RoiPooling
methods.
- CNN (Convolutional Neural Network) - another feature extractor.
- CH (Color Histogram Gradient Comparison).

The goal of the algorithm is to detect introduction credits (known as intros) and ending credits (known as outro) 
from provided video file(-s) without using any of labeled data as algorithm works in fully unsupervised manner. This
can be used to automate the labeling for the skip functionality of as the backend service to cut the intros and outros
from provided video file(-s).


The research papers can be found in current repository under 
the directory <b> --> ./content_detectron/paper/ <b>

<hr>

### Requirements
Before usage install the requirements listed in the project root using command:

`
$ pip install -r -user requirements.txt
`

Moreover, make sure that <b>ffmpeg</b> are installed on your device and added to the <b>PATH</b> variable and tensorflow
GPU with CUDA are set up. 

Internal frameworks to set up:

- Facebook's FAISS
- Keras
- Tensorflow
- FFMPEG

<hr>

### Usage
Before usage make sure that the folder with video files exists and all files are in the same extensions (e.g. mp4).
The algorithm works best and detects intro and outro if it is run on a folder with a season for content of the same brand. 
In addition, it will be easier to detect the detection errors in the event of force majeure.

When running first time it creates feature mapping and feature vectors that mirth take some time. In case of reusage
it takes seconds to detect and form the output file.

<b> Command to run the Content Detectron </b>

`
$python3 main.py --video_dir './video/lbb_new_core_en/' --artifacts_dir './artifacts/'
`

Artifacts directory is the directory where the feature mappings, feature vectors and resized videos files are stored.
In case of improvement of the project the support of no-sql and server storage can be added.

The output of the detection is the <b>outputs.csv</b> file containing table data for intro detection and outro detection.
The structure of the output is listed below:


<table style="width:100%">

  <tr>
    <th>Filename</th>
    <th>Intro Timestamp array</th>
    <th>Outro Timestamp array</th>
  </tr>

  <tr>
    <td>E00017531.mp4</td>
    <td>(0.0, 4.48)</td>
    <td>null</td>
  </tr>

  <tr>
    <td>E00017630.mp4</td>
    <td>(0.0, 4.8)</td>
    <td>(19.84, 24.32)</td>
  </tr>

  <tr>
    <td>E00018051.mp4</td>
    <td>(0.0, 4.48)</td>
    <td>(126.4, 147.52)</td>
  </tr>

  <tr>
    <td>E00018395.mp4</td>
    <td>(0.0, 4.16)</td>
    <td>(122.24, 143.36)</td>
  </tr>

</table>


Score and testing results:
The current implementations works good but the accuracy is 87% and error 13% respectively.
<hr>
