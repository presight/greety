# Greety, your personal welcome home greeter

Greety recognizes persons appearing in front of the camera and greets them with a joyful message. 
Just train greety on faces you want it to recognize, attach a camera and let the magic begin.


## Usage

### Get code and dependencies
Start by cloning the project with the openface subproject
`git clone --recursive https://github.com/presight/greety`

Setup openface as described at http://cmusatyalab.github.io/openface/setup/

Get dlib face landmarks, which are used for detecting faces
`./openface/models/dlib/get-models.sh`


### Extract features

The first step is to collect images of faces you want greety to recognize in a folder, let's call it `{dir}`.

#### Multiple faces per image
If you have a set of uncategorized images and want to extract as many faces as possible, this command will extract and align all face images found in the images and put them in `generated/aligned/` 

`python ./align_dlib_multiple.py {dir} align outerEyesAndNose generated/aligned --size 96`

After the aligned images have been generated you have to move the images to sub folders named after the desired labels, for example `{dir}/person1`.

#### One face per image
Put all images in sub directories for each person, for example `{dir}/person1` etc, then run the following command: 

`python ./openface/util/align_dlib.py {dir} align outerEyesAndNose generated/aligned --size 96`

#### Generate images from webcam 
Run `feature_saver.py` and all found faces will be saved in `generated/unknown`. Remove bad images and move the remaining images to sub directories in `{dir}` corresponding to their labels.


### Generate face representations
First remove the cache from eventual previous runs
`rm generated/aligned/cache.t7` 

Generate face representations
`./openface/batch-represent/main.lua -outDir ./generated -data ./generated/aligned/`


### Train the classifier
Train greety classificator, optionaly with a specified conf file
`python train.py` or `python train.py default.conf`


### Run greety! ###
Run greety, optionaly with a specified conf file
`python greety.py` or `python greety.py default.conf`


### [Optional] Generate unknown face embeddings ###
Download a set of unknown images not included in the dataset to learn, for example a subset of http://vis-www.cs.umass.edu/lfw/.
Generate `./generated/unknown.npy` from the images in {lfw_directory}. The resulting file will contain face embeddings that will represent the unknown faces.
./openface/demos/web/create-unknown-vectors.py --outputFile ./generated/unknown.npy --dlibFacePredictor ./openface/models/dlib/shape_predictor_68_face_landmarks.dat --model ./openface/models/openface/nn4.small2.v1.t7 {lfw_directory}`

Also make sure `unknown_reps` in the config point to the generated `unknown.npy` file


## [Optional] Text to speech ##
To get text to voice synthesizing working you can install [espeak](http://espeak.sourceforge.net/) or [marytts](http://mary.dfki.de/), or implement support for a lib of your choosing.


## Limitations ##
So far openface hasn't reached a good decision on how to classify unknown faces, and thus the current system with DBN generates some false positives.