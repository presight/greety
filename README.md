# Greety, your personal welcome home greeter

Greety recognizes persons appearing in front of the camera and greets them with a joyful message. 

Just train greety on faces you want it to recognize, attach a camera and let the magic begin.


## Usage

### Get code with dependencies
Start by cloning the project with the openface subproject
`git clone --recursive https://github.com/presight/greety`

Setup openface as described at http://cmusatyalab.github.io/openface/setup/

Get dlib face landmarks, which are used for detecting faces
`./openface/models/dlib/get-models.sh`



#http://cmusatyalab.github.io/openface/demo-3-classifier/

### Extract features

The first step is to collect images of faces you want greety to recognize in a folder, let's call it `{dir}`.

#### Multiple faces per image
If you have a set of uncategorized images and want to extract as many faces as posible, this command will extract and align all face images found in the images and put them in `generated/aligned/` 

`python ./openface/util/align_dlib_multiple.py {dir} align outerEyesAndNose generated/aligned --size 96`

After the aligned images have been generated you have to move the images to sub folders named after the desired labels, for example `{dir}/person1`.

#### One face per image
Put all images in sub directories for each person, for example `{dir}/person1` etc, then run the following command: 

`python ./openface/util/align_dlib.py {dir} align outerEyesAndNose generated/aligned --size 96`

#### Generate images from webcam 
Run `feature_saver.py` and all found faces will be saved in `generated/unknown`. Remove bad images and move them to sub directories in `{dir}` corresponding to their labels.

### Generate face representations
First remove the cache from eventual previous runs
`rm generated/aligned/cache.t7` 

Generate face representations
`./openface/batch-represent/main.lua -outDir ./generated -data ./generated/aligned/`

### Train the classifier
`python openface/demos/classifier.py train ./generated/ --classifier RadialSvm`

### Run greety! ###
`python greety.py`

### Adapt parameters ###
Dive into greety.py and change `video_capture_device`, `person_confidence_threshold` and other parameters to get it running smoothly

## Todo ##
* Max saved images per minute? or session? or max size of unknown folder?