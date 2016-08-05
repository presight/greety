# happy-cerberus

## Don't say hi to cerberus, it will say hi to you!
Happy cerberus is a joyful artificial guard with face detection that greets you when you come home and warn when intruders enter your home.

Train cerberus on faces you want it to recognize using openface/torch and it will play customized greetings when familliar faces are detected. Get notified when new faces appear.

git clone --recursive https://github.com/presight/happy-cerberus

or

cd openface
git submodule init
git submodule update

setup openface http://cmusatyalab.github.io/openface/setup/

 

Get dlib face landmarks
./openface/models/dlib/get-models.sh


http://cmusatyalab.github.io/openface/demo-3-classifier/


python ./openface/util/align_dlib_multiple.py {directory with list of photos} align outerEyesAndNose generated/test --size 96
python ./openface/util/align_dlib.py {...} align outerEyesAndNose generated/test --size 96

./openface/batch-represent/main.lua -outDir ./generated -data ./generated/normalized/

python openface/demos/classifier.py train ./generated/ --classifier RadialSvm