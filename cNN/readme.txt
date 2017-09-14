1. setup the environment
   open terminal to run below command
   if use python >=3.0
      sudo pip3 install -r requirements.txt
   else
      sudo pip install -r requirements.txt

2. download UrbanSound8K dataset from internet
   https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html
   

3. change dataset directory according to your environment (constant.py)
   DATA_AUDIO_DIR = '/home/kay/UrbanSound8K/audio' >> the dataset directory
   OUTPUT_DIR = '/home/kay/raw-waveforms' >> the dataset save directory after process

4. run process_data.py to process data

5. run model_run.py to train model


   

