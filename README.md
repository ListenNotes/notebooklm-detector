# NotebookLM Detector

This is a simple tool to detect if an audio file is generated by NotebookLM.

## Detect

Install dependencies first:

```shell
$ pip install -r requirements.txt
```

Run the script to detect:
```shell
$ python notebooklm_detector --action predict --file_path [filename].mp3
```

## Train

You can train and regenerate model.pkl:

Step 1: Put NotebookLM-generated audio files (mp3, wav, or mp4) in datasets/ai/ folder. 
And put human-produced audio files in datasets/human/ folder.  

Step 2: Run the script to train:
```shell
$ python notebooklm_detector --action train --dataset_path datasets
```