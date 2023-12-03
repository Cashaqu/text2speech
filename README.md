# text2speech

The project "text2speech" is based on https://github.com/coqui-ai/TTS

```vits_train.py``` in this project - modified file from https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/vits_tts/train_vits.py
```dataset_preparation.ipynb``` - processing dataset for training

The trained model and config file: [gdrive](https://drive.google.com/file/d/1ADS4K64_znBYanfXLUY2nsOTbDlBWTpY/view?usp=sharing)

To run the model:
```sh
python -m pip install TTS
tts-server --model_path path/to/model.pth --config_path path/to/config.json
```
and go to the address ```http://[::1]:5002``` in your browser. 
## Original sample:

[sample_original.webm](https://github.com/Cashaqu/text2speech/assets/131269265/1df67ca8-b604-45fc-b6db-df0f89265eb7)


## Generated sample:

[sample_generated.webm](https://github.com/Cashaqu/text2speech/assets/131269265/4cffb84d-dd27-43f2-ad27-565b25478db2)
