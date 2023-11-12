# text2speech

The project "text2speech" is based on https://github.com/coqui-ai/TTS

```vits_train.py``` in this project - modified file from https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/vits_tts/train_vits.py
```dataset_preparation.ipynb``` - processing dataset for training

The trained model will be available via the link:
link will be added later

To run the model:
```sh
python -m pip install TTS
tts-server --model_path path/to/model.pth --config_path path/to/config.json
```
and go to the address ```http://[::1]:5002``` in your browser. 
