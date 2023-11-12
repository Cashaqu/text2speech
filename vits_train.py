import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Dataset path
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.txt", path=os.path.join(output_path, "data/")
)

# Define audio configuration
audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

# Define phonemes
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ͡"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

# Define alphabet and punctuation
character_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="!¡'(),-—.:;¿?АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя «°±µ»$%&‘’“`”„",
        punctuations="!¡'(),-—.:;¿? ",
        phonemes=_phonemes
   )

# Define VITS config
config = VitsConfig(
    audio=audio_config,
    characters=character_config,
    run_name="vits_BT_ru",
    batch_size=24,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="ru-ru",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    save_best_after=1000,
    save_checkpoints=True,
    save_all_best=True,
    mixed_precision=True,
    max_text_len=250,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences=[
        ["Съешь ещё этих мягких французских булочек да выпей чаю."],
        ["Безмозглый широковещательный цифровой передатчик сужающихся экспонент."],
        ["Флегматичная эта верблюдица жуёт у подъезда засыхающий горький шиповник."]
    ]
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

# Define custom format for dataset
def formatter(root_path, manifest_file, **kwargs):
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path + '/wavs/', cols[0] + '.wav')
            text = cols[1]
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items

# train/eval
train_samples, eval_samples = load_tts_samples(
    [dataset_config],
    eval_split=True,
    formatter=formatter
)

# Define model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# Training config
trainer = Trainer(
    TrainerArgs(continue_path='./vits_BT_ru-November-10-2023_05+05PM-0000000'),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
