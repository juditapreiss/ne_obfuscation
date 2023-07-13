import os
import jiwer
import torch
import argparse
import logging
import tempfile
import subprocess
import pandas as pd
import audio2numpy as a2n
from datasets import Dataset
from datasets import logging as datasets_logging
from datasets import config as datasets_config
from datasets.utils import logging as datasets_utils_logging
from transformers import Wav2Vec2Processor
from transformers import HubertForCTC
from transformers import logging as transformer_logging
from model_align import align_and_mask
from substitute_names import mask_and_reconstruct_sentence
from constants import classifier_model
from constants import sampling_rate
from substitute_audio import obfuscate_audio


logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformer_logging.set_verbosity_error()
datasets_logging.set_verbosity_error()
datasets_logging.disable_progress_bar()
datasets_utils_logging.set_verbosity_error()

datasets_config.HF_DATASETS_OFFLINE = True


parser = argparse.ArgumentParser(description="Obfuscation pipeline for speech inputs.")
parser.add_argument('--asr_dir', metavar='DIR', dest='asr_dir', required=True, help='Path to the ASR model.')
parser.add_argument('--ner_dir', metavar='DIR', dest='ner_dir', required=True, help='Path to the NER model.')
parser.add_argument('--input_wav', metavar='FILEPATH', dest='wav_file', required=True, help='Path to the wav file to process.')
parser.add_argument('--masking_model', type=str, dest='masking_model', default='albert', help='Masking model to use for obfuscation (albert|bert|roberta).')
parser.add_argument('--first_name_file', type=str, dest='first_name_file', required=True, help='Path to a file of first names, one name per line.')
parser.add_argument('--last_name_file', type=str, dest='last_name_file', required=True, help='Path to a file of last names, one name per line.')
parser.add_argument('--top_k', type=int, dest='top_k', default=20, help='Top number of masked model suggestions.')
parser.add_argument('--reference_wav', type=str, dest='reference_wav', required=True, help='Path to reference wav.')
parser.add_argument('--output_wav', type=str, dest='output_wav', required=True, help='Path to obfuscated output wav.')


def extract_input_values(audio_path, processor):
    if not os.path.exists(audio_path):
        return None
    signal, sr = a2n.audio_from_file(audio_path)
    assert sr == sampling_rate
    return processor(signal, sampling_rate=sampling_rate).input_values[0]


def transcript_generator(wav_file, processor):
    input_values = extract_input_values(wav_file, processor)
    yield {'input_values': input_values}


def map_to_result(batch, model, processor):
    with torch.no_grad():
        #input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(pred_ids, output_word_offsets=True, output_char_offsets=False)
    batch["pred_str"] = predictions['text'][0]
    batch["pred_word_offsets"] = predictions['word_offsets'][0]
    #batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch
    

def transcribe_file(asr_dir, wav_file, asr_file):
    processor = Wav2Vec2Processor.from_pretrained(asr_dir)
    model = HubertForCTC.from_pretrained(asr_dir)
    logger.debug('Loaded Wav2Vec2Processor and ASR HubertForCTC model: %s' % asr_dir)

    testDS = Dataset.from_generator(transcript_generator, gen_kwargs={'wav_file': wav_file, 'processor': processor})
    logger.debug('Created Huggingface dataset from example: %s' % testDS)

    results = testDS.map(map_to_result, fn_kwargs = {"model": model, "processor": processor})
    logger.debug("Results: %s" % results)
    results.to_csv(asr_file)
    logger.debug('Saved ASR to: %s' % asr_file)
    
    
def recognition_file(ner_dir, wav_file, ner_file):
    processor = Wav2Vec2Processor.from_pretrained(ner_dir)
    model = HubertForCTC.from_pretrained(ner_dir)
    logger.debug('Loaded Wav2Vec2Processor and NER HubertForCTC model: %s' % ner_dir)
    
    testDS = Dataset.from_generator(transcript_generator, gen_kwargs={'wav_file': wav_file, 'processor': processor})
    logger.debug('Created Huggingface dataset from example: %s' % testDS)

    results = testDS.map(map_to_result, fn_kwargs = {"model": model, "processor": processor})
    logger.debug("Results: %s" % results)
    results.to_csv(ner_file)
    logger.debug('Saved NER to: %s' % ner_file)


def asr_ner_align(asr_file, ner_file, masking_model):

    nerDF = pd.read_csv(ner_file, index_col=0).rename_axis('id').reset_index()
    logger.debug(nerDF[['id', 'pred_word_offsets']].head())

    asrDF = pd.read_csv(asr_file, index_col=0).rename_axis('id').reset_index()
    logger.debug(asrDF[['id', 'pred_word_offsets']].head())

    mergedDF = pd.merge(nerDF.rename(columns={'pred_word_offsets': 'ne_offsets', 'text': 'ne_text'}), asrDF.rename(columns={'pred_word_offsets': 'asr_offsets', 'text': 'asr_text'}), on='id', how='inner')
    logger.debug('mergedDF size: %s' % str(mergedDF.shape))

    mergedDF['asr_sent'], mergedDF['masked'], mergedDF['originals'] = zip(*mergedDF.apply(lambda x: align_and_mask(x, masking_model), axis=1))
    logger.debug('Merged DF: %s' % mergedDF)

    # This is only one sentence
    assert mergedDF.shape[0] == 1
    
    return mergedDF['asr_sent'].iloc[0], mergedDF['masked'].iloc[0], mergedDF['originals'].iloc[0]
    

def convert_text_to_speech(text, reference_wav, output_file):
    command = ["tts", "--text", text, "--model_name", "tts_models/multilingual/multi-dataset/your_tts", "--speaker_wav", reference_wav, "--out_path", output_file, "--language_idx", "en"]
    #logger.debug(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    process.wait()
    

def extract_text_from_asr(filepath):
    asrDF = pd.read_csv(filepath, index_col=0).rename_axis('id').reset_index()
    logger.debug(asrDF[['id', 'pred_str']].head())

    assert asrDF.shape[0] == 1
    asr_text = asrDF['pred_str'].iloc[0]

    return asr_text


def get_ne_time_offset(ner_dir):
    processor = Wav2Vec2Processor.from_pretrained(ner_dir)
    model = HubertForCTC.from_pretrained(ner_dir)
    ne_time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
    return ne_time_offset


def main(ner_dir, asr_dir, wav_file, masking_model, first_name_file, last_name_file, top_k, reference_wav, output_wav):
    # Extract transcriptions from audio
    asr_handle_1 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    asr_file_1 = asr_handle_1.name
    asr_handle_1.close()
    transcribe_file(asr_dir, wav_file, asr_file_1)
    logger.debug('Saved transcription1 to: %s' % asr_file_1)
    
    # Extract NEs from audio
    ner_handle_1 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    ner_file_1 = ner_handle_1.name
    ner_handle_1.close()
    recognition_file(ner_dir, wav_file, ner_file_1)
    logger.debug('Saved NER1 to: %s' % ner_file_1)
    
    # Construct masked sentence
    asr_sent, masked_sent, original_values = asr_ner_align(asr_file_1, ner_file_1, masking_model)
    logger.info('ASR sentence: %s' % asr_sent)
    logger.debug('Masked sentence: %s' % masked_sent)
    logger.debug('Original values: %s' % original_values)
    
    # Use masking model to suggest name substitutions
    substituted_sent, reconstructed_sent = mask_and_reconstruct_sentence(masking_model, asr_sent, masked_sent, original_values, first_name_file, last_name_file, top_k)
    logger.debug('Substitute(s): %s' % substituted_sent)
    logger.info('Reconstructed sentence: %s' % reconstructed_sent)
    
    # Generate reconstructed sentence
    generated_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    generated_file = generated_handle.name
    generated_handle.close()
    convert_text_to_speech(reconstructed_sent, reference_wav, generated_file)
    logger.debug('Converted text to speech is in: %s' % generated_file)

    # Run ASR and NER on the generated reconstructed sentence
    asr_handle_2 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    asr_file_2 = asr_handle_2.name
    asr_handle_2.close()
    transcribe_file(asr_dir, generated_file, asr_file_2)
    logger.debug('Saved transcription2 to: %s' % asr_file_2)
    
    ner_handle_2 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    ner_file_2 = ner_handle_2.name
    ner_handle_2.close()
    recognition_file(ner_dir, generated_file, ner_file_2)
    logger.debug('Saved NER2 to: %s' % ner_file_2)

    # Reconstruct audio
    ne_time_offset = get_ne_time_offset(ner_dir)
    obfuscate_audio(wav_file, ner_file_1, generated_file, ner_file_2, output_wav, ne_time_offset)
    
    # Report confidence in obfuscated output
    asr_handle_3 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    asr_file_3 = asr_handle_3.name
    asr_handle_3.close()
    transcribe_file(asr_dir, output_wav, asr_file_3)
    logger.debug('Saved ASR3 to: %s' % asr_file_3)

    # Find error rates against the reconstructed sentence
    text = extract_text_from_asr(asr_file_3)
    wer = jiwer.wer(text, reconstructed_sent)
    cer = jiwer.cer(text, reconstructed_sent)

    logger.info('WER: %s' % wer)
    logger.info('CER: %s' % cer)
    
    # Remove temporary files
    os.unlink(asr_file_1)
    os.unlink(asr_file_2)
    os.unlink(asr_file_3)
    os.unlink(ner_file_1)
    os.unlink(ner_file_2)
    os.unlink(generated_file)


if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()

    # Check parameters
    assert args.masking_model in classifier_model
    assert os.path.exists(args.ner_dir)
    assert os.path.exists(args.asr_dir)
    assert os.path.exists(args.wav_file)
    assert args.wav_file.endswith(".wav")
    assert os.path.exists(args.first_name_file)
    assert os.path.exists(args.last_name_file)
    assert os.path.exists(args.reference_wav)

    main(args.ner_dir, args.asr_dir, args.wav_file, args.masking_model, args.first_name_file, args.last_name_file, args.top_k, args.reference_wav, args.output_wav)
