import re
import logging
import constants
import pandas as pd
from pydub import AudioSegment

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_starts_and_ends_from_row(row):
    # Set up memory for where to insert new portion of speech
    continuing_ne = False
    ne_starts = []
    ne_ends = []
    # Strip brackets
    offset_lists = row[1:-1].split('\n')
    for offset_list in offset_lists:
        match = re.match("^[ ]*{'end_offset': ([0-9]+), 'start_offset': ([0-9]+), 'word': ['\"]([^ ]+)['\"]}[ ]*$", offset_list)
        assert match
        end_offset = int(match.group(1))
        start_offset = int(match.group(2))
        word = match.group(3)
        if word.startswith('n'):
            if not continuing_ne:
                ne_starts.append(start_offset)
            continuing_ne = True
            last_end = end_offset
        else:
            if continuing_ne:
                ne_ends.append(last_end)
                continuing_ne = False
    if continuing_ne:
        # Got to end without going past end of NE
        ne_ends.append(last_end)
    # Lists
    return ne_starts, ne_ends


# Change volumes
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def join_ner_annotations(ner_file_1, ner_file_2):
    ne1DF = pd.read_csv(ner_file_1, index_col=0).rename(columns={'pred_word_offsets': 'first_word_offsets', 'audio_file': 'first_audio_file', 'pred_str': 'first_text'})
    ne2DF = pd.read_csv(ner_file_2, index_col=0).rename(columns={'pred_word_offsets': 'second_word_offsets', 'audio_file': 'second_audio_file', 'pred_str': 'second_text'})
    
    assert ne1DF.shape[0] == 1
    assert ne2DF.shape[0] == 1

    concatDF = pd.concat([ne1DF, ne2DF], axis=1)
    
    return concatDF['first_word_offsets'].iloc[0], concatDF['second_word_offsets'].iloc[0]


def obfuscate_audio(audio_file_1, ner_file_1, audio_file_2, ner_file_2, obfuscated_file, ne_time_offset):

    # Extract words offsets from both versions
    first_word_offsets, second_word_offsets = join_ner_annotations(ner_file_1, ner_file_2)

    # Assumes second NE is in the same position as in first
    first_ne_starts, first_ne_ends = extract_starts_and_ends_from_row(first_word_offsets)
    second_ne_starts, second_ne_ends = extract_starts_and_ends_from_row(second_word_offsets)
    if len(first_ne_starts) != len(second_ne_starts):
        logger.warning('CANNOT obfuscate: %s vs %s' % (first_ne_starts, second_ne_starts))
        return

    # Reconstruct audio
    reconstructed_audio = AudioSegment.empty()

    # Normalize second audio to match loudness of first    
    first_audio = AudioSegment.from_wav(audio_file_1)
    first_loudness = first_audio.dBFS
    unnorm_second_audio = AudioSegment.from_wav(audio_file_2)
    second_audio = match_target_amplitude(unnorm_second_audio, first_loudness)
    
    # Insert NEs in original audio
    first_start = 0
    for i in range(len(first_ne_starts)):
        first_end = round(first_ne_starts[i] * ne_time_offset, 2) * 1000
        reconstructed_audio += first_audio[first_start:first_end]
        first_start = round(first_ne_ends[i] * ne_time_offset, 2) * 1000
        second_start = round(second_ne_starts[i] * ne_time_offset, 2) * 1000
        second_end = round(second_ne_ends[i] * ne_time_offset, 2) * 1000
        reconstructed_audio += second_audio[second_start:second_end]
    reconstructed_audio += first_audio[first_start:]

    # Save obfuscated audio to new file
    reconstructed_audio.export(obfuscated_file, format="wav")
    logger.debug('Saved obfuscated file to: %s' % obfuscated_file)


