import os
import re
import json
import torch
import logging
import argparse
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from dataclasses import dataclass
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import HubertForCTC
from transformers import TrainingArguments
from transformers import Trainer
from typing import Dict, List, Union
from sklearn.model_selection import train_test_split
from constants import sampling_rate

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Builds a HuBERT model for either transcripts or NER (depending on input) from the English speech NER dataset.")
parser.add_argument('--array_dir', metavar='DIR_PATH', dest='array_dir', required=True, help='Path to the array directory')
parser.add_argument('--text_dir', metavar='DIR_PATH', dest='txt_dir', required=True, help='Path to the txt directory')
parser.add_argument('--out_dir', metavar='DIR_PATH', dest='output_dir', required=True, help='Path to output directory')
parser.add_argument('--model', metavar='model', dest='hubert_model', default='facebook/hubert-base-ls960', help='Model to use')
parser.add_argument('--type', metavar='type', dest='input_type', default='asr', help='Classification type: asr or ner')
parser.add_argument('-n', '--num_epochs', type=int, dest='num_epochs', default=3, help='Number of epochs')
parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=12, help='Batch size')
parser.add_argument('--train_proportion', type=int, dest='train_proportion', default=60, help='Percentage of the data to be retained for training')
parser.add_argument('--val_proportion', type=int, dest='val_and_test_proportion', default=50, help='Percentage of the val/test portion of the data to be used for validation')
parser.add_argument('--random', type=int, dest='random_state', default=3, help='Default value of random state (for repeating experiments)')

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\{\|\$\]]'

if torch.cuda.is_available():
    fp16 = True
    device = 'cuda'
else:
    fp16 = False
    device = 'cpu'

# Dataset input assumptions:
# Needs to have test_ids

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    # Class from huggingface.com
        
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def create_train_validation_test_split(array_dir, txt_dir, random_state):
    array_ids = list(map(lambda x: x[:-7], os.listdir(array_dir)))
    text_ids = list(map(lambda x: x[:-4], os.listdir(txt_dir)))
    # Overlap the two lists
    overlap_ids = list(set(array_ids).intersection(text_ids))
    # Total number of examples we actually have
    total_examples = len(overlap_ids)
    logger.debug('Total examples: %d' % total_examples)
    # Create splits
    fraction = val_and_test_proportion / 100
    train_ids, test_ids = train_test_split(overlap_ids[:total_examples], test_size=fraction, random_state=random_state)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=random_state)
    logger.debug('Training data size: %s' % len(train_ids))
    logger.debug('Validation data size: %s' % len(val_ids))
    logger.debug('Test data size: %d' % len(test_ids))
    return train_ids, val_ids, test_ids


def remove_punctuation(original):
    text = re.sub(chars_to_ignore_regex, '', original).lower() + " "
    return text


def convert_to_io_chars(original):
    inside = 0
    text = ''
    for letter in original:
        if letter == '|' and inside == 0:
            inside = 1
            character = 'n'
        elif letter == '$' and inside == 0:
            inside = 1
            character = 'l'
        elif letter == "{" and inside == 0:
            inside = 1
            character = 'e'
        elif letter == ']' and inside:
            inside = 0
        elif letter == ' ':
            if inside == 1:
                text += character
            else:
                text += ' '    
        else:
            if inside:
                text += character
            else:
                text += 'o'
                
    return text.strip()


def create_ner_vocab_dict(vocab_file):
    # Fixed by substitutions in convert_to_io_chars so manually created
    if not os.path.exists(vocab_file):
        vocab_dict = {}
        vocab_dict["o"] = len(vocab_dict)
        vocab_dict["e"] = len(vocab_dict)
        vocab_dict["l"] = len(vocab_dict)
        vocab_dict["n"] = len(vocab_dict)
        vocab_dict["|"] = len(vocab_dict)
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        with open(vocab_file, 'w') as vocab_handle:
            json.dump(vocab_dict, vocab_handle)
        logger.debug('Created new NER vocab file: %s' % vocab_file)
    else:
        with open(vocab_file, "r") as vocab_handle:
            vocab_dict = json.load(vocab_handle)
        logger.debug('Using existing NER vocab file: %s' % vocab_file)

    return vocab_dict


def create_asr_vocab_dict(vocab_file, txt_paths_list):
    if not os.path.exists(vocab_file):
        # Need to read in whole training dataset to extract words
        data = {}
        for txt_path in txt_paths_list:
            assert os.path.exists(txt_path)
            with open(txt_path, "r") as fh:
                content = fh.read().replace('\n', ' ')
            # Add id and content to dict
            data[txt_path] = re.sub(chars_to_ignore_regex, '', content).lower() + " "
        dataDF = pd.DataFrame.from_dict(data, orient='index', columns=['text']).rename_axis('id').reset_index()

        dataDF['vocab'] = dataDF.apply(lambda x: list(set(x['text'])), axis=1)
        vocab_list = list(set([a for b in dataDF.vocab.tolist() for a in b]))

        # Add padding and unknown
        vocab_list.append('[PAD]')
        vocab_list.append('[UNK]')
        
        # Convert vocab to dict
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        # Replace space with something more readable
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        
        with open(vocab_file, 'w') as vocab_handle:
            json.dump(vocab_dict, vocab_handle)

        logger.debug('Created new ASR vocab file: %s' % vocab_file)
    else:
        with open(vocab_file, "r") as vocab_handle:
            vocab_dict = json.load(vocab_handle)
        logger.debug('Created new ASR vocab file: %s' % vocab_file)
        
    return vocab_dict


def compute_metrics_with_processor(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
    return compute_metrics


def dataset_generator(array_dir, txt_dir, file_ids, input_type, processor):

    assert input_type in ["asr", "ner"]

    for file_id in file_ids:
        entry = {}
        entry["id"] = file_id
        # Read in signal array
        array_path = os.path.join(array_dir, file_id + ".txt.gz")
        assert os.path.exists(array_path)
        signal = np.loadtxt(array_path)
        entry["array"] = signal
        # Set sampling_rate (dataset is known to have this rate)
        entry["sampling_rate"] = sampling_rate
        # Read in corresponding text
        text_path = os.path.join(txt_dir, file_id + ".txt")
        assert os.path.exists(text_path)
        with open(text_path, "r") as fh:
            text = fh.readline()
        entry["text"] = remove_punctuation(text)
        if input_type == 'ner':
            # Convert to chars
            entry["chars"] = convert_to_io_chars(text.lower())
        # Create input values
        entry['input_values'] = processor(signal, sampling_rate=16000).input_values[0]
        entry['input_length'] = len(entry['input_values'])

        with processor.as_target_processor():
            if input_type == 'ner':
                entry['labels'] = processor(entry['chars']).input_ids
            else:
                entry['labels'] = processor(entry['text']).input_ids

        yield entry


def train_hubert_model(input_type, array_dir, txt_dir, vocab_file, hubert_model, model_path, train_ids, val_ids, checkpoint_dir, num_epochs, batch_size):

    if input_type == 'ner':
        vocab_dict = create_ner_vocab_dict(vocab_file)
    else:
        vocab_dict = create_asr_vocab_dict(vocab_file, [os.path.join(txt_dir, f) for f in os.listdir(txt_dir)])

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", bos_token=None, eos_token=None, word_delimiter_token="|")
    tokenizer.save_pretrained(model_path)
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = HubertForCTC.from_pretrained(
        hubert_model,
        ctc_loss_reduction="mean", 
        vocab_size=len(vocab_dict),
        eos_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.freeze_feature_encoder()

    # report_to: Can be changed to e.g. "tensorboard"
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        num_train_epochs=num_epochs,
        fp16=fp16,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = compute_metrics_with_processor(processor)

    trainer = Trainer(model=model,
                      data_collator=data_collator,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=Dataset.from_generator(dataset_generator, gen_kwargs = {"array_dir": array_dir, "txt_dir": txt_dir, "file_ids": train_ids, "input_type": input_type, 'processor': processor}),
                      eval_dataset=Dataset.from_generator(dataset_generator, gen_kwargs={"array_dir": array_dir, "txt_dir": txt_dir, "file_ids": val_ids, "input_type": input_type, 'processor': processor}),
                      tokenizer=processor.feature_extractor,
                      )
    
    trainer.train()
    trainer.save_model(model_path)
    logger.debug('Model saved to: %s' % model_path)
    
    return model


def load_hubert_model(model_path, device):
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    model.to(device)

    return model, processor


def map_to_hubert_result(batch, model, processor, device):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(pred_ids, output_word_offsets=True, output_char_offsets=False)
    batch["pred_str"] = predictions['text'][0]
    batch["pred_word_offsets"] = predictions['word_offsets'][0]
    batch["pred_text"] = processor.decode(batch["labels"], group_tokens=False)
  
    return batch
    

def produce_hubert_results(array_dir, txt_dir, model, processor, device, test_ids, result_file):
    results = Dataset.from_generator(dataset_generator, gen_kwargs = {"array_dir": array_dir, "txt_dir": txt_dir, "file_ids": test_ids, 'input_type': 'ner', 'processor': processor}).map(map_to_hubert_result, fn_kwargs={"model": model, "processor": processor, "device": device})
    results.to_csv(result_file)
    logger.debug('Results saved to: %s' % result_file)
    

def main(array_dir, txt_dir, random_state, input_type, hubert_model, vocab_file, checkpoint_dir, num_epochs, batch_size, model_path, results_file):
    train_ids, val_ids, test_ids = create_train_validation_test_split(array_dir, txt_dir, random_state)
    logger.info('Created train / validation / test split.')
    train_hubert_model(input_type, array_dir, txt_dir, vocab_file, hubert_model, model_path, train_ids, val_ids, checkpoint_dir, num_epochs, batch_size)
    logger.info('Trained HuBERT model.')

    if len(test_ids) > 0:
        model, processor = load_hubert_model(model_path, device)
        produce_hubert_results(array_dir, txt_dir, model, processor, device, test_ids, result_file)
        logger.debug('Produced results to: %s' % result_file)


if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()
    assert os.path.exists(args.array_dir)
    assert os.path.exists(args.txt_dir)
    assert args.input_type in ['asr', 'ner']

    # The remaining data is going to be validation & test
    val_and_test_proportion = 100 - args.train_proportion
    
    # Create output directories
    if not os.path.exists(args.output_dir):
        os.mkdirs(args.output_dir)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints_" + str(args.train_proportion) + "_e" + str(args.num_epochs) + "_" + str(args.batch_size))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model_dir = os.path.join(args.output_dir, "model_" + args.input_type + "_base")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, "sample_" + '_'.join(args.hubert_model.split('-')[-2:len(args.hubert_model)]) + "_" + str(args.train_proportion) + "_e" + str(args.num_epochs) + "_b" + str(args.batch_size))

    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    # Output files
    result_file = os.path.join(args.output_dir, "results_" + args.input_type + "_" + str(args.train_proportion) + "_e" + str(args.num_epochs) + "_" + str(args.batch_size)) + ".csv"
    vocab_file = os.path.join(model_path, "vocab.json")
    
    logger.info('Building a(n) %s model' % args.input_type.upper())
    
    main(args.array_dir, args.txt_dir, args.random_state, args.input_type, args.hubert_model, vocab_file, checkpoint_dir, args.num_epochs, args.batch_size, model_path, result_file)
    
