import random
import logging
from nltk.corpus import stopwords
from transformers import pipeline
from constants import mask_token
from constants import classifier_model
from constants import titles
from constants import ambiguous_names
from transformers import logging as transformer_logging

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformer_logging.set_verbosity_error()


def read_and_lower_names(filepath):
    name_list = []
    with open(filepath, "r") as fp:
        for line in fp:
            name_list.append(line.strip().lower())
    return name_list


def filter_names(word_list):
    return [w for w in word_list if not w in stopwords.words('english') and not w in ambiguous_names]


def prepare_names(first_names_file, last_names_file):

    first_list = read_and_lower_names(first_names_file)
    last_list = read_and_lower_names(last_names_file)

    first_list = filter_names(first_list)
    last_list = filter_names(last_list)

    return first_list, last_list


def potentially_correct_type(suggestion, original, first_names, last_names):
    if (original in first_names) and (suggestion in first_names):
        return True
    elif original in last_names and suggestion in last_names:
        return True
    elif suggestion in first_names or suggestion in last_names:
        return True
    return False


def titled_name(sentence, masking_model):
    for title in titles:
        titled_person = title + " " + mask_token[masking_model]
        if titled_person in sentence:
            return True
    return False


def select_replacement(similars, original, sentence, first_names, last_names, masking_model):

    chosen_replacement = None
    
    for similar in similars:
        if (similar['sequence'].lower() != sentence.lower()) and potentially_correct_type(similar['token_str'], original, first_names, last_names):
            chosen_replacement = similar['token_str']
            break

    # 2. Pick a random replacement out of the gazetteer list
    # 2.1 Heuristic approach to deal with first / last name

    if chosen_replacement == None:
        if original in first_names:
            while chosen_replacement == original or chosen_replacement == None:
                chosen_replacement = random.choice(first_names)
        elif original in last_names:
            while chosen_replacement == original or chosen_replacement == None:
                chosen_replacement = random.choice(last_names)
        elif titled_name(sentence, masking_model):
            # Pick randomly: first check for preceding title
            logger.debug('- chosing from titled name')
            while chosen_replacement == original or chosen_replacement == None:
                chosen_replacement = random.choice(last_names)
        else:
            chosen_replacement = random.choice(first_names)

    logger.debug('Chosen replacement %s for %s' % (chosen_replacement, original))
    return chosen_replacement


def generate_substitution_via_masking(masking_model, classifier, sentence, masked_sentence, originals, first_names, last_names):

    if not mask_token[masking_model] in masked_sentence:
        return sentence

    # 1. Try generating something with fill-mask
    # 1.1 Check it's not identical to the original
    # 1.2 Check if any appear in the gazetteer as a possible NE of the same type
    # 1.3 Check it's not been used too many times

    similars = classifier(masked_sentence)

    # Produces list of lists if multiple masks present but a single
    # depth list if only one mask
    
    chosen_replacements = [None] * len(originals)
    
    if isinstance(similars[0], list):
        for i in range(len(similars)):
            chosen_replacements[i] = select_replacement(similars[i], originals[i], sentence, first_names, last_names, masking_model)
    else:
        chosen_replacements[0] = select_replacement(similars, originals[0], sentence, first_names, last_names, masking_model)
    
    return chosen_replacements


def reconstruct_sentence(masked_sentence, substitutions, masking_model):
    words = masked_sentence.split()
    reconstructed_sentence = ""
    counter = 0
    for word in words:
        if reconstructed_sentence != "":
            reconstructed_sentence += " "
        if mask_token[masking_model] in word:
            reconstructed_sentence += word.replace(mask_token[masking_model], substitutions[counter])
            counter += 1
        else:
            reconstructed_sentence += word
    return reconstructed_sentence


def mask_and_reconstruct_sentence(masking_model, asr_sent, masked_sent, original_values, first_names_file, last_names_file, top_k):
    
    first_names, last_names = prepare_names(first_names_file, last_names_file)
    
    classifier = pipeline("fill-mask", model=classifier_model[masking_model], top_k=top_k)

    substituted_sent = generate_substitution_via_masking(masking_model, classifier, asr_sent, masked_sent, original_values, first_names, last_names)
    reconstructed_sent = reconstruct_sentence(masked_sent, substituted_sent, masking_model)

    logger.debug('Reconstructed sentences: %s' % reconstructed_sent)
    return substituted_sent, reconstructed_sent
    
