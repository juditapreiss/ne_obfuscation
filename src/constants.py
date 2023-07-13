### Constants

# Cannot be changed due to the data used to train HuBERT models
sampling_rate = 16000

# Masking tokens differ per model
mask_token = {}
mask_token['roberta'] = "<mask>"
mask_token['albert'] = "[MASK]"
mask_token['bert'] = "[MASK]"

# Actual HuggingFace model to use
classifier_model = {}
classifier_model['roberta'] = 'distilroberta-base'
classifier_model['bert'] = 'bert-large-uncased'
classifier_model['albert'] = 'albert-xxlarge-v2'

# Currently recognized titles
titles = ["Mr", "Ms", "Mrs", "Dr", "Prof", "Revd"]

# Currently recognized ambiguous names
ambiguous_names = ["back", "home", "lady", "main", "nos", "oh", "kay", "question"]
