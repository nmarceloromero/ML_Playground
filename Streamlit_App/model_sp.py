# Original code : https://huggingface.co/mrm8488/spanish-t5-small-sqac-for-qa
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Global constant.
MODEL_NAME = 'mrm8488/spanish-t5-small-sqac-for-qa'

# Load model.
def get_model_and_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    return model, tokenizer

def inference(question, context, model, tokenizer):
    input_text = 'question: %s  context: %s' % (question, context)
    features = tokenizer([input_text ], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
    return tokenizer.decode(output[0], skip_special_tokens=True)
