# Import generic wrappers.
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Global constant.
MODEL_NAME = "illuin/camembert-base-fquad"

# Load model.
def get_model_and_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

# Inference.
def inference(context, question, model, tokenizer):
    # Tokenize input.
    inputs = tokenizer(question, context, return_tensors="pt")
    # Apply model.
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the highest probability from the model output for the start and end positions.
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index   = outputs.end_logits.argmax()
    # Decode the predicted tokens to get the answer.
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    # Return answer.
    return tokenizer.decode(predict_answer_tokens)