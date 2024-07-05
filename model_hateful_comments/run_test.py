import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
from utils.train_eval_utils import update_running_metrics, accuracy_score, precision_score, recall_score, f1_score


def run_tests(model_path, texts, ys):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Create the pipeline with truncation and padding
    pipeline = TextClassificationPipeline(
        model=model, 
        tokenizer=tokenizer,
        padding=True, 
        truncation=True, 
        max_length=512
    )
    running_loss = float(0)
    running_corrects = 0
    true_labels = []
    predicted_labels = []

    for i, t in enumerate(texts):
        y = ys[i]
        inputs = tokenizer(t, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        if not isinstance(y, torch.LongTensor):
            y = y.long()
        inputs["labels"] = y
        y_pred = model(**inputs)
        loss = y_pred.loss
        logits = y_pred.logits
        _, predicted_label = torch.max(logits, dim=1)
        y = torch.tensor([y])
        running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(loss, predicted_label, y, running_loss, running_corrects, true_labels, predicted_labels)
        
    
    avg_loss = running_loss/len(texts)
    accuracy = float(running_corrects)/ len(texts)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return avg_loss, accuracy, precision, recall, f1





if __name__ == "__main__":
    model_path = "facebook/roberta-hate-speech-dynabench-r4-target"
    texts = ['This is a test text.', 'I hate gays and black people.', 'I love eating sushis.', 'Muslims are a danger for democracy.', 'Black people are nice.']
    ys = torch.tensor([torch.tensor([torch.tensor(0.)]), torch.tensor([torch.tensor(1.)]), torch.tensor([torch.tensor(0.)]), torch.tensor([torch.tensor(1.)]), torch.tensor([torch.tensor(1.)])])
    avg_loss, accuracy, precision, recall, f1 = run_tests(model_path, texts, ys)
    print(f"Avg loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")