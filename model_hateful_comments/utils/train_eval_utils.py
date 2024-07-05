import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer
from models.model import all_model_names

tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

def precision_score(true_labels, predicted_labels):
    true_positive = Counter()
    false_positive = Counter()

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if predicted_label == 1:
            if true_label == predicted_label:
                true_positive[1] += 1
            else:
                false_positive[1] += 1

    precision = true_positive[1] / (true_positive[1] + false_positive[1] + 1e-9)
    return precision

def recall_score(true_labels, predicted_labels):
    true_positive = Counter()
    false_negative = Counter()

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1:
            if true_label == predicted_label:
                true_positive[1] += 1
            else:
                false_negative[1] += 1

    recall = true_positive[1] / (true_positive[1] + false_negative[1] + 1e-9)
    return recall

def f1_score(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return f1

# Define validation function
def evaluate_model(model, loader, criterion, model_name, dataset_name, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in loader:
            y, y_pred = run_model_pred(model, data, model_name, dataset_name, device)
            if model_name == 'fb-roberta-hate' or model_name =='bert-class':
                loss = y_pred.loss
                logits = y_pred.logits

                # Compute predictions
                _, pred_label = torch.max(logits, dim=1)
                # Update running metrics
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                    loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
                )
            elif model_name == 'multimodal-transformer' or model_name == 'img-text-transformer':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(y_pred, y).to(device)
                _, pred_label = torch.max(y_pred, dim=1)
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                    loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
                )
            else:
                loss = criterion(y_pred, y).to(device)
                # Accumulate loss, corrects, true and predicted labels 
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(loss, y_pred, y, running_loss, running_corrects, true_labels, predicted_labels)
        
    # Calculate average loss
    avg_loss = running_loss / len(loader)
    accuracy = float(running_corrects) / len(loader)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return avg_loss, accuracy, f1, precision, recall

def run_model_pred(model, data, model_name, dataset_name, device):
    assert model_name in all_model_names
    # Initialize y and y_pred with zeros
    #batch_size = data.size(0)
    y = torch.zeros((1, 1))
    y_pred = torch.zeros((1, 1))
    data = data.to(device)

    if model_name == "text-class":
        if dataset_name == "Palestine_convs_roberta":
            x, y, _, _ = data
            x = x.to(device)
            y_pred = model(x).to(device)
            y_pred = y_pred.squeeze(-1)
        elif dataset_name == "hateful_discussions":
            x = data.x
            x = x.to(device)


    elif model_name == "simple-graph":
        if dataset_name == "Palestine_convs_roberta":
            x, y, edge_index, _ = data
            x = x.to(device)
            edge_index = edge_index.to(device)
            y_pred = model(x, edge_index).to(device)
            y = y.transpose(0, 1).squeeze(-1)
            y = y.to(device)
        elif dataset_name == "hateful_discussions":
            x = data
            x = x.to(device)
    elif model_name == "bert-class":
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        x = data.x
        y = data.y
        labels = y.long().to(device)
        input_ids = x["input_ids"][masked_index].to(device)
        attention_mask = x["attention_mask"][masked_index].to(device)
        y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    elif model_name == "roberta-class":
        if dataset_name == "hateful_discussions":
            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
            x = data.x
            input_ids = x["input_ids"][masked_index].to(device)
            attention_mask = x["attention_mask"][masked_index].to(device)
            y_pred = model(input_ids, attention_mask)
            y_pred = y_pred.flatten()
            y = data.y


    elif model_name == "fb-roberta-hate":
        if dataset_name == "hateful_discussions":
            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
            texts = data.x_text
            my_text = texts[masked_index]
            my_dic, _, _, _ = my_text
            full_text = my_dic['body'] if 'body' in my_dic.keys() else ''
            y = data.y
            inputs = tokenizer(full_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            if not isinstance(y, torch.LongTensor):
                inputs["labels"] = y.long()
            inputs = {key: value.to(device) for key, value in inputs.items()}
            y_pred = model(**inputs)

    elif model_name == "multimodal-transformer" or model_name == "img-text-transformer":
        if dataset_name == "hateful_discussions":
            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
            texts = data.x_text
            my_text = texts[masked_index]
            my_dic, _, _, _ = my_text
            y_pred, _ = model(data)
            y_pred = y_pred.to(device)
            y = data.y

    y = y.to(device)
    #y_pred = y_pred.to(device)
    return y, y_pred

def update_running_metrics(loss, predicted_label, y, running_loss, running_corrects, true_labels, predicted_labels):
    running_loss += loss.item()
    #threshold = 0.5  # Example threshold
    # Ensure y_pred is a floating-point tensor

    # Compute the predicted label using sigmoid and threshold    
    #predicted_label = torch.tensor(1.0) if torch.sigmoid(y_pred).item() > threshold else torch.tensor(0.0)
   
    #only for fb hate roberta
    good_pred = False
    if float(predicted_label.item()) == float(y.item()):
        running_corrects += 1
        good_pred = True
    #running_corrects += torch.sum(predicted_label.item() == y.item())
    true_labels.extend(y.cpu().numpy())
    predicted_labels.extend(predicted_label)
    return running_loss, running_corrects, true_labels, predicted_labels, good_pred
