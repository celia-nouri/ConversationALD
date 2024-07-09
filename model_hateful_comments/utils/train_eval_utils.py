import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer
from models.model import all_model_names
import wandb
from tqdm import tqdm 
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
wandb.init(project="hatespeech-class")

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


def train(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(val_loader))
    print("Test set size: ", len(test_loader))
    num_epochs, model_name, validation, size = args.epochs, args.model, args.validation, args.size
    print("Train: epochs=", num_epochs, ", dataset_name=hateful_discussions", ", model=", model_name)
    model.to(device)
    best_val_loss = float('inf')
    # Generate a unique timestamp string
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_check_path = f"./models/checkpoints/{timestamp}_{model_name}_{size}.pt"
    # Training loop
    for epoch in range(num_epochs):
        running_loss = float(0)
        running_corrects = 0
        true_labels = []
        predicted_labels = []
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for index, data in enumerate(progress_bar):
            optimizer.zero_grad()
            data = data.to(device)
            y, y_pred = run_model_pred(model, data, model_name, 'hateful_discussions', device)
            if model_name == 'fb-roberta-hate' or model_name =='bert-class':
                loss = y_pred.loss
                loss.backward()
                optimizer.step()
                logits = y_pred.logits
                # Compute predictions
                _, pred_label = torch.max(logits, dim=1)
                # Update running metrics
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                    loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
                )
            elif model_name == 'multimodal-transformer' or model_name =='img-text-transformer':
                # Compute the loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(y_pred, y).to(device)
                loss.backward()
                optimizer.step()
                _, pred_label = torch.max(y_pred, dim=1)
                # Update running metrics on training set 
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels)
            
            elif y_pred.shape == y.shape:
                y_pred = y_pred.to(device)
                loss = criterion(y_pred, y).to(device)
                loss.backward()
                optimizer.step()
                # Update running metrics on training set 
                running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(loss, y_pred, y, running_loss, running_corrects, true_labels, predicted_labels)

            else:
                print('y pred is ', y_pred, ' and it has a shape of ', y_pred.shape)
                print('y is ', y, ' and it has a shape of ', y.shape)
                print("ERROR: output and expected predictions have different shapes. y_pred: ", y_pred.shape, " , target y: ", y.shape)
                return
     
        avg_loss = running_loss/ len(train_loader)
        epoch_accuracy = float(running_corrects) / len(train_loader)
        epoch_precision = precision_score(true_labels, predicted_labels)
        epoch_recall = recall_score(true_labels, predicted_labels)
        epoch_f1 = f1_score(true_labels, predicted_labels)


        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": epoch_accuracy,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1
        })
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.4f}, Train Precision: {epoch_precision:.4f}, "
            f"Train Recall: {epoch_recall:.4f}, Train F1 Score: {epoch_f1:.4f}")

        # Validation
        if validation:
            avg_val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(model, val_loader, criterion, model_name, 'hateful_discussions', device)
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_accuracy
            })
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")
            
            # Update best validation loss and save checkpoint if needed
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_check_path)
    if not validation:
        torch.save(model.state_dict(), model_check_path)

    # Finally, evaluate on the test set and report all metrics
    print("Running evaluation ...")
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, model_name, 'hateful_discussions', device)
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    })
    print(f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, "
        f"Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}")

    # Finish the run
    wandb.finish()