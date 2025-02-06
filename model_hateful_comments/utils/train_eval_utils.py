import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer, LongformerTokenizer
from models.model import all_model_names
import wandb
from tqdm import tqdm 
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from utils.construct_graph import get_graph, get_hetero_graph
import json



tokenizerRobertaHS = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
wandb.init(project="hatespeech-class")

def precision_score(true_labels, predicted_labels):
    true_positive = 0
    false_positive = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if predicted_label == 1:
            if true_label == predicted_label:
                true_positive += 1
            else:
                false_positive += 1
    if true_positive + false_positive == 0:
        return 0.0  # Avoid division by zero
    precision = true_positive / (true_positive + false_positive)
    return precision

def recall_score(true_labels, predicted_labels):
    true_positive = 0
    false_negative = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1:
            if true_label == predicted_label:
                true_positive += 1
            else:
                false_negative += 1


    if true_positive + false_negative == 0:
        return 0.0  # Avoid division by zero

    recall = true_positive / (true_positive + false_negative)
    return recall

def f1_score(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def get_criterion(device, balanced=True):
    if not balanced:
        class_counts = torch.tensor([22355, 5200])  # CAD class distribution: about 80%, 20%
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        print('class weights ', class_weights, ' class 0 should have lower weight since it has more samples')
        
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()

def get_context_texts(conv_array, index):
    my_id = conv_array[index][0]['id']
    parent_id = conv_array[index][0]['parent_id']

    parent_id, parent_text, post_id, post_text = "", "", "", ""
    for i, comment in enumerate(conv_array):
        if parent_id == comment[0]['id']:
            parent_text = comment[0]['body']
        if not parent_id or parent_id == "" or parent_id == "NA":
            post_id = comment[0]['id']
            post_text = comment[0]['body']
    if post_id == parent_id:
        return [parent_text]
    return [post_text, parent_text]

def get_context_texts_all(conv_array, index):
    my_id = conv_array[index][0]['id']
    all_texts = []
    for comment in conv_array:
        if my_id != comment[0]['id']:
            all_texts.append(comment[0]['body'])
    return all_texts

def get_all_texts(conv_array):
    all_texts = []
    for comment in conv_array:
        all_texts.append(comment[0]['body'])
    return all_texts


def get_post_parent(conv_array, index):
    my_id = conv_array[index][0]['id']
    parent_id = conv_array[index][0]['parent_id']

    parent_id, parent_text, post_id, post_text = "", "", "", ""
    for i, comment in enumerate(conv_array):
        if parent_id == comment[0]['id']:
            parent_text = comment[0]['body']
        if not parent_id or parent_id == "" or parent_id == "NA":
            post_id = comment[0]['id']
            post_text = comment[0]['body']

    return [post_text, parent_text]

# Define validation function
def evaluate_model(model, loader, criterion, model_name, dataset_name, device, output_file="", tokenizer=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    true_labels = []
    predicted_labels = []
    if model_name == "longform-class":
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    elif model_name == "xlmr-class":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    with torch.no_grad():

        with open(output_file, 'w') if output_file else None as outfile:
            for data in loader:
                with autocast():
                    y, y_pred = run_model_pred(model, data, model_name, dataset_name, device, tokenizer)
                    if model_name == 'fb-roberta-hate' or model_name =='bert-class' or model_name == "bert-concat" or model_name == "longform-class" or model_name == "xlmr-class" or model_name == "roberta-class" or model_name == "modernbert-class":
                        loss = y_pred.loss
                        logits = y_pred.logits

                        # Compute predictions
                        _, pred_label = torch.max(logits, dim=1)
                        # Update running metrics
                        running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                            loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
                        )
                    elif model_name == 'multimodal-transformer' or model_name == 'img-text-transformer' or model_name == "text-graph-transformer" or model_name == 'gat-model' or model_name == 'gat-test' or model_name == 'hetero-graph' or model_name == "bertwithneighconcat" or model_name == "bert-ctxemb" or model_name == "ctxembed":
                        criterion = get_criterion(device).to(device)
                        y = y.long().to(device)
                        if model_name == "ctxembed":
                            y = y.squeeze(1)
                        logits = y_pred.to(device)
                        loss = criterion(y_pred, y).to(device)
                        _, pred_label = torch.max(y_pred, dim=1)
                        running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                            loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
                        )
                    
                    else:
                        loss = criterion(y_pred, y).to(device)
                        # Accumulate loss, corrects, true and predicted labels 
                        running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(loss, y_pred, y, running_loss, running_corrects, true_labels, predicted_labels)
                    if outfile:
                        my_output = {'pred_label' : pred_label.item(), 'y': y.item(), 'y_pred': logits.detach().cpu().numpy().tolist()}
                        if model_name == "bert-class" or model_name == "bert-concat" or model_name == "bert-ctxemb" or model_name == "longform-class" or model_name == "xlmr-class" or model_name == "roberta-class" or model_name == "modernbert-class":
                            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
                            text = data.x_text[masked_index][0]['body'] 
                            my_output['text'] = text
                            my_output['x'] = text = data.x_text[masked_index][0]
                            my_output['masked_index'] = masked_index.detach().cpu().numpy().tolist() 
                        elif model_name == 'gat-test':
                            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
                            text = data.x_text[masked_index][0]["body"]
                            _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, data["y_mask"], with_temporal_edges=False, undirected=False)
                            texts = [data.x_text[i][0]["body"] for i in conv_indices_to_keep]
                            my_output['text'] = text
                            my_output['texts'] = texts
                            my_output['conv_indices_to_keep'] = conv_indices_to_keep
                        elif model_name == 'hetero-graph':
                            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
                            text = data.x_text[masked_index]
                            mask = data["y_mask"]
                            num_comment_nodes, _, num_users, _, conv_indices_to_keep, my_new_mask_idx = get_hetero_graph(data.x_text, mask, with_temporal_edges=False)
                            texts = [data.x_text[i][0]["body"] for i in conv_indices_to_keep]
                            my_output['text'] = text
                            my_output['texts'] = texts
                            my_output['conv_indices_to_keep'] = conv_indices_to_keep
                            my_output['num_comment_nodes'] = num_comment_nodes
                            my_output['num_users'] = num_users
                        json.dump(my_output, outfile)
                        outfile.write('\n')
                                        
    # Calculate average loss
    avg_loss = running_loss / len(loader)
    accuracy = float(running_corrects) / len(loader)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return avg_loss, accuracy, f1, precision, recall

def run_model_pred(model, data, model_name, dataset_name, device, tokenizer=None):
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
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
        y_pred = model(token_type_ids=encoding.get("token_type_ids", None), input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], labels=labels)
    elif model_name == "xlmr-class":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
        y_pred = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], labels=labels)
    elif model_name == "longform-class":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
        global_attention_mask = torch.zeros_like(encoding["attention_mask"])
        global_attention_mask[:, 0] = 1  # Apply global attention to the first token
        y_pred = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], global_attention_mask=global_attention_mask, labels=labels)
    elif model_name == "roberta-class":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
        y_pred = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], labels=labels)
    elif model_name == "modernbert-class":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
        y_pred = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], labels=labels)

    elif model_name == "bert-concat" :
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        context_texts = get_context_texts_all(data.x_text, masked_index)
        labels = y.long().to(device)
        y_pred = model(context_texts, text, labels=labels) 

    elif model_name == "bert-ctxemb":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        y_pred = model(data, text, labels=labels) 

    elif model_name == "bertwithneighconcat":
        y = data.y
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        context_texts = get_post_parent(data.x_text, masked_index)
        labels = y.long().to(device)
        y_pred = model(context_texts, text, labels=labels) 
        
    elif model_name == "ctxembed":
        masked_index = data.y_mask.nonzero(as_tuple=True)[0]
        text = data.x_text[masked_index][0]['body']
        labels = y.long().to(device)
        y_pred = model(data, text, labels=labels)         

    elif model_name == "fb-roberta-hate":
        if dataset_name == "hateful_discussions":
            masked_index = data.y_mask.nonzero(as_tuple=True)[0]
            texts = data.x_text
            my_text = texts[masked_index]
            my_dic, _, _, _ = my_text
            full_text = my_dic['body'] if 'body' in my_dic.keys() else ''
            y = data.y
            inputs = tokenizerRobertaHS(full_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            if not isinstance(y, torch.LongTensor):
                inputs["labels"] = y.long()
            inputs = {key: value.to(device) for key, value in inputs.items()}
            y_pred = model(**inputs)

    elif model_name == "multimodal-transformer" or model_name == "img-text-transformer" or model_name == "text-graph-transformer":
        if dataset_name == "hateful_discussions":
            y_pred, _ = model(data)
            y_pred = y_pred.to(device)
            y = data.y.to(device)
    elif model_name == 'gat-model' or model_name == 'gat-test' or model_name == 'hetero-graph':
        try:
            # Example operation that might run out of memory
            y_pred = model(data).to(device)
            y = data.y.to(device)

        except RuntimeError as e:
            # Check if it's an out-of-memory error
            if "CUDA out of memory" in str(e) and device.type == 'cuda':
                print("CUDA out of memory error caught. Attempting to clear cache...")
                torch.cuda.empty_cache()
                
                # Optionally, you can retry the operation or handle it differently
                # Warning: Retrying can lead to another out-of-memory error if the
                # memory requirements are too high.
                try:
                    y_pred = model(data)
        
                except RuntimeError as retry_e:
                    print("Retry failed after clearing cache. Exiting or alternative handling needed.")
                    raise retry_e  # Re-raise the exception after retry failure
                
            else:
                # Re-raise the error if it's not a CUDA out-of-memory error
                raise e
        y_pred = y_pred.to(device)
        y = data.y.long()

    y = y.to(device)
    #y_pred = y_pred.to(device)
    return y, y_pred

def update_running_metrics(loss, predicted_label, y, running_loss, running_corrects, true_labels, predicted_labels):

    running_loss += loss.item()

    good_pred = False
    if float(predicted_label.item()) == float(y.item()):
        running_corrects += 1
        good_pred = True
    #running_corrects += torch.sum(predicted_label.item() == y.item())
    true_labels.extend(y.cpu().numpy())
    predicted_labels.append(predicted_label.item())

    return running_loss, running_corrects, true_labels, predicted_labels, good_pred


def train(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(val_loader))
    print("Test set size: ", len(test_loader))
    num_epochs, model_name, validation, size = args.epochs, args.model, args.validation, args.size
    print("Train: epochs=", num_epochs, ", dataset_name=hateful_discussions", ", model=", model_name)
    #model.to(device)
    best_val_f1 = float('-inf')
    best_model = model 
    # Patience is the maximum number of epoch with decaying validation scores we will wait for, before early stopping the training
    patience = 20
    trigger_times = 0
    # Generate a unique timestamp string
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_check_path = f"./models/checkpoints/{timestamp}_{model_name}_{size}.pt"
    print(f"Saving model to ", model_check_path)
    scaler = GradScaler()
    accumulation_steps = 16
    if model_name == "longform-class":
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    elif model_name == "xlmr-class":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Training loop
    for epoch in range(num_epochs):
        running_loss = float(0)
        running_corrects = 0
        true_labels = []
        predicted_labels = []
        model.train()
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            #torch.cuda.memory._record_memory_history()
        for index, data in enumerate(progress_bar):
            criterion = get_criterion(device).to(device)
            with autocast():
                y, y_pred = run_model_pred(model, data, model_name, 'hateful_discussions', device, tokenizer)
                y = y.to(device)
                if model_name == 'fb-roberta-hate' or model_name =='bert-class' or model_name == "bert-concat" or model_name == "longform-class" or model_name == "xlmr-class" or model_name == "roberta-class" or model_name == "modernbert-class":
                    labels = y.long().to(device)
                    logits = y_pred.logits.to(device)
                    loss = criterion(logits, labels).to(device)
                elif model_name == 'multimodal-transformer' or model_name =='img-text-transformer' or model_name =='text-graph-transformer' or model_name == 'gat-model' or model_name == 'gat-test' or model_name == 'hetero-graph' or model_name == "bertwithneighconcat" or model_name == "bert-ctxemb" or model_name == "ctxembed":
                    logits = y_pred.to(device)
                    labels = y.long().to(device)
                    if model_name == "ctxembed":
                        labels = labels.squeeze(1)
                    loss = criterion(logits, labels).to(device)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            # accumulate gradients for 
            if (index + 1) % accumulation_steps == 0 or index == len(train_loader) - 1:                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # Compute predictions
            _, pred_label = torch.max(logits, dim=1)
            # Update running metrics
            running_loss, running_corrects, true_labels, predicted_labels, _ = update_running_metrics(
                loss, pred_label, y, running_loss, running_corrects, true_labels, predicted_labels
            )
                
        # record memory utilization snapshot for debugging
        #if device.type == 'cuda':
        #    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

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

        # If validation, compute and report the main metrics on the validation set
        if validation:
            avg_val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(model, val_loader, criterion, model_name, 'hateful_discussions', device, f"{model_name}_{size}_{epoch}_val_outputs.tsv", tokenizer)
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            })
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")
            
            # Update best validation f1 score, best model and save checkpoint
            if val_f1 > best_val_f1:
                print("Replacing best validation F1 score from ", best_val_f1 , " to ", val_f1)
                best_val_f1 = val_f1
                best_model = model
                trigger_times = 0
                torch.save(model.state_dict(), model_check_path)
            # Early stopping logic 
            else:
                trigger_times += 1
                if trigger_times > patience:
                    print('Early stopping!')
                    break
    if not validation:
        torch.save(model.state_dict(), model_check_path)

    # Finally, evaluate on the test set and report all metrics
    print("Running evaluation ...")
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(best_model, test_loader, criterion, model_name, 'hateful_discussions', device, f"{model_name}_{size}_test_outputs.tsv", tokenizer)
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