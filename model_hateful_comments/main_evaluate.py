import torch 
import evaluate
import numpy as np
from tqdm import tqdm 
import wandb
import argparse
from models.model import *
from mydatasets.dataloader import get_data_loaders
from transformers import AdamW
import torch.nn.functional as F
from utils.train_eval_utils import evaluate_model
from HatefulDiscussionsModeling.model_hateful_comments.models.model import get_device

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='hateful_discussions', type=str, help='dataset name, can take one of these values: ["Palestine_convs_roberta", "hateful_discussions"]')

    parser.add_argument('--seed', type=int, default=100, help='random seed') 
    
    parser.add_argument('--model', type=str, default='roberta-class', help='can take one of the following values: ["text-class", "distil-class", "simple-graph", "roberta-class]') 

    args = parser.parse_args()

    device = get_device()

    dataset_name = args.dataset_name
    model_name = args.model
    hidden_channels = 64
    num_heads = 1
    print("Dataset name: ", dataset_name, ". Model name: ", model_name)

    # Log hyperparameters
    wandb.config = {
        "model": model_name,
        "dataset": dataset_name,
    }    

    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name, batch_size=1, num_workers=0, args=args)

    # Instantiate your model
    model = SimpleTextClassifier()
    if model_name == "simple-graph":
        model = SimpleGraphModel(in_channels=768, hidden_channels=hidden_channels, num_heads=num_heads)
    elif model_name == "distil-class":
        model = DistilBERTClass()
    elif model_name == "text-class":
        model = SimpleTextClassifier()
    elif model_name == "roberta-class":
        model = RoBERTaHateClassifier()
    elif model_name == 'fb-roberta-hate':
        tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    
    # Define optimizer and loss function
    criterion = F.binary_cross_entropy_with_logits
    if model_name == "roberta-class":
        criterion = F.binary_cross_entropy_with_logits
    #criterion = loss_fn

    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(valid_loader))
    print("Test set size: ", len(test_loader))


    evaluate(args, model, model_name, dataset_name, test_loader, criterion, device=device)

def evaluate(model, model_name, dataset_name, test_loader, criterion, device):
    # Log hyperparameters
    wandb.config = {
        "mode": "evaluation",
        "model": model_name,
        "dataset": dataset_name,
    }    
    # Finally, evaluate on the test set and report all metrics
    print("Running evaluation ...")
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, model_name, dataset_name, device)
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


if __name__ == "__main__":
    main_eval()
    #main_data_dump("data/hatefuldiscussion_data/traindata_dump.jsonl")
