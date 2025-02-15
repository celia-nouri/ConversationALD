import json
import torch
import wandb
from models.model import all_model_names, all_base_pretrained_models, get_device
from transformers import AutoTokenizer
import torch.nn.functional as F
from mydatasets.mydataloaders import get_graph_dataloaders
from models.model import get_model

import argparse


def run_eval(checkpoint_path, args):
    model_name = args.model
    size = args.size
    validation = args.validation
    n_epochs = args.epochs
    learning_rate = args.lr
    undirected = args.undirected
    temp_edges = args.temp_edges
    num_layers = args.num_layers


    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "small-1000", "medium", "large", "cad", "cad-small"], "Invalid size setting: {}".format(size)

    device = get_device()

    print(f"Evaluating {model_name} on {size} ALD Conversation dataset with validation={validation}...")
    print(f"Model hyperparams are num layers {num_layers}, undirected {undirected}, temporal edges {temp_edges}")
    print("Checkpoint path ", checkpoint_path)

    # Log hyperparameters
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "hidden_channels": 64,
        "num_heads": 1,
        "model": model_name,
        "dataset": 'ald_conversation',
        "undirected": undirected,
        "temp_edges": temp_edges,
        "size": size,
        "validation": validation,
        "eval_mode":True,
    }
    _, _, test_loader = get_graph_dataloaders(size, validation, 0)
    print("Test set size: ", len(test_loader))

    model = get_model(args, model_name, hidden_channels=64, num_heads=1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    criterion = F.binary_cross_entropy_with_logits

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Finally, evaluate on the test set and report all metrics
    print("Running evaluation ...")
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, model_name, 'ald_conversation', device, f"{model_name}_{size}_eval_test_outputs.tsv", tokenizer)

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
    parser = argparse.ArgumentParser()

    models_string = json.dumps(all_model_names)
    pretrained_model_string = json.dumps(all_base_pretrained_models)

    parser.add_argument('--model', type=str, default='gat-test', help='the model to use, can take one of the following values: ' + models_string)
    # Options are "bert-base-uncased", "bert-base-cased", "roberta-base", "xlm-roberta-base", "allenai/longformer-base-4096", "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large" 
    parser.add_argument('--pretrained-model-name', type=str, default="bert-base-uncased", help='name for pretrained text model to use to generate text embeddings, can take one of the following values: ' + pretrained_model_string)

    parser.add_argument('--num-layers', type=int, default=3, help='the number of GAT layers in graph models')
    parser.add_argument('--undirected', type=bool, default=False, help='define the graph model as an undirected graph')
    parser.add_argument('--temp-edges', type=bool, default=False, help='add temporal edges to the graph')
    parser.add_argument('--trim', type=str, default="recent", help='graph construction trimming streatgy, should be either affordance, recent, or left empty for no trimming.')
    parser.add_argument('--new-trim', type=bool, default=False, help='rather or not to use the new trimming strategy (edge from post to target node only, instead of edges from post to all other nodes)')

    parser.add_argument('--size', type=str, default='cad', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, metavar='E', help='learning rate')

    args = parser.parse_args()

    run_eval("path_to_repo/models/checkpoints/gat-recent-3l-42_3906579.pt" , args)
                                   