# **ConversationALD: Graph Neural Networks for Abusive Language Detection in Social Media**

This repository accompanies the paper:  
**Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights**  
*Célia Nouri, Jean-Philippe Cointet, Chloé Clavel*  
[arXiv:2504.01902](https://arxiv.org/abs/2504.01902)

We introduce a graph-based approach to Abusive Language Detection (ALD) that models Reddit conversations as graphs, capturing both content and structural context.  
Our method leverages Graph Neural Networks (GNNs), and especially Graph Attention Networks (GATs) to outperform traditional context-agnostic and linear context-aware models.

## **Repository Structure**

```
ConversationALD/
├── cmd/
│   ├── models/                   # Model definitions, and directory to store checkpoints
│   ├── mydatasets/               # Dataset handling
│   ├── notebooks/                # Jupyter notebooks for data exploration, and visualisation 
│   ├── utils/                    # Utility functions (for data processing, training and evaluation)
│   ├── analyze_graphs.py         # Graph analysis scripts
│   ├── display_attention_weights.py
│   ├── dump_data.py              # Data preprocessing
│   ├── evaluate_many.py          # Entry point to evaluate various models (called by run_eval)
│   ├── experiments.py            # Entry point to train an experiment model (called by run_train)
│   ├── main_evaluate.py          # Entry point to evaluate a model (called by run_eval)
│   ├── run_eval.batch            # Batch script for evaluation (SLURM)
│   ├── run_eval_cpu.batch        # CPU-specific evaluation script (SLURM)
│   └── run_train.batch           # Training script (SLURM)
├── data/
│   ├── create_balanced_ds.py     # Dataset balancing script
│   ├── split-indices.py          # Helper script for data splitting
│   └── .gitignore                # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## **Installation**

Clone the repository:
``` git clone https://github.com/celianouri/ConversationALD.git``` 
``` cd ConversationALD``` 
Create and activate a virtual environment (using python or conda):
``` python3 -m venv venv``` 
``` source venv/bin/activate``` 
Install dependencies:
``` pip install -r requirements.txt``` 

## **Dataset Preparation**

### 1. Download the CAD Dataset
We utilize the Contextual Abuse Dataset (CAD) introduced by Vidgen et al. (NAACL 2021). The dataset includes annotated Reddit conversations with abuse labels contextualized within conversation threads.

Paper: Introducing CAD: the Contextual Abuse Dataset
[github link](https://github.com/dongpng/cad_naacl2021/)
[aclanthology:2021.naacl-main.182](https://aclanthology.org/2021.naacl-main.182/)
Download the dataset from the GitHub repository or the associated Zenodo link provided therein.

### 2. Extract Full Reddit Conversations
To reconstruct full Reddit conversation threads, we recommend using the [Arctic Shift project](https://arctic-shift.photon-reddit.com)
This tool provides access to archived Reddit data, allowing for the extraction of complete conversation threads necessary for our graph-based modeling.

Note: Due to Reddit's data policies, full conversation data may not be publicly distributable. Researchers interested in accessing the reconstructed conversations used in our study may contact us directly for potential collaboration.

## **Running the Codee**

### 1. Preprocess the Data
Ensure that the CAD dataset and the extracted Reddit conversations are formatted as graph.pt files, placed appropriately within the data/ directory, as `graph-xx.pt` files. 

### 2. Train the Model
Modify the arguments in the `experiments.py` file, specifying the model name, number of layers, dataset size, trimming strategy, and graph construction method (directed, with or without temporal edges).
Then, initiate model training locally directly from the python file 
``` python cmd/experiments.py```

or using the provided batch script (SLURM):
```bash cmd/run_train.batch```

### 3. Evaluate the Model
After training, evaluate the model's performance by modifying the `cmd/main_evaluate.py` arguments to match the ones used for training. Also, update the path to the model checkpoint in that same file.

Then, initiate model evaluation locally using the python file 
``` python cmd/main_evaluate.py```

or using the provided batch script (SLURM):
``` bash cmd/run_eval.batch``` 
For CPU-based evaluation, use:
``` bash cmd/run_eval_cpu.batch``` 

Evaluation metrics and results will be outputted to the console, or saved as specified in the evaluation scripts (out and err files).

## **Citation**

If you utilize this codebase or the methodologies presented in our paper, please cite:

```
@article{nouri2025graphically,
  title={Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights},
  author={Nouri, Célia and Cointet, Jean-Philippe and Clavel, Chloé},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025},
}
```

## **Contact**

For questions, collaborations, or access to the reconstructed Reddit conversations, please reach out to me by e-mail, using the e-mail provided in the paper.

We hope this repository serves as a valuable resource for researchers and practitioners working on context-aware abusive language detection.
