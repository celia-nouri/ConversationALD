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
│   ├── models/                   # Model definitions
│   ├── mydatasets/               # Dataset handling
│   ├── notebooks/                # Jupyter notebooks for exploration
│   ├── utils/                    # Utility functions
│   ├── analyze_graphs.py         # Graph analysis scripts
│   ├── display_attention_weights.py
│   ├── dump_data.py              # Data preprocessing
│   ├── evaluate_many.py          # Batch evaluation
│   ├── experiments.py            # Experiment configurations
│   ├── main_evaluate.py          # Evaluation entry point
│   ├── run_eval.batch            # Batch script for evaluation
│   ├── run_eval_cpu.batch        # CPU-specific evaluation script
│   └── run_train.batch           # Training script
├── data/
│   ├── create_balanced_ds.py     # Dataset balancing script
│   ├── split-indices.py          # Data splitting script
│   └── .gitignore                # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## **Installation**

Clone the repository:
``` git clone https://github.com/yourusername/ConversationALD.git``` 
``` cd ConversationALD``` 
Create and activate a virtual environment:
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
Ensure that the CAD dataset and the extracted Reddit conversations are placed appropriately within the data/ directory. Use the provided scripts to preprocess and prepare the data:

``` python data/create_balanced_ds.py``` 
``` python data/split-indices.py``` 

### 2. Train the Model
Initiate model training using the provided batch script:

bash cmd/run_train.batch
This script will train the GNN model on the prepared dataset, utilizing the configurations specified in cmd/experiments.py.

### 3. Evaluate the Model
After training, evaluate the model's performance:

``` bash cmd/run_eval.batch``` 
For CPU-based evaluation, use:

``` bash cmd/run_eval_cpu.batch``` 

Evaluation metrics and results will be outputted to the console and saved as specified in the evaluation scripts.

## **Visualizations and Analysis**

To analyze and visualize attention weights and graph structures:

`python cmd/display_attention_weights.py`
`python cmd/analyze_graphs.py`
These scripts provide insights into the model's focus areas and the structural properties of the conversation graphs.

## **Citation**

If you utilize this codebase or the methodologies presented in our paper, please cite:

@article{nouri2025graphically,
  title={Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights},
  author={Nouri, Célia and Cointet, Jean-Philippe and Clavel, Chloé},
  journal={arXiv preprint arXiv:2504.01902},
  year={2025}
}

## **Contact**

For questions, collaborations, or access to the reconstructed Reddit conversations, please reach out to:

Célia Nouri: celia.nouri@inria.fr
Jean-Philippe Cointet: jeanphilippe.cointet@sciencespo.fr
Chloé Clavel: chloe.clavel@inria.fr
ar5iv
We hope this repository serves as a valuable resource for researchers and practitioners working on context-aware abusive language detection.
arXiv
