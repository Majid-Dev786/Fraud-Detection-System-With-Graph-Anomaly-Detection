# Fraud Detection System with Graph Anomaly Detection

## Project Overview

This project introduces a sophisticated Fraud Detection System that leverages Graph Anomaly Detection techniques to identify fraudulent transactions within a dataset. 

Utilizing a blend of PyTorch for neural network construction, PyTorch Geometric for graph neural network architecture, and Neo4j for graph database management, the system aims to detect anomalies by analyzing transaction patterns. 

By applying advanced machine learning algorithms on graph-structured data, it offers a novel approach to fraud detection that goes beyond traditional methods, providing higher precision and recall rates.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Detailed Project Description](#detailed-project-description)
3. [Getting Started](#getting-started)
4. [Usage in Real World Scenarios](#usage-in-real-world-scenarios)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)

## Detailed Project Description

The system is designed to process transaction data stored in a Neo4j graph database, using a Graph Convolutional Network (GCN) model for anomaly detection. 

Transactions are considered as nodes in the graph, with edges representing the relationships between them (e.g., sender, receiver). 

The model is trained to identify patterns associated with fraudulent activities by learning from the structural and transactional information present in the graph.

### Key Components:

- **Neo4jConnector:** Manages the connection to the Neo4j graph database and executes Cypher queries to fetch transaction data.
- **DataProcessor:** Prepares and processes the data for training, including generating graph edges and preparing node features.
- **ModelBuilder:** Constructs and trains a GCN model using PyTorch Geometric.
- **ModelEvaluator:** Evaluates the model's performance on a test set using precision, recall, and F1-score metrics.
- **AnomalyFinder:** Identifies anomalies based on the model's predictions.
- **BubbleGraph:** Visualizes the detected anomalies using a bubble graph representation.

## Getting Started

To use this system, start by cloning the repository from GitHub:
```
git clone https://github.com/Majid-Dev786/Fraud-Detection-System-With-Graph-Anomaly-Detection.git
```
Navigate into the project directory and ensure that you have the necessary prerequisites installed.

## Usage in Real World Scenarios

This system can be employed in various real-world scenarios where transaction data is available and fraud detection is crucial, such as in financial institutions, online marketplaces, and payment processors. 

By integrating this system, organizations can enhance their security measures against fraud, minimizing financial losses and improving customer trust.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.6 or later
- PyTorch and PyTorch Geometric
- Neo4j Python driver
- Plotly (for visualization)

## Installation

Follow these steps to set up the environment and run the script:
1. Install Python dependencies:
```
pip install torch torch-geometric neo4j plotly
```
2. Ensure Neo4j is installed and running on your system. Configure the `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` constants in the script to match your Neo4j instance credentials.
3. Run the script:
```
python Building A Basic Fraud Detection System With Graph Anomaly Detection From Scratch.py
```

The script will connect to the Neo4j database, process the transaction data, train the model, and output the evaluation results along with anomaly predictions visualized through a bubble graph.
