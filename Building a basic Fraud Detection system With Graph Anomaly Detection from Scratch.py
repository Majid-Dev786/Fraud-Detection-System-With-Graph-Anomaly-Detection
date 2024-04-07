# Importing necessary libraries and modules
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from neo4j import GraphDatabase
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

# Constants for Neo4j database connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "GDB-FAD"
NEO4J_PASSWORD = "arian786"
# Threshold to consider a transaction as high amount
HIGH_AMOUNT_THRESHOLD = 1000000

# Configure logging
logging.basicConfig(level=logging.INFO)

class Neo4jConnector:
    # Initialize connector with database credentials
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    # Connect to the Neo4j database
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except Exception as e:
            logging.error("Failed to create the driver:", e)
            self.driver = None

    # Close the database connection
    def close(self):
        if self.driver:
            self.driver.close()

    # Execute a Cypher query
    def execute_cypher_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return list(result)

    # Load transactions with fraud flag or exceeding the high amount threshold
    def load_transactions(self, high_amount_threshold):
        query = f'''
        MATCH (t:Transaction)
        WHERE t.fraud = true OR t.amount > {high_amount_threshold}
        RETURN t.id AS id, t.fraud AS fraud, t.amount AS amount
        '''
        return self.execute_cypher_query(query)

class DataProcessor:
    # Initialize processor with transaction data
    def __init__(self, fraud_transactions, high_amount_transactions, records):
        self.fraud_transactions = fraud_transactions
        self.high_amount_transactions = high_amount_transactions
        self.records = records

    # Process and prepare data for model training
    def process_data(self):
        # Extract IDs and create mappings
        fraud_transactions_ids = [transaction_id for transaction_id, _ in self.fraud_transactions]
        high_amount_transactions_ids = [transaction_id for transaction_id, _ in self.high_amount_transactions]
        all_transactions_ids = list(set(fraud_transactions_ids + high_amount_transactions_ids))
        transaction_id_to_index = {transaction_id: index for index, transaction_id in enumerate(all_transactions_ids)}
        # Generate indices for graph edges
        fraud_transactions_indices = [transaction_id_to_index[tid] for tid in fraud_transactions_ids]
        high_amount_transactions_indices = [transaction_id_to_index[tid] for tid in high_amount_transactions_ids]
        edge_index = torch.tensor([
            fraud_transactions_indices + high_amount_transactions_indices,
            [0] * len(fraud_transactions_indices) + [1] * len(high_amount_transactions_indices)],
            dtype=torch.long)
        # Prepare node features and labels
        x = torch.tensor([[amount] for _, _, amount in self.records], dtype=torch.float)
        num_nodes = len(all_transactions_ids)
        y = torch.zeros(num_nodes, dtype=torch.long)
        for idx in fraud_transactions_indices:
            y[idx] = 1
        # Oversample fraud transactions to balance dataset
        fraud_indices = [i for i, label in enumerate(y) if label == 1]
        oversample_indices = fraud_indices * (num_nodes // len(fraud_indices) - 1)
        x = torch.cat([x, x[oversample_indices]], dim=0)
        y = torch.cat([y, y[oversample_indices]], dim=0)
        num_nodes = len(y)
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        # Split data into training and test sets
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:int(0.8 * num_nodes)] = 1
        data.train_mask = train_mask
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[int(0.8 * num_nodes):] = 1
        data.test_mask = test_mask
        return data

class ModelBuilder:
    class Net(torch.nn.Module):
        # Define neural network architecture
        def __init__(self, num_node_features):
            super(ModelBuilder.Net, self).__init__()
            self.conv1 = GCNConv(num_node_features, 32)
            self.conv2 = GCNConv(32, 16)
            self.conv3 = GCNConv(16, 2)

        # Forward pass through the network
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

    # Initialize model builder with processed data
    def __init__(self, data):
        self.data = data

    # Build and train the model
    def build_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ModelBuilder.Net(self.data.num_node_features).to(device)
        self.data = self.data.to(device)
        # Prepare for training
        num_classes = 2
        class_counts = [sum(self.data.y == i) for i in range(num_classes)]
        class_weights = [1.0 / (c + 1e-10) for c in class_counts]
        weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.NLLLoss(weight=weights)
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(self.data)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
        return model

class ModelEvaluator:
    # Initialize evaluator with trained model and data
    def __init__(self, model, data):
        self.model = model
        self.data = data

    # Evaluate the model on the test set
    def evaluate(self):
        self.model.eval()
        _, pred = self.model(self.data).max(dim=1)
        test_mask = self.data.test_mask

        true_labels = self.data.y[test_mask].cpu().numpy()
        pred_labels = pred[test_mask].cpu().numpy()

        # Calculate performance metrics
        if sum(pred_labels) == 0:
            print('No positive samples predicted, setting precision and recall to zero.')
            precision, recall, f1 = 0, 0, 0
        else:
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)

        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1-score: {:.4f}'.format(f1))

class AnomalyFinder:
    # Initialize finder with trained model and data
    def __init__(self, model, data):
        self.model = model
        self.data = data

    # Find anomalies in the test set
    def find_anomalies(self):
        self.model.eval()
        _, pred = self.model(self.data).max(dim=1)
        anomalies = pred[self.data.test_mask].nonzero().tolist()

        return anomalies

class BubbleGraph:
    # Initialize bubble graph with transaction data
    def __init__(self, fraud_transactions, high_amount_transactions):
        self.fraud_transactions = fraud_transactions
        self.high_amount_transactions = high_amount_transactions

    # Create and show a bubble graph visualization
    def create_graph(self, anomalies):
        fraud_transactions_x = [transaction_id for transaction_id, _ in self.fraud_transactions]
        fraud_transactions_y = [amount for _, amount in self.fraud_transactions]
        high_amount_transactions_x = [transaction_id for transaction_id, _ in self.high_amount_transactions]
        high_amount_transactions_y = [amount for _, amount in self.high_amount_transactions]

        fig = go.Figure()

        # Add scatter plots for different types of transactions
        fig.add_trace(go.Scatter(
            x=fraud_transactions_x,
            y=fraud_transactions_y,
            mode='markers',
            marker=dict(color='red'),
            name='Proven Fraud'
        ))

        fig.add_trace(go.Scatter(
            x=high_amount_transactions_x,
            y=high_amount_transactions_y,
            mode='markers',
            marker=dict(
                color=high_amount_transactions_y,
                colorscale='Rainbow',
                colorbar=dict(
                    title='Transaction Amount',
                    len=0.8,
                )
            ),
            name='Maybe Fraud Due To High Amount'
        ))

        fig.add_trace(go.Scatter(
            x=anomalies,
            y=[0]*len(anomalies),
            mode='markers',
            marker=dict(color='black'),
            name='Predicted Anomalies'
        ))

        fig.show()

def main():
    # Main execution flow
    neo4j_connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_connector.connect()

    if neo4j_connector.driver is None:
        logging.error("Failed to connect to Neo4j.")
        return

    records = neo4j_connector.load_transactions(HIGH_AMOUNT_THRESHOLD)
    fraud_transactions = [(record["id"], record["amount"]) for record in records if record["fraud"]]
    high_amount_transactions = [(record["id"], record["amount"]) for record in records if record["amount"] > HIGH_AMOUNT_THRESHOLD]

    neo4j_connector.close()

    if len(fraud_transactions) == 0 and len(high_amount_transactions) == 0:
        logging.info("No Anomalies Found In The DataSet.")
        return

    data_processor = DataProcessor(fraud_transactions, high_amount_transactions, records)
    data = data_processor.process_data()

    model_builder = ModelBuilder(data)
    model = model_builder.build_model()

    model_evaluator = ModelEvaluator(model, data)
    model_evaluator.evaluate()

    anomaly_finder = AnomalyFinder(model, data)
    anomalies = anomaly_finder.find_anomalies()

    bubble_graph = BubbleGraph(fraud_transactions, high_amount_transactions)
    bubble_graph.create_graph(anomalies)

if __name__ == "__main__":
    main()