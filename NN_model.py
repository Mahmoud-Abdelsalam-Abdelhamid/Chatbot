import torch
import torch.nn as nn

# class NeuralNetwork(nn.Module):
    
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNetwork, self).__init__()
        
#         self.l1 = nn.Linear(input_size, hidden_size)
#         # self.l2 = nn.Linear(hidden_size, hidden_size)
#         # self.l3 = nn.Linear(hidden_size, hidden_size)
#         self.l4 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, X):
#         l1_out = self.l1(X)
#         l1_out = self.relu(l1_out)
#         l1_out = self.dropout(l1_out)
        
#         # l2_out = self.l2(l1_out)
#         # l2_out = self.relu(l2_out)
#         # l2_out = self.dropout(l2_out)

#         # l3_out = self.l3(l2_out)
#         # l3_out = self.relu(l3_out)
#         # l3_out = self.dropout(l3_out)

#         l4_out = self.l4(l1_out)
        
#         return l4_out  # Specify dim=1 for softmax over classes


class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
        
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.l1(X)
        X = self.norm1(X)  # Normalize activations
        X = self.relu(X)
        X = self.dropout(X)

        X = self.l2(X)
        X = self.norm2(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.l3(X)
        
        return X
    
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNetwork, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.norm1 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
#         self.l2 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.7)

#     def forward(self, X):
#         X = self.l1(X)
#         X = self.norm1(X)  # Normalize activations
#         X = self.relu(X)
#         X = self.dropout(X)
        
#         X = self.l2(X)
#         return X




