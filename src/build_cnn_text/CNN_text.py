
import torch
import torch.nn as nn
import torch.functional as F


class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNN_Text,self).__init__()

        # Embedding layer 
        self.embedding = nn.Embedding( num_embeddings= vocab_size, 
                                      embedding_dim=embed_dim, 
                                      padding_idx=vocab['<PAD>'])  # padding_idx = index của <PAD>, embeddings cho PAD se khong duoc hoc
        
        # Convolution layer voi cac kernel size khac nhau (3, 4, 5)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=5)

        # Dropout de tranh overfiting
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer: 3 conv * 100 filters
        self.fc = nn.Linear(3 * 100, num_classes) 

    def forward(self,x):
        # x : [batch_size, seq_len]

        x = self.embedding(x) # [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1) # [batch_size, embed_dim, seq_len]

        # Apply convolution + ReLU
        x1 = F.relu(self.conv1(x)) # [batch_size, 100, seq_len-3+1]
        x2 = F.relu(self.conv2(x)) # [batch_size, 100, seq_len-4+1]
        x3 = F.relu(self.conv3(x)) # [batch_size, 100, seq_len-5+1]

        # Max pooling tren toan bo seq_len
        x1 = F.max_pool1d(x1, kernel_size= x1.shape[2]).squeeze(2) # [batch_size, 100]
        x2 = F.max_pool1d(x2, kernel_size= x2.shape[2]).squeeze(2) # [batch_size, 100]
        x3 = F.max_pool1d(x3, kernel_size= x3.shape[2]).squeeze(2) # [batch_size, 100]

        # Concatenate cac feature map
        x = torch.cat((x1, x2, x3), dim=1) # [batch_size, 300]
        x = self.dropout(x)

        # Full connected
        return self.fc(x)
        