import torch
from torch import nn
from transformers import BertModel
from torch.nn import MultiheadAttention, Linear, TransformerEncoder, TransformerEncoderLayer

class StockPredictor(nn.Module):
    def __init__(self, num_features, pretrained_model_name='bert-base-uncased', num_labels=3, dropout=0.1):
        super(StockPredictor, self).__init__()
        
        self.num_features = num_features 
        self.embed_dim = 256  # Consistent embedding dimension for all layers
        
        # BERT with partial fine-tuning
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        for name, param in self.bert.named_parameters():
            # Freeze all but the last 4 layers of BERT
            if 'layer' in name and int(name.split('.')[2]) < 8:  # 12 layers total, unfreeze last 4
                param.requires_grad = False

        # Projection layers
        self.financial_data_projection = Linear(self.num_features, self.embed_dim)
        self.bert_projection = Linear(768, self.embed_dim)

        # Positional encoding for financial data (assuming sequence length up to 50)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 50, self.embed_dim))  # Learnable
        
        # Cross-attention with multiple heads
        self.cross_attention = MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=4, batch_first=True
        )
        
        # Transformer encoder with normalization and dropout
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=4, batch_first=True, 
            dim_feedforward=self.embed_dim * 2, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = Linear(self.embed_dim, num_labels)

    def forward(self, input_ids, attention_mask, financial_data):
        # BERT output
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_outputs.last_hidden_state  # (batch_size, text_seq_len, 768)
        bert_sequence_output = self.bert_projection(bert_sequence_output)  # (batch_size, text_seq_len, 256)

        # Project and add positional encoding to financial data
        financial_data = self.financial_data_projection(financial_data)  # (batch_size, fin_seq_len, 256)
        fin_seq_len = financial_data.size(1)
        financial_data = financial_data + self.pos_encoding[:, :fin_seq_len, :]  # (batch_size, fin_seq_len, 256)

        # Cross-attention (financial_data as query, bert_sequence_output as key/value)
        attn_output, _ = self.cross_attention(
            financial_data, bert_sequence_output, bert_sequence_output
        )  # (batch_size, fin_seq_len, 256)
        
        # Residual connection and normalization
        attn_output = self.norm1(attn_output + financial_data)
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(attn_output)  # (batch_size, fin_seq_len, 256)
        
        # Residual connection and normalization
        transformer_output = self.norm2(transformer_output + attn_output)
        
        # Pooling over sequence (mean instead of just CLS)
        pooled_output = transformer_output.mean(dim=1)  # (batch_size, 256)
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
                
        return logits
