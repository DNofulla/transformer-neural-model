import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Functional

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_index, model):
        """
        The Neural Language Model is initialized.
        :param vocab_index: The Vocabulary Indexer.
        :param model: The Trained PyTorch model.
        """
        
        # We initialize the model and set it to evaluation mode
        self.model = model
        self.model.eval()
        
        # We set the vocabulary index and the device
        self.vocab_index = vocab_index
        self.device = next(model.parameters()).device

    def get_next_char_log_probs(self, context):
        """
        param context: The context string for the Language Model.
        Returns The log probability distribution.
        """

        # We set the model to evaluation mode
        with torch.no_grad():
            # We pad the context with spaces and get the last 16 characters
            maximum_context_length = 16
            padded_context = ' ' * maximum_context_length + context
            padded_context = padded_context[-maximum_context_length:]

            # We convert the context to indices and create a context tensor
            context_indices = [self.vocab_index.index_of(c) for c in padded_context]
            context_tensor = torch.tensor(context_indices, dtype=torch.long, device=self.device).unsqueeze(0)

            # We get the logits for the context tensor and get the logits for the last time step
            logits = self.model(context_tensor)
            last_logits = logits[0, -1, :]

            # We get the log probabilities for the last logits and return the numpy array
            log_probabilities = Functional.log_softmax(last_logits, dim=-1)
            return log_probabilities.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        """
        param next_chars: The sequence of characters for the context.
        param context: The context string for the Language Model.
        
        Returns The log probability of the sequence.
        """
        
        # We set the model to evaluation mode and initialize the total log probability
        total_log_probabilities = 0.0
        current_context = context
        for current_character in next_chars:
            # We get the log probabilities for the current context and character 
            log_probabilities = self.get_next_char_log_probs(current_context)
            current_character_index = self.vocab_index.index_of(current_character)
            
            # We add the log probability of the current character to the total log probability and update the context
            total_log_probabilities += log_probabilities[current_character_index]
            current_context = (current_context + current_character)[-19:]
            
        return total_log_probabilities


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # We initialize the positional encoding and register the buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # We calculate the division term and set the positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # We set the sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # We register the buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        :param x: The input tensor
        :return: The tensor with the added positional encodings
        """
        
        # We add the positional encodings to the input tensor
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
    
        self.embedding = nn.Embedding(vocab_size, d_model) # Embedding layer
        self.pos_encoder = PositionalEncoding(d_model) # Positional encoding layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) # Encoder layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers) # Transformer encoder
        self.fc_out = nn.Linear(d_model, vocab_size) # Output layer

    def forward(self, src, src_mask=None):
        """
        :param src: The input tensor
        :param src_mask: The mask tensor
        :return: The output tensor
        """
        
        # We add the positional encodings to the input tensor and pass it through the transformer encoder
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim) 
        src = self.pos_encoder(src)
        
        # We pass the input tensor through the transformer encoder and the output layer
        output = self.transformer_encoder(src, mask=src_mask) 
        output = self.fc_out(output)
        
        return output


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    # We initialize vocabulary size, model dimension, number of heads, number of layers, feedforward dimension, dropout, chunk size, batch size, number of epochs, learning rate, and device
    vocabulary_size = len(vocab_index)
    model_dimension = 128
    number_of_heads = 4
    number_of_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    chunk_size = 20
    batch_size = 64
    number_of_epochs = 5
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We initialize the Transformer Language Model, optimizer, criterion, and move the model to the selected device (CPU or GPU)
    model = TransformerLanguageModel(
        vocabulary_size, model_dimension, number_of_heads, number_of_layers, dim_feedforward, dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # We create the input and output contexts, train indices, pad token, and we initialize the input and output contexts   
    input_contexts = []
    output_contexts = []
    train_indices = [vocab_index.index_of(c) for c in train_text]
    pad_token = vocab_index.index_of(' ')

    # We loop through the train indices and create the input and output contexts
    for i in range(0, len(train_indices) - chunk_size):
        
        # We get the input and output sequences
        input_sequence = [pad_token] + train_indices[i:i + chunk_size - 1]
        output_sequence = train_indices[i:i + chunk_size]
        
        # We append the input and output sequences to the input and output contexts
        input_contexts.append(input_sequence)
        output_contexts.append(output_sequence)

    # We convert the input and output contexts to tensors
    input_contexts = torch.tensor(input_contexts, dtype=torch.long)
    output_contexts = torch.tensor(output_contexts, dtype=torch.long)

    # We create the dataset and train loader
    dataset = torch.utils.data.TensorDataset(input_contexts, output_contexts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # The training loop
    model.train()
    for epoch in range(number_of_epochs):
        
        # We initialize the total loss
        total_loss = 0.0
        
        # We loop through the training loader
        for input_batch, output_batch in train_loader:
            
            # We move the input and output batch to the selected device and create the source mask
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            src_mask = torch.triu(torch.full((input_batch.size(1), input_batch.size(1)), float('-inf')), diagonal=1).to(device)

            # We zero the gradients, get the output, compute the loss, backpropagate, and update the optimizer step
            optimizer.zero_grad()
            output = model(input_batch, src_mask) 
            loss = criterion(output.view(-1, vocabulary_size), output_batch.view(-1))
            loss.backward()
            optimizer.step()

            # We add the loss to the total loss
            total_loss += loss.item()
        
        # We calculate the average loss and perplexity for this epoch and print it
        avg_loss = total_loss / len(train_loader)
        perplexity = math.exp(avg_loss)
        print(f'Epoch {epoch + 1}/{number_of_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}')

    # We create the trained Neural Language Model and return it
    return NeuralLanguageModel(vocab_index, model)