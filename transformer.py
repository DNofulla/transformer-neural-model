import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import math


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


class Transformer(nn.Module):
    def __init__(self, vocabulary_size, max_sequence_length, model_dimension, internal_dimension, number_of_classes, number_of_layers):
        """
        The Transformer model is initialized.
        
        :param vocabulary_size: The vocabulary size.
        :param max_sequence_length: The maximum sequence length.
        :param model_dimension: The model dimension.
        :param internal_dimension: The internal dimension.
        :param number_of_classes: The number of output classes.
        :param number_of_layers: The number of TransformerLayers.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, model_dimension) # Embedding layer
        self.positional_encoding = PositionalEncoding(model_dimension, max_sequence_length) # Positional encoding
        self.layers = nn.ModuleList([TransformerLayer(model_dimension, internal_dimension) for i in range(number_of_layers)]) # List of TransformerLayers
        self.output_layer = nn.Linear(model_dimension, number_of_classes) # Output layer
        self.log_softmax = nn.LogSoftmax(dim=-1) # Log Softmax function

    def forward(self, input_indices):
        """
        The forward function performs a forward pass through the Transformer model.
        
        :param input_indices: The Tensor of input indices.
        :return: The tuple of log probabilities and attention maps.
        """
        # We embed the input indices and add the positional encodings.
        embeddings = self.embedding(input_indices)
        embeddings = self.positional_encoding(embeddings)
        
        # We initialize the attention maps.
        attention_maps = []
        
        # We go through each Transformer layer and we get the output embeddings and attention map for the layer, and append the attention map to our attention map list.
        for layer in self.layers:
            embeddings, attention_map = layer(embeddings)
            attention_maps.append(attention_map)
        
        # We get the logits using the output layer, and we compute log probabilities using log softmax.
        logits = self.output_layer(embeddings)
        log_probabilities = self.log_softmax(logits)
        
        return log_probabilities, attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, model_dimension, internal_dimension):
        """
        The transformer layer is initialized.
        
        :param model_dimension: The model dimension.
        :param internal_dimension: The internal dimension.
        """
        super(TransformerLayer, self).__init__()
        
        # Query Layer, Key Layer, and Value Layer (Linear layers)
        self.query_layer = nn.Linear(model_dimension, internal_dimension)
        self.key_layer = nn.Linear(model_dimension, internal_dimension)
        self.value_layer = nn.Linear(model_dimension, internal_dimension)
        
        # Output projection layer
        self.output_projection = nn.Linear(internal_dimension, model_dimension) 
        
        # Feedforward network | Linear layer => ReLU activation => Linear layer
        self.feedforward_network = nn.Sequential(
            nn.Linear(model_dimension, internal_dimension),
            nn.ReLU(),
            nn.Linear(internal_dimension, model_dimension)
        ) 

    def forward(self, input_vectors):
        """
        TransformerLayer performs a forward pass through the Transformer layer.
        
        :param input_vectors: The input tensor.
        :return: The tuple that contains an output tensor and an attention map.
        """
        # We get the sequence length, the queries, the keys and the values
        sequence_length = input_vectors.size(0)
        queries = self.query_layer(input_vectors)
        keys = self.key_layer(input_vectors)     
        values = self.value_layer(input_vectors)
        
        # We compute the scaled dot-product attention scores, and initialize the key dimension
        key_dimension = queries.size(-1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(key_dimension) # (queries[i] • keys[j]) / sqrt(dk)
        
        # We apply a backward-only attention mask
        attention_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(input_vectors.device)
        attention_mask = attention_mask == 1
        scores = scores.masked_fill(attention_mask, float('-inf'))
        
        # We use the softmax function to get the attention weights, and compute the output
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        
        # We then project the attention output back to the model dimension and compute the residual output
        attention_output = self.output_projection(attention_output)
        residual_output = input_vectors + attention_output 
        
        # We then get our Feedforward Network output and the final output
        feedforward_output = self.feedforward_network(residual_output)
        output = residual_output + feedforward_output
        
        return output, attention_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, number_of_positions: int = 20, batched=False):
        """
        Initializes the positional encoding module.
        
        :param model_dimension: Dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs).
        :param number_of_positions: The number of positions that need to be encoded; the maximum sequence length this
        module will see.
        :param batched: True if you are using batching, False otherwise.
        """
        super().__init__()
        
        # We initialize the embedding layer and batching status
        self.embedding = nn.Embedding(number_of_positions, model_dimension)
        self.batched = batched

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.
        
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim].
        :return: A tensor of the same size with positional embeddings added in.
        """
        
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.arange(0, input_size)).type(torch.LongTensor).to(x.device)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            embedding_unsqueezed = self.embedding(indices_to_embed).unsqueeze(0)
            return x + embedding_unsqueezed
        else:
            return x + self.embedding(indices_to_embed)


def train_classifier(args, training_data, development_data):
    """
    Trains the Transformer classifier model.
    
    :param args: Command-line arguments or configuration.
    :param training_data: List of training examples.
    :param development_data: List of development (validation) examples.
    :return: Trained Transformer model.
    """
    
    # We calculate the vocabulary size from the training data
    vocabulary_size = max([example.input_tensor.max().item() for example in training_data]) + 1
    
    # We set the maximum sequence length, number of classes, model dimension, internal dimension, number of layers and learning rate
    max_sequence_length = 20
    number_of_classes = 3
    model_dimension = 64
    internal_dimension = 128
    number_of_layers = 2
    learning_rate = 1.15e-5

    # We initialize the Transformer model, optimizer, loss function and number of epochs
    model = Transformer(vocabulary_size, max_sequence_length, model_dimension, internal_dimension, number_of_classes, number_of_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()
    num_epochs = 10

    # The training loop
    for epoch in range(num_epochs):
        # We shuffle the training data
        model.train()
        random.shuffle(training_data)
        
        # We initialize the total loss
        total_loss = 0.0
        
        # We loop through each example in the training data
        for example in training_data:
            
            # We zero the gradients and perform a forward pass
            optimizer.zero_grad()
            log_probabilities, _ = model(example.input_tensor)
            
            # We compute the loss, perform a backward pass and take an optimization step
            loss = loss_function(log_probabilities, example.output_tensor)
            loss.backward()
            optimizer.step()
            
            # We add the loss to the total loss
            total_loss += loss.item()
            
        # We calculate the average loss and print it along with the epoch number
        average_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
        
        # We evaluate the model on the development set
        model.eval()
        # We initialize the number of correct predictions and the total number of predictions
        number_of_correct_predictions = 0
        total_number_of_predictions = 0
        
        # We loop through each example in the development data
        with torch.no_grad():
            for example in development_data:
                # We perform a forward pass and get the predictions
                log_probabilities, _ = model(example.input_tensor)
                predictions = torch.argmax(log_probabilities, dim=-1)
                
                # We calculate the number of correct predictions and the total number of predictions
                number_of_correct_predictions += (predictions == example.output_tensor).sum().item()
                total_number_of_predictions += example.output_tensor.size(0)
                
        # We calculate the development data accuracy and print it
        accuracy = number_of_correct_predictions / total_number_of_predictions
        print(f"Dev Accuracy: {accuracy:.4f}")

    # We return the trained model
    return model

def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param d
    o_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))