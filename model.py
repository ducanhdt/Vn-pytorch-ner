from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import autograd

import time
import _pickle as cPickle

import urllib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')

from hyper_parameter import *
from ultil import *
from model_ultil import *

import os
import sys
import codecs
import re
import numpy as np

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25,char_embedding_dim=25, use_gpu=False
                 , use_crf=True, char_mode='CNN'):
        '''
        Input parameters:
                
                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU, 
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''
        
        super(BiLSTM_CRF, self).__init__()
        
        #parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim
            
            #Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            
            #Performing LSTM encoding on the character embeddings
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)
                
            #Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

        #Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            #Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
    
        #Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])
        
        #Lstm Layer:
        #input dimension: word embedding dimension + character level representation
        #bidirectional=True, specifies that we are using the bidirectional LSTM
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(t+char_lstm_dim*2, hidden_dim, bidirectional=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
        
        #Initializing the lstm layer using predefined function for initialization
        init_lstm(self.lstm)
        
        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        
        #Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag) 

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            
            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _score_sentences(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _get_lstm_features(self, sentence, chars2, chars2_length, d):
    
        if self.char_mode == 'LSTM':
            
                chars_embeds = self.char_embeds(chars2).transpose(0, 1)
                
                packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
                
                lstm_out, _ = self.char_lstm(packed)
                
                outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
                
                outputs = outputs.transpose(0, 1)
                
                chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
                
                if self.use_gpu:
                    chars_embeds_temp = chars_embeds_temp.cuda()
                
                for i, index in enumerate(output_lengths):
                    chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
                
                chars_embeds = chars_embeds_temp.clone()
                
                for i in range(chars_embeds.size(0)):
                    chars_embeds[d[i]] = chars_embeds_temp[i]
        
        
        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)

            ## Creating Character level representation using Convolutional Neural Netowrk
            ## followed by a Maxpooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

            ## Loading word embeddings
        embeds = self.word_embeds(sentence)

        ## We concatenate the word embeddings and the character level representation
        ## to create unified representation for each word
        embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)

        ## Dropout on the unified embeddings
        embeds = self.dropout(embeds)

        ## Word lstm
        ## Takes words as input and generates a output at each step
        lstm_out, _ = self.lstm(embeds)

        ## Reshaping the outputs from the lstm layer
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)

        ## Dropout on the lstm output
        lstm_out = self.dropout(lstm_out)

        ## Linear layer converts the ouput vectors to tag space
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats
    def _forward_alg(self, feats):
        '''
        This function performs the forward algorithm explained above
        '''
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
            
        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat.view(-1, 1)
            
            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            tag_var = forward_var + self.transitions + emit_score
            
            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)
            
            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)
            
            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha
    def forward(self, sentence, chars, chars2_length, d):
        
        '''
        The function calls viterbi decode and generates the 
        most probable sequence of tags for the sentence
        '''
        
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(sentence, chars, chars2_length, d)
        # viterbi to get tag_seq
        
        # Find the best path, given the features.
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
    def viterbi_decode(self, feats):
        '''
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        '''
        backpointers = []
        # analogous to forward
        
        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy() # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy() 
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t] # holds the viterbi variables for this step
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
                
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
            
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features(sentence, chars2, chars2_length, d)

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentences(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores

    