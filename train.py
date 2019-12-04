import torch 
from model import *
from model_ultil import *
from hyper_parameter import *
from ultil import *
from gensim.models import KeyedVectors as FastText
import numpy as np

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

import os
import sys
import codecs
import re

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    
    # We assume by default the tags lie outside a named entity
    default = tags['O']
    
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    
    chunks = []
    
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                # If chunk class is B, i.e., its a beginning of a new named entity
                # or, if the chunk type is different from the previous one, then we
                # start labelling it as a new entity
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def evaluating(model, datas, best_F,dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates 
     1) Flag to save the model 
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = [] # A list that stores predicted tags
    save = False # Flag that tells us if the model needs to be saved
    new_F = 0.0 # Variable to store the current F1-Score (may not be the best)
    correct_preds, total_correct, total_preds = 0., 0., 0. # Count variables
    
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        
        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        
        
        if parameters['char_mode'] == 'CNN':
            d = {} 

            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        
        # We are getting the predicted output from our model
        if use_gpu:
            val,out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val,out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out
    
        
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks      = set(get_chunks(ground_truth_id,tag_to_id))
        lab_pred_chunks = set(get_chunks(predicted_id,
                                         tag_to_id))

        # Updating the count variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)
    
    # Calculating the F1-Score
    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("{}: new_F: {} best_F: {} ".format(dataset,new_F,best_F))
    
    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model
    
    if new_F>best_F:
        best_F=new_F
        save=True

    return best_F, new_F, save


#load data

train_sentences = load_sentences(parameters['train'], parameters['zeros'])
test_sentences = load_sentences(parameters['test'], parameters['zeros'])
dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])
# train_sentences = train_sentences[0:100]

#update BIOES tag
update_tag_scheme(train_sentences, parameters['tag_scheme'])
update_tag_scheme(dev_sentences, parameters['tag_scheme'])
update_tag_scheme(test_sentences, parameters['tag_scheme'])

#create mapping for word tag and char
dico_words,word_to_id,id_to_word = word_mapping(train_sentences, parameters['lower'])
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

#preparing final dataset

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
print("{} / {} / {} sentences in train / dev / test.".format(len(train_data), len(dev_data), len(test_data)))

#load word embanding glove
# all_word_embeds = {}
# for i, line in enumerate(codecs.open(parameters['embedding_path'], 'r', 'utf-8')):
#     s = line.strip().split()
#     if len(s) == parameters['word_dim'] + 1:
#         all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

#load Fasttext embedding

# print("load Fasttextpretrain embadding")
# all_word_embeds = FastText.load_word2vec_format("./data/VNdata/cc.vi.300.vec")
# parameters['word_dim'] = len(all_word_embeds['hay'])
# print('fasttext embedding dim =' + str(parameters['word_dim']))

# word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters['word_dim']))

# for w in word_to_id:
#     try:
#         word_embeds[word_to_id[w]] = all_word_embeds[w]
#     except:
#         continue
# word_embeds[word_to_id['<UNK>']] = [0.001 for i in range(parameters['word_dim'])] 

#save pretrain
# with open(mapping_file, 'wb') as f:
#     mappings = {
#         'word_to_id': word_to_id,
#         'tag_to_id': tag_to_id,
#         'char_to_id': char_to_id,
#         'parameters': parameters,
#         'word_embeds': word_embeds
#     }
#     cPickle.dump(mappings, f)

# print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
print('load embedding matrix')

# def load_pre():
mappings = {}
with open(mapping_file, 'rb') as f:
    mappings = cPickle.load(f)
    word_to_id = mappings['word_to_id']
    # tag_to_id  = mappings['word_to_id']
    char_to_id = mappings['char_to_id'] 
    word_embeds =  mappings['word_embeds']

print('word_to_id: ', len(word_to_id))


#creating the model 
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                tag_to_ix=tag_to_id,
                embedding_dim=parameters['word_dim'],
                hidden_dim=parameters['word_lstm_dim'],
                use_gpu=use_gpu,
                char_to_ix=char_to_id,
                pre_word_embeds=word_embeds,
                use_crf=parameters['crf'],
                char_mode=parameters['char_mode'])

print("Model Initialized!!!")

#Reload a saved model, if parameter["reload"] is set to a path
if parameters['reload']:
    if not os.path.exists(parameters['reload']):
        print("downloading pre-trained model")
        model_url="https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/raw/master/trained-model-cpu"
        urllib.request.urlretrieve(model_url, parameters['reload'])
    model.load_state_dict(torch.load(parameters['reload']))
    print("model reloaded :", parameters['reload'])

if parameters['use_gpu']:
    print("use_gpu=TRUE")
    model.cuda()
else:
    print('use cpu only')
#turning training hyper parameter

learning_rate = 0.015
momentum = 0.9
number_of_epochs = parameters['epoch'] 
decay_rate = 0.05
gradient_clip = parameters['gradient_clip']
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

#variables which will used in training process
losses = [] #list to store all losses
loss = 0.0 #Loss Initializatoin
best_dev_F = -1.0 # Current best F-1 Score on Dev Set
best_test_F = -1.0 # Current best F-1 Score on Test Set
best_train_F = -1.0 # Current best F-1 Score on Train Set
all_F = [[0, 0, 0]] # List storing all the F-1 Scores
eval_every = len(train_data) # Calculate F-1 Score after this many iterations
plot_every = 2000 # Store loss after this many iterations
count = 0 #Counts the number of iterations


# parameters['reload']=False

if parameters['trainable']:
    tr = time.time()
    model.train(True)
    for epoch in range(1,number_of_epochs):
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            data = train_data[index]

            ##gradient updates for each data entry
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']
            
            if parameters['char_mode'] == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))
            
            if parameters['char_mode'] == 'CNN':

                d = {}

                ## Padding the each word to max word size of that sentence
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))


            targets = torch.LongTensor(tags)

            #we calculate the negative log-likelihood for the predicted tags using the predefined function
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
            loss += neg_log_likelihood.data / len(data['words'])
            neg_log_likelihood.backward()

            #we use gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
            optimizer.step()

            #Storing loss
            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            #Evaluating on Train, Test, Dev Sets
            if count % (eval_every) == 0 and count > (eval_every * 20) or \
                    count % (eval_every*4) == 0 and count < (eval_every * 20):
                model.train(False)
                best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F,"Train")
                best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F,"Dev")
                if save:
                    print("Saving Model to ", model_name)
                    torch.save(model.state_dict(), model_name)
                best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F,"Test")

                all_F.append([new_train_F, new_dev_F, new_test_F])
                model.train(True)

            #Performing decay on the learning rate
            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate/(1+decay_rate*count/len(train_data)))

    print(time.time() - tr)
    plt.plot(losses)
    plt.show()

if not parameters['trainable']:
    #reload the best model saved from training
    model.load_state_dict(torch.load(model_name))

#test 

model_testing_sentences = ['Hiệp định Paris về Việt Nam hoặc hiệp định Paris 1973 là hiệp định chấm dứt chiến tranh.']

#parameters
lower=parameters['lower']

#preprocessing
final_test_data = []
for sentence in model_testing_sentences:
    s=sentence.split()
    str_words = [w for w in s]
    words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else '<UNK>'] for w in str_words]
    
    # Skip characters that are not in the training set
    chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]
    
    final_test_data.append({
        'str_words': str_words,
        'words': words,
        'chars': chars,
    })

#prediction
predictions = []
print("Prediction:")
print("word : tag")
for data in final_test_data:
    words = data['str_words']
    chars2 = data['chars']

    d = {} 
    
    # Padding the each word to max word size of that sentence
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(data['words']))

    # We are getting the predicted output from our model
    if use_gpu:
        val,predicted_id = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
    else:
        val,predicted_id = model(dwords, chars2_mask, chars2_length, d)

    pred_chunks = get_chunks(predicted_id,tag_to_id)
    temp_list_tags=['NA']*len(words)
    for p in pred_chunks:
        temp_list_tags[p[1]]=p[0]
        
    for word,tag in zip(words,temp_list_tags):
        print(word,':',tag)
    print('\n')
