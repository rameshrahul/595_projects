#!/usr/bin/env python
# coding: utf-8

# # Homework 2 Parts 1-3: Word2Vec
# 
# This homework will have you implementing word2vec using PyTorch and let you familiarize yourself with building more complex neural networks and the larger PyTorch development infrastructure.
# 
# Broadly, this homework consists of a few major parts:
# 1. Implement a `Corpus` class that will load the dataset and convert it to a sequence of token ids
# 2. Implement negative sampling to select tokens to be used as negative examples of words in the context
# 3. Create your dataset of positive and negative examples per context and load it into PyTorch's `DataLoader` to use for sampling
# 4. Implement a `Word2Vec` class that is a PyTorch neural network
# 5. Implement a training loop that samples a _batch_ of target words and their respective positive/negative context words
# 6. Implement rare word removal and frequent word subsampling
# 7. Run your model on the full dataset for at least one epoch
# 8. Do the exploratory parts of the homework
# 9. Save vectors and word-indexing data for later use in training a classifier
# 
# After Step 5, you should be able to run your word2vec implementation on a small dataset and verify that it's learning correctly. Once you can verify everything is working, proceed with steps 6 and beyond. **Please note that this list is a general sketch and the homework PDF has the full list/description of to-dos and all your deliverables.**
# 
# ### Estimated performance times on medium dataset
# 
# We designed this homework to be run on a laptop-grade CPU, so no GPU is required. If your primary computing device is a tablet or similar device, this homework can also be _developed_ on that device but then run on a more powerful machine in the Great Lakes cluster (for free). Such cases are the exception though. Following, we report on the estimated times from our reference implementation on the medium dataset for longer-running or data-intensive pieces of the homework. Your timing may vary based on implementation design; major differences in time (e.g., 10x longer) usually point to a performance bug.
# 
# * Reading and tokenizing: ~5 seconds
# * Subsampling and converting to token ids: ~15 seconds
# * Generating the list of training examples: ~2 minutes (~15 minutes before the random number generator fix)
# * Training one epoch: ~12 minutes

# In[2]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm, trange
from collections import Counter
import random
from torch import optim
import gzip
import math
import wandb

# Helpful for computing cosine similarity--Note that this is NOT a similarity!
from scipy.spatial.distance import cosine

# Handy command-line argument parsing
import argparse

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

# We'll use this to save our models
from gensim.models import KeyedVectors

import pickle

#
# IMPORTANT NOTE: Always set your random seeds when dealing with stochastic
# algorithms as it lets your bugs be reproducible and (more importantly) it lets
# your results be reproducible by others.
#
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


# In[3]:


import os
os.environ['WANDB_API_KEY'] = 'b52c89333beb85a2b1137e25b011353bac754299'
get_ipython().system('wandb login')


# # Create an efficient random number generator (do this part this later)
# 
# Computers have to work to generate random numbers. However, the effort for getting those random numbers varies by how many you ask for. In practice, it's _much_ more efficient to generate many random numbers at once, rather than one at a time. 
# 
# 
# In generating the training data for word2vec, you'll be generating a lot of random numbers. We've added a helpful class that you can use to eventually speed things up. You should use an instance of this `RandomNumberGenerator` class to generate the numbers you need (rather than using `np.random`). This class should work as a quick drop-in in its current implementation. **You should change the implementation of this class _only_ after getting the rest of your code debugged**. Once you get things working, update the code in this class so that it will create large buffers of random numbers and then when asked for a new number, read the next number from the buffer instead of calling `random` or `randint`. Essentially, this class will pre-allocate many random numbers ahead of time and then return them in order to avoid the overhead of generating them one at a time. You could see up to an ~80% performance improvement in your negatve sampling code generation as a result of this!

# In[5]:


class RandomNumberGenerator:
    ''' 
    A wrapper class for a random number generator that will (eventually) hold buffers of pre-generated random numbers for
    faster access. For now, it just calls np.random.randint and np.random.random to generate these numbers 
    at the time they are needed.
    '''

    def __init__(self, buffer_size, seed=12345):
        '''
        Initializes the random number generator with a seed and a buffer size of random numbers to use

        Args:
            buffer_size: The number of random numbers to pre-generate. You will eventually want 
                         this to be a large-enough number than you're not frequently regenerating the buffer
            seed: The seed for the random number generator
        '''
        self.max_val = -1
        self.buffer_size = buffer_size
        self.seed = seed
        # TODO (later): create a random number generator using numpy and set its seed    
        # TODO (later): pre-generate a buffer of random floats to use for random()
        self.fill_float_buffer()
            

    def fill_float_buffer(self):
        self.float_buffer = np.random.random(size=self.buffer_size)
        self.float_index = 0

    def random(self):
        '''
        Returns a random float value between 0 and 1
        '''
        # TODO (later): get a random number from the float buffer, rather than calling np.random.random
        # NOTE: If you reach the end of the buffer, you should refill it with new random float numbers
        random_float = self.float_buffer[self.float_index]
        self.float_index += 1
        if self.float_index == self.buffer_size:
            self.fill_float_buffer()
        return random_float

    def set_max_val(self, max_val):
        '''
        Sets the maximum integer value for randint and creates a buffer of random integers
        '''
        self.max_val = max_val
        # NOTE: This default implemenation just sets the max_val and does not create a buffer of random integers
        # TODO (later): Implement a buffer of random integers (for now, we'll just use np.random.randint)
        self.int_buffer = np.random.randint(0, self.max_val, self.buffer_size)
        self.int_index = 0

    def randint(self):
        '''
        Returns a random int value between 0 and self.max_val (inclusive)
        '''        
        if self.max_val == -1:
            raise ValueError("Need to call set_max_val before calling randint")

        random_int = self.int_buffer[self.int_index]
        self.int_index += 1
        if self.int_index == self.buffer_size:
            self.set_max_val(self.max_val)
        
        # TODO (later): get a random number from the int buffer, rather than calling np.random.randint
        # NOTE: If you reach the end of the buffer, you should refill it with new random ints
        return random_int


# ## Create a class to hold the data
# 
# Before we get to training word2vec, we'll need to process the corpus into some representation. The `Corpus` class will handle much of the functionality for corpus reading and keeping track of which word types belong to which ids. The `Corpus` class will also handle the crucial functionality of generating negative samples for training (i.e., randomly-sampled words that were not in the target word's context).
# 
# Some parts of this class can be completed after you've gotten word2vec up and running, so see the notes below and the details in the homework PDF.

# In[56]:


class Corpus:
    
    def __init__(self, rng: RandomNumberGenerator):

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.rng = rng

        # These state variables become populated with function calls
        #
        # 1. load_data()
        # 2. generate_negative_sampling_table()
        #
        # See those functions for how the various values get filled in

        self.word_to_index = {} # word to unique-id
        self.index_to_word = {} # unique-id to word

        # How many times each word occurs in our data after filtering
        self.word_counts = Counter()

        # A utility data structure that lets us quickly sample "negative"
        # instances in a context. This table contains unique-ids
        self.negative_sampling_table = []
        
        # The dataset we'll use for training, as a sequence of unqiue word
        # ids. This is the sequence across all documents after tokens have been
        # randomly subsampled by the word2vec preprocessing step
        self.full_token_sequence_as_ids = None
        
    def tokenize(self, text):
        '''
        Tokenize the document and returns a list of the tokens
        '''
        return self.tokenizer.tokenize(text)        

    def load_data(self, file_name, min_token_freq):
        '''
        Reads the data from the specified file as long long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        '''

        # Step 1: Read in the file and create a long sequence of tokens for
        # all tokens in the file
        all_tokens = []
        print('Reading data and tokenizing')

        with open(file_name, 'r') as file:
            file_content = file.read()
            # for line in file:
            #     for word in line.split():
            #         all_tokens.append(word.lower())
        file_content = file_content.lower()
        all_tokens = self.tokenize(file_content)
    
        # Step 2: Count how many tokens we have of each type
        print('Counting token frequencies')

        token_freq = Counter(all_tokens)


        # Step 3: Replace all tokens below the specified frequency with an <UNK>
        # token. 
        #
        # NOTE: You can do this step later if needed
        print("Performing minimum thresholding")

        # token_freq['unk'] = 0
        # for token, freq in token_freq.items():
        #     if freq < min_token_freq:
        #         token_freq['unk'] += freq
        #         token_freq[token] = 0

        # # clear out 0 tokens
        # token_freq = +token_freq
        # if 'unk' not in token_freq:
        #     token_freq['unk'] = 0


        # Step 4: update self.word_counts to be the number of times each word
        # occurs (including <UNK>)

        self.word_counts = token_freq
        
        # Step 5: Create the mappings from word to unique integer ID and the
        # reverse mapping.

        self.word_to_index = {} # word to unique-id
        self.index_to_word = {} # unique-id to word

        index = 0
        for token, freq in token_freq.items():
            self.word_to_index[token] = index
            self.index_to_word[index] = token
            index += 1
        
        # Step 6: Compute the probability of keeping any particular *token* of a
        # word in the training sequence, which we'll use to subsample. This subsampling
        # avoids having the training data be filled with many overly common words
        # as positive examples in the context

        probabilities = {}
        for word in self.word_to_index:
            idx = self.word_to_index[word]
            p_wi = self.word_counts[word]/len(all_tokens)
            if p_wi == 0:
                probabilities[idx] = 1
            else:
                probabilities[idx] = (math.sqrt(p_wi/0.001) + 1) * .001/p_wi
            # if probabilities[idx] <= 0.5:
            #     print(word)
                        
        # Step 7: process the list of tokens (after min-freq filtering) to fill
        # a new list self.full_token_sequence_as_ids where 
        #
        # (1) we probabilistically choose whether to keep each *token* based on the
        # subsampling probabilities (note that this does not mean we drop
        # an entire word!) and 
        #
        # (2) all tokens are convered to their unique ids for faster training.
        #
        # NOTE: You can skip the subsampling part and just do step 2 to get
        # your model up and running.
            
        # NOTE 2: You will perform token-based subsampling based on the probabilities in
        # word_to_sample_prob. When subsampling, you are modifying the sequence itself 
        # (like deleting an item in a list). This action effectively makes the context
        # window  larger for some target words by removing context words that are common
        # from a particular context before the training occurs (which then would now include
        # other words that were previously just outside the window).

        self.full_token_sequence_as_ids = []
        for token in all_tokens:
            if token in self.word_to_index:
                self.full_token_sequence_as_ids.append(self.word_to_index[token])
            else:
                self.full_token_sequence_as_ids.append(self.word_to_index['unk'])

        self.full_token_sequence_as_ids = [token for token in self.full_token_sequence_as_ids if probabilities[token] >= self.rng.random()]

        # Helpful print statement to verify what you've loaded
        print('Loaded all data from %s; saw %d tokens (%d unique)' \
              % (file_name, len(self.full_token_sequence_as_ids),
                 len(self.word_to_index)))
        

    
    #This helper function rounds the array, but preserves the sum, to help us in creating the sampled table
    def custom_round(self, arr, target):
        rounded_arr = np.round(arr).astype(int)
        current_sum = np.sum(rounded_arr)
        diff = target - current_sum
    
        if diff == 0:
            return rounded_arr
    
        fractional_parts = arr - np.floor(arr)
        abs_diffs = np.abs(arr - rounded_arr)
        sorted_indices = np.argsort(abs_diffs)
    
        if diff > 0:  # Need to increment
            for idx in sorted_indices:
                if diff == 0:
                    break
                if rounded_arr[idx] < np.ceil(arr[idx]):
                    rounded_arr[idx] += 1
                    diff -= 1
        elif diff < 0:  # Need to decrement
            for idx in sorted_indices:
                if diff == 0:
                    break
                if rounded_arr[idx] > np.floor(arr[idx]):
                    rounded_arr[idx] -= 1
                    diff += 1
                    
        return rounded_arr
    
    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        '''
        Generates a big list data structure that we can quickly randomly index into
        in order to select a negative training example (i.e., a word that was
        *not* present in the context). 
        '''       
        
        # Step 1: Figure out how many instances of each word need to go into the
        # negative sampling table. 
        #
        # HINT: np.power and np.fill might be useful here        
        print("Generating sampling table")

        index_probs = [self.word_counts[self.index_to_word[x]] for x in self.index_to_word]
        index_probs = np.power(np.array(index_probs), exp_power)
        index_probs = index_probs / np.sum(index_probs) * table_size
        index_probs = self.custom_round(index_probs, int(table_size))

        # Step 2: Create the table to the correct size. You'll want this to be a
        # numpy array of type int

        table = np.zeros(int(table_size))

        # Step 3: Fill the table so that each word has a number of IDs
        # proportionate to its probability of being sampled.
        #
        # Example: if we have 3 words "a" "b" and "c" with probabilites 0.5,
        # 0.33, 0.16 and a table size of 6 then our table would look like this
        # (before converting the words to IDs):
        #
        # [ "a", "a", "a", "b", "b", "c" ]
        #
        table = np.repeat(range(len(index_probs)), index_probs)
        self.rng.set_max_val(len(table))
        self.negative_sampling_table = table


    def generate_negative_samples(self, cur_context_word_id, num_samples):
        '''
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        '''

        results = []

        # Create a list and sample from the negative_sampling_table to
        # grow the list to num_samples, avoiding adding a negative example that
        # has the same ID as the current context_word

        table = self.negative_sampling_table

        for n in range(num_samples):
            random_int = self.rng.randint()
            index = table[random_int]
            while index == cur_context_word_id:
                random_int = self.rng.randint()
                index = table[random_int]
            results.append(index)

        return results


# ## Create the corpus
# 
# Now that we have code to turn the text into training data, let's do so. We've provided several files for you to help:
# 
# * `reviews-word2vec.tiny.txt` -- use this to debug your corpus reader
# * `reviews-word2vec.med.txt` -- use this to debug/verify the whole word2vec works
# * `reviews-word2vec.large.txt.gz` -- use this when everything works to generate your vectors for later parts
# * `reviews-word2vec.HUGE.gz` -- _do not use this_ unless (1) everything works and (2) you really want to test/explore. This file is not needed at all to do your homework.
# 
# We recommend starting to debug with the first file, as it is small and fast to load (quicker to find bugs). When debugging, we recommend setting the `min_token_freq` argument to 2 so that you can verify that part of the code is working but you still have enough word types left to test the rest.
# 
# You'll use the remaining files later, where they're described.
# 
# In the next cell, create your `Corpus`, read in the data, and generate the negative sampling table.

# In[59]:


# min_token_freq = 10

corpus = Corpus(RandomNumberGenerator(10000))
# corpus.load_data('reviews.f25.large.txt', min_token_freq)
# corpus.generate_negative_sampling_table()


# # ## Generate the training data
# # 
# # Once we have the corpus ready, we need to generate our training dataset. Each instance in the dataset is a target word and positive and negative examples of contexts words. Given the target word as input, we'll want to predict (or not predict) these positive and negative context words as outputs using our network. Your task here is to create a python `list` of instances. 
# # 
# # Your final training data should be a list of tuples in the format ([target_word_id], [word_id_1, ...], [predicted_labels]), where each item in the list is a list:
# # 1. The first item is a list consisting only of the target word's ID.
# # 2. The second item is a list of word ids for both context words and negative samples 
# # 3. The third item is a list of labels to predicted for each of the word ids in the second list (i.e., `1` for context words and `0` for negative samples). 
# # 
# # You will feed these tuples into the PyTorch `DatasetLoader` later that will do the converstion to `Tensor` objects. You will need to make sure that all of the lists in each tuple are `np.array` instances and are not plain python lists for this `Tensor` converstion to work.

# # In[61]:


# window_size = 2
# num_negative_samples_per_target = 2


# max_context_words = window_size*2*(1+num_negative_samples_per_target)

# training_data = []
    
# # Loop through each token in the corpus and generate an instance for each, 
# # adding it to training_data

# seq = corpus.full_token_sequence_as_ids

# for index, token in enumerate(tqdm(seq)):

#     # if corpus.index_to_word[token] == 'unk':
#     #     continue

#     context_words = []
#     context_labels = []
    
#     # For exach target word in our dataset, select context words 
#     # within +/- the window size in the token sequence

#     for i in range(max(0, index-window_size), min(len(training_data), index+window_size+1)):
#         if i == index:
#             continue
#         context_words.append(seq[i])
#         context_labels.append(1)
    
#     # For each positive target, we need to select negative examples of
#     # words that were not in the context. Use the num_negative_samples_per_target
#     # hyperparameter to generate these, using the generate_negative_samples()
#     # method from the Corpus class
    
#     for i in range(len(context_words)):
#         context_words.extend(corpus.generate_negative_samples(context_words[i], num_negative_samples_per_target))
#         context_labels.extend([0]*num_negative_samples_per_target)

#     # NOTE: this part might not make sense until later when you do the training 
#     # so feel free to revisit it to see why it happens.
#     #
#     # Our training will use batches of instances together (compare that 
#     # with HW1's SGD that used one item at a time). PyTorch will require
#     # that all instances in a batches have the same size, which creates an issue
#     # for us here since the target wordss at the very beginning or end of the corpus
#     # have shorter contexts. 
#     # 
#     # To work around these edge-cases, we need to ensure that each instance has
#     # the same size, which means it needs to have the same number of positive
#     # and negative examples. Since we are short on positive examples here (due
#     # to the edge of the corpus), we can just add more negative samples.
#     #
#     # YOUR TASK: determine what is the maximum number of context words (positive
#     # and negative) for any instance and then, for instances that have fewer than
#     # this number of context words, add in negative examples.
#     #
#     # NOTE: The maximum is fixed, so you can precompute this outside the loop
#     # ahead of time.

#     if len(context_words) < max_context_words:
#         num_to_generate = max_context_words - len(context_words)
#         context_words.extend(corpus.generate_negative_samples(token, num_to_generate))
#         context_labels.extend([0]*num_to_generate)
#     training_data.append( (np.array([token]), np.array(context_words), np.array(context_labels)) )


# # ## Create the network
# # 
# # We'll create a new neural network as a subclass of `nn.Module` like we did in Homework 1. However, _unlike_ the network you built in Homework 1, we do not need to used linear layers to implement word2vec. Instead, we will use PyTorch's `Emedding` class, which maps an index (e.g., a word id in this case) to an embedding. 
# # 
# # Roughly speaking, word2vec's network makes a prediction by computing the dot product of the target word's embedding and a context word's embedding and then passing this dot product through the sigmoid function ($\sigma$) to predict the probability that the context word was actually in the context. The homework write-up has lots of details on how this works. Your `forward()` function will have to implement this computation.

# # In[63]:


# class Word2Vec(nn.Module):
    
#     def __init__(self, vocab_size, embedding_size):
#         super(Word2Vec, self).__init__()
        
#         # Save what state you want and create the embeddings for your
#         # target and context words
#         self.target_embeddings = nn.Embedding(vocab_size, embedding_size)
#         self.context_embeddings = nn.Embedding(vocab_size, embedding_size)

#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size


#         self.sigmoid_layer = nn.Sigmoid()
        
#         # Once created, let's fill the embeddings with non-zero random
#         # numbers. We need to do this to get the training started. 
#         #
#         # NOTE: Why do this? Think about what happens if all the embeddings
#         # are all zeros initially. What would the predictions look like for
#         # word2vec with these embeddings and how would the updated work?
        
#         self.init_emb(init_range=0.5/self.vocab_size)
        
#     def init_emb(self, init_range):
        
#         # Fill your two embeddings with random numbers uniformly sampled
#         # between +/- init_range

#         nn.init.uniform_(self.target_embeddings.weight, a=-init_range, b=init_range)
#         nn.init.uniform_(self.context_embeddings.weight, a=-init_range, b=init_range)
        
#     def forward(self, target_word_id, context_word_ids):
#         ''' 
#         Predicts whether each context word was actually in the context of the target word.
#         The input is a tensor with a single target word's id and a tensor containing each
#         of the context words' ids (this includes both positive and negative examples).
#         '''
        
#         # NOTE 1: This is probably the hardest part of the homework, so you'll
#         # need to figure out how to do the dot-product between embeddings and return
#         # the sigmoid. Be prepared for lots of debugging. For some reference,
#         # our implementation is three lines and really the hard part is just
#         # the last line. However, it's usually a matter of figuring out what
#         # that one line looks like that ends up being the hard part.
        
#         # NOTE 2: In this homework you'll be dealing with *batches* of instances
#         # rather than a single instance at once. PyTorch mostly handles this
#         # seamlessly under the hood for you (which is very nice) but batching
#         # can show in weird ways and create challenges in debugging initially.
#         # For one, your inputs will get an extra dimension. So, for example,
#         # if you have a batch size of 4, your input for target_word_id will
#         # really be 4 x 1. If you get the embeddings of those targets,
#         # it then becomes 4x50! The same applies to the context_word_ids, except
#         # that was alreayd a list so now you have things with shape 
#         #
#         #    (batch x context_words x embedding_size)
#         #
#         # One of your tasks will be to figure out how to get things lined up
#         # so everything "just works". When it does, the code looks surprisingly
#         # simple, but it might take a lot of debugging (or not!) to get there.
        
#         # NOTE 3: We *strongly* discourage you from looking for existing 
#         # implementations of word2vec online. Sadly, having reviewed most of the
#         # highly-visible ones, they are actually wrong (wow!) or are doing
#         # inefficient things like computing the full softmax instead of doing
#         # the negative sampling. Looking at these will likely leave you more
#         # confused than if you just tried to figure it out yourself.
        
#         # NOTE 4: There many ways to implement this, some more efficient
#         # than others. You will want to get it working first and then
#         # test the timing to see how long it takes. As long as the
#         # code works (vector comparisons look good) you'll receive full
#         # credit. However, very slow implementations may take hours(!)
#         # to converge so plan ahead.
        
        
#         # Hint 1: You may want to review the mathematical operations on how
#         # to compute the dot product to see how to do these
        
#         # Hint 2: the "dim" argument for some operations may come in handy,
#         # depending on your implementation


#         # TODO: Implement the forward pass of word2vec
        
        
        
#         target = self.target_embeddings(target_word_id) #select appropriate row
#         context = self.context_embeddings(context_word_ids) #select appropriate rows

#         context = torch.transpose(context, 1, 2)        
#         output_layer = torch.bmm(target, context).squeeze(1)

            
#         return self.sigmoid_layer(output_layer)
    


# # ## Train the network!
# # 
# # Now that you have data in the right format and a neural network designed, it's time to train the network and see if it's all working. The trainin code will look surprisingly similar at times to your pytorch code from Homework 1 since all networks share the same base training setup. However, we'll add a few new elements to get you familiar with more common training techniques. 
# # 
# # For all steps, be sure to use the hyperparameters values described in the write-up.
# # 
# # 1. Initialize your optimizer and loss function 
# # 2. Create your network
# # 3. Load your dataset into PyTorch's `DataLoader` class, which will take care of batching and shuffling for us (yay!)
# # 4. Create a new `SummaryWriter` to periodically write our running-sum of the loss to a tensorboard
# # 5. Train your model 
# # 
# # Two new elements show up. First, we'll be using `DataLoader` which is going to sample data for us and put it in a batch (and also convert the data to `Tensor` objects. You can iterate over the batches and each iteration will return all the items eventually, one batch at a time (a full epoch's worth).
# # 
# # The second new part is using `wandb`. As you might have noticed in Homework 1, training neural models can take some time. [Weights & Biases](https://wandb.ai/) is a handy web-based view that you can check during training to see how the model is doing. We'll use it here and periodically log a running sum of the loss after a set number of steps. The Homework write up has a plot of what this looks like. We'll be doing something simple here with wandb but it will come in handy later as you train larger models (for longer) and may want to visually check if your model is converging and is [easy to integrate](https://docs.wandb.ai/guides/integrations/pytorch).
# # 
# # Once you get the code working, to start training, we recommend training on the `reviews-word2vec.med.txt` dataset. This data is small enough you can get through an epoch in a few minutes (or less) while still being large enough you can test whether the model is learning anything by examining common words. Below this cell we've added a few helper functions that you can use to debug and query your model. In particular, the `get_neighbors()` function is a great way to test: if your model has learned anything, the nearest neighbors for common words should seem reasonable (without having to jump through mental hoops). An easy word to test on the `med` data is "january" which should return month-related words as being most similar.
# # 
# # **NOTE**: Since we're training biographies, the text itself will be skewed towards words likely to show up biographices--which isn't necessary like "regular" text. You may find that your model has few instances of words you think are common, or that the model learns poor or unusual neighbors for these. When querying the neighbors, it can help to think of which words you think are likely to show up in biographies on Wikipedia and use those as probes to see what the model has learned.
# # 
# # Once you're convinced the model is learning, switch to the `med` data and train your model as specified in the PDF. Once trained, save your model using the `save()` function at the end of the notebook. This function records your data in a common format for word2vec vectors and lets you load the vectors into other libraries that have more advanced functionality. In particular, you can use the [gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) code in other notebook included to explore the vectors and do simple vector analogies.

# # In[ ]:


# # TODO: Set your training stuff, hyperparameters, models, etc. here
# embedding_size = 50
# vocab_size = len(corpus.word_to_index)
# num_epochs = 1
# batch_size = 1024

# model = Word2Vec(vocab_size, embedding_size)
# lossobj = torch.nn.BCELoss()
# optimizer = torch.optim.AdamW(model.parameters())

# model.train()
# # TODO: Initialize weights and biases (wandb) here 

# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="rmsh-university-of-michigan",
#     # Set the wandb project where this run will be logged.
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata.
#     config={
#         "batch_size": batch_size,
#         "epochs": num_epochs,
#     },
# )

# run.watch(model, log_freq=100)

# # HINT: wrapping the epoch/step loops in nested tqdm calls is a great way
# # to keep track of how fast things are and how much longer training will take

# for epoch in range(num_epochs):

#     loss_sum = 0

#     dataloader = DataLoader(training_data, batch_size = batch_size)
#     # TODO: use your DataLoader to iterate over the data

#     for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
#         # NOTE: since you created the data as a tuple of three np.array instances,
#         # these have now been converted to Tensor objects for us
#         target_ids, context_ids, labels = data
        
#         # TODO: Fill in all the training details here
#         optimizer.zero_grad()
#         output = model(target_ids, context_ids)

        
#         loss = lossobj(output, labels.float())
#         loss.backward()
#         optimizer.step()

#         loss_sum += loss
        
#         # TODO: Based on the details in the Homework PDF, periodically
#         # report the running-sum of the loss to Weights & Biases (wandb).
#         # Be sure to reset the running sum after reporting it.
#         if step % 100 == 0:
#             run.log({"loss":loss_sum})
#             loss_sum = 0
#         # TODO: it can be helpful to add some early stopping here after
#         # a fixed number of steps (e.g., if step > max_steps)
        
# run.finish()

# # once you finish training, it's good practice to switch to eval.
# model.eval()


# # In[ ]:


# # Batch size vs estimated training time

# import matplotlib.pyplot as plt
# batch_sizes = [2, 8, 32, 64, 128, 256, 512, 1024, 2048]
# times = [900, 210, 54, 27, 12, 7, 4, 3, 1.2]
# times = [math.log(time) for time in times]
# for xy in zip(batch_sizes, times):                                       
#     plt.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data') 
# plt.plot(batch_sizes, times)
# plt.xlabel("Batch size")
# plt.ylabel("Estimated time to train (log minutes)")
# plt.title("Batch size vs time to train (TQDM estimation)")
# plt.show()


# # In[ ]:





# # ## Verify things are working
# # 
# # Once you have an initial model trained, try using the following code to query the model for what are the nearest neighbor of a word. This code is intended to help you debug

# # In[ ]:


# def get_neighbors(model, word_to_index, target_word):
#     """ 
#     Finds the top 10 most similar words to a target word
#     """
#     outputs = []
#     for word, index in tqdm(word_to_index.items(), total=len(word_to_index)):
#         similarity = compute_cosine_similarity(model, word_to_index, target_word, word)
#         result = {"word": word, "score": similarity}
#         outputs.append(result)

#     # Sort by highest scores
#     neighbors = sorted(outputs, key=lambda o: o['score'], reverse=True)
#     return neighbors[1:11]

# def compute_cosine_similarity(model, word_to_index, word_one, word_two):
#     '''
#     Computes the cosine similarity between the two words
#     '''
#     try:
#         word_one_index = word_to_index[word_one]
#         word_two_index = word_to_index[word_two]
#     except KeyError:
#         return 0

#     embedding_one = model.target_embeddings(torch.LongTensor([word_one_index]))
#     embedding_two = model.target_embeddings(torch.LongTensor([word_two_index]))
#     similarity = 1 - abs(float(cosine(embedding_one.detach().squeeze().numpy(),
#                                       embedding_two.detach().squeeze().numpy())))
#     return similarity


# # In[ ]:


# get_neighbors(model, corpus.word_to_index, "recommend")


# # In[ ]:


# get_neighbors(model, corpus.word_to_index, "hell")


# # # Save your vectors for the gensim inspection part!
# # 
# # Once you have a fully trained model, save it using the code below. Note that we only save the `target_embeddings` from the model, but you could modify the code if you want to save the context vectors--or even try doing fancier things like saving the concatenation of the two or the average of the two!

# # In[46]:


# def save(model, corpus, filename):
#     '''
#     Saves the model to the specified filename as a gensim KeyedVectors in the
#     text format so you can load it separately.
#     '''

#     # Creates an empty KeyedVectors with our embedding size
#     kv = KeyedVectors(vector_size=model.embedding_size)        
#     vectors = []
#     words = []
#     # Get the list of words/vectors in a consistent order
#     for index in trange(model.target_embeddings.num_embeddings):
#         word = corpus.index_to_word[index]
#         vectors.append(model.target_embeddings(torch.LongTensor([index])).detach().numpy()[0])
#         words.append(word)

#     # Fills the KV object with our data in the right order
#     kv.add_vectors(words, vectors) 
#     kv.save_word2vec_format(filename, binary=False)


# # # Save your vectors / data for the pytorch classifier in Part 4!
# # 
# # We'll be to using these vectors later in Part 4. We want to save them in a format that PyTorch can easily use. In particular you'll need to save the _state dict_ of the embeddings, which captures all of its information. 

# # In[49]:


# save(model, corpus, "ramesh_model.kv")


# # We will also need the mapping from word to index so we can figure out which embedding to use for different words. Save the `corpus` objects mapping to a file using your preferred format (e.g., pickle or json).

# # In[52]:


# with open("corpus.pkl", 'wb') as file_handler:
#         pickle.dump(corpus, file_handler)


# # In[ ]:




