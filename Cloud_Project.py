#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from nltk.stem import PorterStemmer, SnowballStemmer
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nltk
stemmer = SnowballStemmer("english")
import random
import json

import torch
import pandas as pd

# In[75]:


ds=pd.read_csv('food.csv')
ds


# In[76]:


ds.dropna()




# In[83]:


#bagofwords
def bagofwords(t_sentence, words):
    s_words = [stemmer.stem(w) for w in t_sentence]
    # initialize bag with 0 
    b = np.zeros(len(words), dtype=np.float32)
    for index, w in enumerate(words):
        if w in s_words: 
            b[index] = 1

    return b


# In[84]:


with open('intents.json', 'r') as f:
    intents = json.load(f)

a_words = []
tags = []
pairs = []

for i in intents['intents']:
    t = i['tag']
    tags.append(t)
    for p in i['patterns']:
        #tokenisation
        word = word_tokenize(p)
        #words list
        a_words.extend(word)
        #pairs
        pairs.append((word, t))


ignwords = ['?', '.', '!','&']

#ignore all punctuations
#stemming
a_words_stemmed = []
for w in a_words:
    if w not in ignwords:
        a_words_stemmed.append(stemmer.stem(w))
        

#duplicates and sorting
allwords = sorted(set(a_words_stemmed))
tags = sorted(set(tags))





# In[86]:


#training data
X_train = []
y_train = []
for (p,tag) in pairs:
    bag = bagofwords(p, allwords)
    label = tags.index(tag)
    X_train.append(bag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


# In[87]:


# Hyper-parameters 
learning_rate = 0.001
i_size = len(X_train[0])
h_size = 8
o_size = len(tags)


# In[88]:


#NeuralNetwork
class NNet(nn.Module):
    def __init__(self, i_size, n_classes):
        super(NNet, self).__init__()
        self.l1 = nn.Linear(i_size, 8) 
        self.l2 = nn.Linear(8, 10) 
        self.l3 = nn.Linear(10, 6)
        self.l4 = nn.Linear(6, n_classes)
        # Define LeakyReLU with different negative slopes for each layer
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.02)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.03)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.lrelu1(out)
        out = self.l2(out)
        out = self.lrelu2(out)
        out = self.l3(out)
        out = self.lrelu3(out)
        out = self.l4(out)
        return out


# In[89]:


class CDataset(Dataset):

    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train

    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    
    def __len__(self):
        self.n_samples = len(X_train)
        return self.n_samples

dataset = CDataset()
tloader = DataLoader(dataset=dataset,
                          batch_size=10,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[90]:


m = NNet(i_size, o_size).to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

# Training
for epoch in range(1000):
    for (inputs, targets) in tloader:
        inputs = inputs.to(device)
        targets = targets.to(dtype=torch.long).to(device)
        
        # Forward pass
        predictions = m(inputs)
        loss = nn.CrossEntropyLoss()(predictions, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}')



# In[91]:


import spacy
import re
import time
import uuid
import pandas as pd

# Load the English language model
nlp = spacy.load("en_core_web_sm")


menu = []
df = pd.read_csv('Item_to_id.csv')
for index, row in df.iterrows():
    item_name = row['Name']
    price = row['Price']
    menu.append({"item_name": item_name, "price": price})


# In[92]:


cart = []

def display_menu():
    menu_display="Menu:\n"
    for idx, item in enumerate(menu, start=1):
        menu_display+= f"{idx}. {item['item_name']} - ${item['price']}\n"
    menu_display+="you can place an order by typing 'order item_name', add an item to your cart with 'add quantity item_name', proceed to checkout by typing 'checkout'.\n"
    return menu_display

def display_cart():
    #print(f"{bot_name}: ")
    #print("Cart:")
    cart_display = "Cart:\n"
    if cart:
        for idx, item in enumerate(cart, start=1):
            cart_display+=f"{idx}. {item['item_name']} - ${item['price']}\n"
        cart_display+="You can proceed to checkout by typing 'checkout'.\n"
    else:
        cart_display+="Your cart is empty.you can place an order by typing 'order item_name', add an item to your cart with 'add quantity item_name', proceed to checkout by typing 'checkout'.\n"
        
    return cart_display

# In[93]:


def process_input(user_input):
    response_dct={}
    response=""
    user_input = user_input.lower().strip()
    if user_input == 'menu':
        response_dct= {'intent': 'show_menu'}
    elif user_input == 'cart':
        response_dct= {'intent': 'show_cart'}
    elif user_input.startswith('order'):
        match = re.match(r'order\s+(.*)', user_input)
        if match:
            item_name = match.group(1).strip()
            matching_items = [item for item in menu if item['item_name'].lower() == item_name]
            if matching_items:
                if len(matching_items) > 1:
                    #response=f"{bot_name}: Multiple items found with that name. Please specify.\n"
                    response_dct={'intent': 'ambiguous_item_name', 'matching_items': matching_items, 'parameters': {'item_name': item_name}}
                else:
                    response_dct= {'intent': 'order_item', 'parameters': {'item_name': matching_items[0]['item_name']}}
            else:
                #response=f"{bot_name}: Item not found in the menu.\n"
                response_dct= {'intent': 'unknown'}
    elif user_input.startswith('add'):
        match = re.match(r'add\s+(\d+)\s+(.*)', user_input)
        if match:
            response_dct= {'intent': 'add_to_cart', 'parameters': {'quantity': int(match.group(1)), 'item_name': match.group(2).strip()}}
    elif user_input == 'checkout':
        response_dct={'intent': 'checkout'}
    elif user_input == 'exit':
        response_dct={'intent': 'exit'}
    else:
        t_sentence = word_tokenize(user_input)
        X = bagofwords(t_sentence, allwords)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        out = m(X)
        _, pred = torch.max(out, dim=1)

        tag = tags[pred.item()]

        probs = torch.softmax(out, dim=1)
        prob = probs[0][pred.item()]
        if prob.item() > 0.75:
            found_tag = False
            for i in intents['intents']:
                if tag == i["tag"]:
                    responses = i['responses']
                    if len(responses) > 1:
                        response= random.choice(responses)
                    else:
                        response=responses[0]
                    found_tag = True
                    break
            if not found_tag:
                response ="I do not understand."
        else:
            response ="I do not understand."
    return response,response_dct


# In[94]:


def generate_transaction_id():
    timestamp = int(time.time() * 1000)
    transaction_id = f"{timestamp}"
    return transaction_id


# In[95]:


def process_order(intent, parameters):
    response=""
    if intent == 'show_menu':
        response=display_menu()
    elif intent == 'show_cart':
        response=display_cart()
    elif intent == 'order_item':
        item_name = parameters.get('item_name')
        matching_items = [item for item in menu if item['item_name'].lower() == item_name.lower()]
        if matching_items:
            cart.append(matching_items[0])
            response=f"Ordered {item_name}.\n"
            response+="you can add an item to your cart with 'add quantity item_name', view the cart with 'cart', or proceed to checkout by typing 'checkout'."
        else:
            response=f"Item not found in the menu."
    elif intent == 'add_to_cart':
        quantity = parameters.get('quantity')
        item_name = parameters.get('item_name')
        matching_items = [item for item in menu if item['item_name'].lower() == item_name.lower()]
        if matching_items:
            cart.extend([matching_items[0]] * quantity)
            response=f"Added {quantity} {item_name}(s) to cart.\n"
            response+="You can proceed to checkout by typing 'checkout' or add an item to your cart with 'add quantity item_name'"
        else:
            response=f"Item not found in the menu."
    elif intent == 'checkout':
        response=f"Processing checkout...\n"
        response+=display_cart()
        total_price = sum(item['price'] for item in cart)
        response+=f"Total: ${total_price}\n"
        id = generate_transaction_id()
        f = "feedback"
        response+=f"Thank you for your order!. Your order id is "+id + ". You can pay at the time of delivery. Please provide your feedback.\n"
        cart.clear()
    elif intent == 'exit':
        response=f"Exiting the chatbot...\n"
        return response,True
    else:
        response=f"Sorry, I didn't understand.\n"

    return response,False


# In[96]:

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    parsed_input = process_input(userText)
    
  
    if parsed_input[0]!="":
        return parsed_input[0]
    
    else:
        
 
        response, exit_chatbot = process_order(parsed_input[1]['intent'], parsed_input[1].get('parameters', {}))
      
        return response.replace('\n', '<br>')
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)


