# -*- coding: utf-8 -*-

import requests

def login(credentials):
  url = "https://assistive.cin.ufpe.br/reaact-server/auth"

  res = requests.post(url, json=credentials, verify=False)

  user_token = res.json()["token"]

  return user_token

credentials = {
    "username": "pranchacomunicatea@reaact.com.br",
    "password": "ComunicaTEA2020"
}

user_token = login(credentials)

def vocabularies(token):
  
  url = "https://assistive.cin.ufpe.br/reaact-server/vocabularies"

  res = requests.get(url,headers={"Authorization": "Bearer {0}".format(token)},verify=False)

  return res.json()

vocabulary = vocabularies(user_token)[0]

only_first_page = [card for card in vocabulary['cards'] if card["categoryId"] == "00000000-0000-0000-0000-000000000000"]

from anytree import Node

all_nodes = {}

root = Node("00000000-0000-0000-0000-000000000000")

all_nodes["00000000-0000-0000-0000-000000000000"] = root

def create_tree(cards):
  for card in cards:
    if card['type'] != "EMPTY":
      parent_id = card["categoryId"]
      id_ = card['id']
      all_nodes[id_] = Node(id_)
      
      all_nodes[id_].parent = all_nodes[parent_id]
      if len(card['cards']) > 0:
        create_tree(card['cards'])
create_tree(only_first_page)

cards = []
exclude_categories = ["Perguntas","Algo a dizer"]

def extract_cards(cards):
  extracted = []
  for item in cards:
    if item['type'] == "ELEMENT": 
      tokens = item['label'].split(",")
      for t in tokens: 
        extracted.append({
            "id": item['id'],
            "label": t.strip(),
            "extension": item['extension'],
        })
    elif item['type'] == "CATEGORY":
      if item['label'] not in exclude_categories:
        extracted = extracted + extract_cards(item['cards'])
  return extracted


cards = extract_cards(only_first_page)
len(cards)


import io
f = io.open("./files/CETENFolha-1.0.cg", mode="r", encoding="latin-1")
lines = f.readlines()

import re
from tqdm import tqdm
regex = r"\[(.*?)\]"
lemma_freq = {}
for l in tqdm(lines):
  if not l.startswith("$") and not l.startswith("<"):
    parts = l.split("\t")
    if len(parts) > 1:
      w, annotation = parts[0],parts[1]
      matches = re.findall(regex, annotation, re.MULTILINE)
      if len(matches) > 0:
        lemma = matches[0]
        w = w.lower()
        lemma = lemma.lower()
        if w not in lemma_freq:
          lemma_freq[w] = []
        lemma_freq[w].append(lemma)

from collections import Counter
freqs = {}
for w in lemma_freq:
  freqs[w] = [t[0] for t in Counter(lemma_freq[w]).most_common()]

freqs['pesado']

all_words = []
for card in cards:
  all_words.append(card['label'].lower())

TOKENIZER_PATH = "./files/telegraphic_corpus_mwe_tokenizer_2.json"
MODEL_PATH = "./files/trained-bertimbau-1503"

from transformers import PreTrainedTokenizerFast, BertForMaskedLM

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.pad_token = "[PAD]"
tokenizer.sep_token = "[SEP]"
tokenizer.mask_token = "[MASK]"
tokenizer.cls_token = "[CLS]"
tokenizer.unk_token = "[UNK]"

exclude = ["querido",'vestido','salgado',"envergonhado","preocupado","cansado","enjoado","animado","desmotivado",
           "desanimado","pesado","errado","preparado","desorganizado","desconhecido","atrasado","envergonhado","preocupado"]
cards_vocab = {}
for card in cards:
  w = "_".join(card['label'].lower().split(" "))
  if "_" in w:
    cards_vocab[w] = card
  else:
    # lemmatized = lemmatize(w)
    
    changed = False
    if w in freqs:
      n_w = freqs[w][0]

      if w.strip() not in exclude:
        if n_w != w and n_w not in all_words and n_w in list(tokenizer.get_vocab().keys()) and n_w not in list(cards_vocab.keys()) :
          cards_vocab[n_w] = card
          print(w,n_w)
          changed = True
    if not changed:
      cards_vocab[w] = card
    # if len(lemmatized) == 1:
    #   if lemmatized[0] != w and lemmatized[0] not in all_words:
    #     print(lemmatized[0],w)
    #     cards_vocab[lemmatized[0]] = card
    # cards_vocab[w]

model = BertForMaskedLM.from_pretrained(MODEL_PATH)

words = list(cards_vocab.keys())
ids = tokenizer.convert_tokens_to_ids(words)

import torch.nn.functional as F
import numpy as np

def get_top_k(sentence, top_k=3):
  text = " ".join(sentence+['[MASK]','.'])
  input = tokenizer(text, return_tensors='pt')
  output = model(input['input_ids'], input['attention_mask'])

  predictions = F.softmax(output[0], dim=-1)

  predictions.size()

  mask_idx = input['input_ids'].tolist()[0].index(tokenizer.mask_token_id)
  probs = predictions[0, mask_idx, :]
  top_words = np.array(words)[probs[ids].topk(top_k)[1]]
  # top_pictograms = [cards_vocab[w] for w in top_words]
  return top_words
get_top_k(["eu querer"],66)

from scipy import stats


def get_percentil(sentence, percentil=0.01):
  z = stats.norm.ppf(1-percentil)
  text = " ".join(sentence+['[MASK]','.'])
  input = tokenizer(text, return_tensors='pt')
  output = model(input['input_ids'], input['attention_mask'])

  predictions = F.softmax(output[0], dim=-1)

  predictions.size()

  mask_idx = input['input_ids'].tolist()[0].index(tokenizer.mask_token_id)
  probs = predictions[0, mask_idx, :]
  my_probs = probs[ids]

  std = my_probs.std()
  mean = my_probs.mean()

  x = (z*std)+mean
  # return my_probs[my_probs > x]
  return np.array(words)[my_probs > x]
len(get_percentil(["eu", "querer"],0.3))

"""## Mapping"""

import pandas as pd
cards_vocab['comer']
mapping_dicts = []
for word in cards_vocab:
  mapping_dicts.append({
      "word": word,
      "id": cards_vocab[word]['id'],
      "extension": cards_vocab[word]['extension'],
      "label": cards_vocab[word]['label'],
  })
mapping = pd.DataFrame(mapping_dicts)
mapping.head()

def word_to_id(word):
  for i, row in mapping.loc[mapping["word"] == word].iterrows():
    return row.to_dict()
  return None

def id_to_word(id_):
  for i, row in mapping.loc[mapping["id"] == id_].iterrows():
    return row.to_dict()
  return None

def ids_to_words(ids):
  return [id_to_word(id_)['word'] for id_ in ids if id_to_word(id_)['word'] is not None]

def words_to_ids(words):
  return [word_to_id(word)['id'] for word in words if word_to_id(word)['id'] is not None]

words_to_ids(["eu","querer"])

def get_ancestors(id_):
  return [a.name for a in all_nodes[id_].ancestors]
get_ancestors("6ae59b66-bc85-4686-be4a-156fbabfab87")

from nltk.util import flatten

def prediction(ids,method="percentil",param=0.4):
  words = ids_to_words(ids)
  if method == "percentil":
    predicted_words = get_percentil(words, param)
  else:
    if param == 0.4:
      param = 20
    predicted_words = get_top_k(words, param)
  predicted_ids = words_to_ids(predicted_words)
  activate_ids = flatten([get_ancestors(id_) for id_ in predicted_ids]) + predicted_ids
  return list(set(activate_ids))


print(prediction(words_to_ids(["eu", "estar"]),param=0.5))

"""## Server"""

from flask import Flask
from flask import request,render_template,send_from_directory
from flask_cors import CORS
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)

# @app.route('/<path:path>', methods=['GET'])
# def static_proxy(path):
#   return send_from_directory('./reaact-pictogram-prediction/www/', path)

@app.route('/')
def hello():
    return {"nada":1}
  
@app.route("/model",methods=['POST'])
def inference():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
      json = request.json
  else:
      return 'Content-Type not supported!'

  body = json
  method = body["method"]
  if "param" in body:
    param = body["param"]
  sentence = body['sentence']
  predicted = prediction(sentence,method,param)
  return {"show":predicted}
  
if __name__ == "__main__":
  app.run()