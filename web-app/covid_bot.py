
# MODEL IMPORTS
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.


from transformers import AutoTokenizer, AutoModel
import torch

# UTILITY IMPORTS 
import pandas as pd
from scipy import spatial
from scipy.special import softmax
import numpy as np
import re
from flask import jsonify

#@title Configure the model { run: "auto" }
BERT_MODEL = "https://tfhub.dev/google/experts/bert/pubmed/2" # @param {type: "string"} ["https://tfhub.dev/google/experts/bert/wiki_books/2", "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  "https://tfhub.dev/google/experts/bert/pubmed/2", "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1"


def preprocess_sentences(input_sentences):
    input_sentences = [input_sentence.lower() for input_sentence in input_sentences]
    # input_sentences = [re.sub(r"[^a-zA-Z0-9]+", ' ', input_sentence) 
    # for input_sentence in input_sentences]
    return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I) 
            for input_sentence in input_sentences]

def similarity(vector1,vector2):
  cosine_similarity = 1 - spatial.distance.cosine(vector1, vector2)
  # print(cosine_similarity)
  return cosine_similarity

# Use pre-trained bert trained on pubmed articles to generate embeddings
def generate_embeddings(df):
    
  preprocess = hub.load(PREPROCESS_MODEL)
  bert = hub.load(BERT_MODEL)

  questions = df['question'].tolist()
  answers = df['answer'].tolist()

  qn_inputs = preprocess(questions)
  qn_outputs = bert(qn_inputs)
  # print("\nPooled Qn embeddings:")
  # print(qn_outputs["pooled_output"])

  ans_inputs = preprocess(answers)
  ans_outputs = bert(ans_inputs)
  # print("\nPooled Ans embeddings:")
  # print(ans_outputs["pooled_output"])
  
  df['qn_pubmed_bert']=None
  df['ans_pubmed_bert']=None

  for i in range(0,len(df)):
    df['qn_pubmed_bert'].iloc[i] = qn_outputs["pooled_output"][i]
    df['ans_pubmed_bert'].iloc[i] = ans_outputs["pooled_output"][i]
  
  return df
    


# ASK A QUESTION AND FIND QUESTION THAT IS MOST SIMILAR IN THE DATASET 
def generate_top_k_answers(queries,df,k):
  
  # Input qn from user 
  questions = [queries]

  preprocess = hub.load(PREPROCESS_MODEL)
  bert = hub.load(BERT_MODEL)

  # Get embedding of input question
  qn_inputs = preprocess(questions)
  qn_outputs = bert(qn_inputs)
  print("\nPooled Qn embeddings:")
  input_embedding = qn_outputs["pooled_output"]

  # Identify the most similar question using cosine similarity 
  most_similar_qn_index = 0
  similarity_score = 0.0
  for i in range(len(df)):
    sim_score = similarity(softmax(df['qn_pubmed_bert'][i]),softmax(input_embedding))
    if sim_score>similarity_score:
      similarity_score= sim_score
      most_similar_qn_index=i

  # print("\n\n Qn: "+str(most_similar_qn_index)+" : "+str(similarity_score)+"\n")
  # print("Qn: "+df['question'].iloc[most_similar_qn_index])

  # print("Actual Answer:")
  # print(df['answer'].iloc[most_similar_qn_index])


  # FIND SET OF TOP 5 ANSWERS THAT IS MOST SIMILAR TO THE ACTUAL ANSWER IN THE DATASET 

  #Get Actual Answer Embedding
  ans_to_input_embedding = df['ans_pubmed_bert'].iloc[most_similar_qn_index]
  
  # Calculate the Cosine Similarity between the actual answer and all other answers
  cosine_similar_answer_scores = []
  ans_embeddings = df['ans_pubmed_bert'].tolist()
  numRows = len(ans_embeddings)
  for i in range(numRows):
    cosine_similarity_score = similarity(ans_embeddings[i],ans_to_input_embedding)
    cosine_similar_answer_scores.append(cosine_similarity_score)

  # Get the top 5 most similar answers using cosine similarity (incuding the actual answer)
  top_k_answers_index = np.argsort(cosine_similar_answer_scores)[-k:]
  top_k_answer_values = [cosine_similar_answer_scores[i] for i in top_k_answers_index]

  top_k_answers = []
  print("\n\n Top k related answers: \n")
  for i in top_k_answers_index:
    top_k_answers.append(df['answer'].iloc[i])  
    print(df['answer'].iloc[i]+"\n")

  return top_k_answers_index,top_k_answers

 



 




# def text_generation(top_5_answers):
#   # RUN ON PRE-TRAINED COVID-TUNED GPT-2 MODEL 

#   ## Documentation: 
#   ## Collab: How to generate sentence: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb#scrollTo=3kY8P9VG8Gi9
#   ## Article: https://huggingface.co/blog/how-to-generate


#   from transformers import AutoTokenizer, AutoModelWithLMHead

#   tokenizer = AutoTokenizer.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv")

#   model = AutoModelWithLMHead.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv", pad_token_id=tokenizer.eos_token_id)


#   # encode context the generation is conditioned on
#   input_ids = tokenizer.encode(top_5_answers, return_tensors='pt')

#   # set seed to reproduce results. Feel free to change the seed though to get different results
#   tf.random.set_seed(0)

#   # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
#   sample_outputs = model.generate(
#       input_ids,
#       do_sample=True, 
#       max_length=50, 
#       top_k=50, 
#       top_p=0.95, 
#       num_return_sequences=3
#   )

#   return sample_outputs

#   # print("Output:\n" + 100 * '-')
#   # for i, sample_output in enumerate(sample_outputs):
#   #   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

#   # print("\n\n")
#   # print(top_5_answers)









# # def load_embedding_module():
# #   module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')
# #   return module

# #   # # Load the BERT encoder and preprocessing models
# #   # preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
# #   # bert = hub.load('https://tfhub.dev/google/experts/bert/pubmed/2')
# #   # return preprocess, bert


# # def create_qna_embeddings(data, module):  
# #   # NORMAL USE : tf_hub -> 1 sentence : 1 embedding
# #   # Create response embeddings
# #   response_encodings = module.signatures['response_encoder'](
# #     input=tf.constant(preprocess_sentences(data.Answer)),
# #     context=tf.constant(preprocess_sentences(data.Context)))['outputs']
# #   return response_encodings



# # def run_covid_bot(questions,data,module,response_encodings):
  
# #   # Create encodings for test questions
# #   question_encodings = module.signatures['question_encoder'](
# #       tf.constant(preprocess_sentences(questions))
# #   )['outputs']

# #   # Get the responses
# #   test_responses = data.Answer[np.argmax(np.inner(question_encodings, response_encodings), axis=1)]

# #   # Show them in a dataframe
# #   generated_output = pd.DataFrame({'Test Questions': questions, 'Test Responses': test_responses})
# #   return generated_output.to_string() #.to_html()


# tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
# model = AutoModel.from_pretrained("gsarti/biobert-nli")


# def generate_embeddings(sentence):
#   tokenized_sentence = tokenizer.tokenize(sentence)
#   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
#   with torch.no_grad():
#       vector = model(torch.LongTensor([indexed_tokens]).to('cuda'))[1].cpu().numpy().tolist()
#   return vector[0]


# def cosine_smilarity(v1, v2):
#     cosine_similarity = 1 - spatial.distance.cosine(v1, v2)
#     return cosine_similarity



# # def run_covid_bot(input,embeddings):
# #   try:
#     # embeddings = hkl.load('who_covid_19_question_embedding.hkl')
#     # CSV_FILE = "../data/who_covid_19_data.csv"
 
#   #   #Load AutoModel from huggingface model repository
#   #   # model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
#   #   # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
#   #   tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
#   #   model = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")

#   #   #Tokenize sentences
#   #   encoded_input = tokenizer(input, padding=True, truncation=True, max_length=128, return_tensors='pt')


#   #   with torch.no_grad():
#   #       model_output = model(**encoded_input)
#   #       input_embedding = model_output[0][:,0]


#   #   # input_embedding = model.encode(input)
#   #   q_id = 0;
#   #   max_score = 0;
#   #   for i, e in enumerate(embeddings):
#   #       similarity = cosine_smilarity(input_embedding, e)
#   #       if similarity > max_score :
#   #           q_id = i
#   #           max_score = similarity
#   #   result = data["answer"][q_id]
#   #   similar_question = data["question"][q_id]
#   #   return jsonify({'success': True, 'user_question': input , 'similar_question':similar_question, 'similarity': max_score,  'answer': result })
#   # except:
#   #   traceback.print_exc()
#   #   return jsonify({'success': False, 'utterances': None})





# # # ##### Use USE pretrained model to extract response encodings.
# # # import tensorflow as tf
# # # import tensorflow_hub as hub
# # # import tensorflow_text



# # def run_covid_bot(questions):

# #   data = pd.read_excel("../data/WHO_FAQ.xlsx")

# #   # Load module containing USE
# #   module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

# #   # NORMAL USE : tf_hub -> 1 sentence : 1 embedding

# #   # Create response embeddings
# #   response_encodings = module.signatures['response_encoder'](
# #           input=tf.constant(preprocess_sentences(data.Answer)),
# #           context=tf.constant(preprocess_sentences(data.Context)))['outputs']


# #   # Create encodings for test questions
# #   question_encodings = module.signatures['question_encoder'](
# #       tf.constant(preprocess_sentences(questions))
# #   )['outputs']

# #   # Get the responses
# #   test_responses = data.Answer[np.argmax(np.inner(question_encodings, response_encodings), axis=1)]

# #   # Show them in a dataframe
# #   generated_output = pd.DataFrame({'Test Questions': questions, 'Test Responses': test_responses})
# #   return generated_output
 

# # if __name__ == '__main__':
# #   from generate_sentence_embeddings import generate_embeddings
# #   sentence_embeddings = generate_embeddings()
# #   run_covid_bot("what is covid?",sentence_embeddings)




