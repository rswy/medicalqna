# Medical Question & Answering with Deep Learning
- This application is a deep learning based medical Q&A system that is able to serve up responses to medical queries or statements made by patients. The goal is to see how far NLP can be used emulate the kind of interaction between a patient and doctor when the patient visits the clinic 

### Methodology
- Use a BERT model pretrained on MEDLINE/PubMed (https://tfhub.dev/google/experts/bert/pubmed/2)  to generate the embeddings for every question and answer separately
- When a new question from the user comes in, its text embedding will be generated and compared against the question embeddings from the dataset to identify the question most similar to it. Let's call this question 1.
- The answer to question 1 will be compared against 5 other answers that are most similar to it (using cosine similarity)
- The five similar answers will then be used as input to a GPT-2 model (which needs to be fine-tuned) which will generate a single answer from them (This last step requires further training and tweaking of the gpt-2 model to ensure the result is able to make sense of the 5 answers)

### Deployment
- The application is deployed as a flask application and the codes can be found in the web-app folder
### Data
- The data used here is a set of questions and answers from medical conversations made between doctors and patients



## Collab Notebook


Google Collab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/rswy/medicalqna/blob/master/medicalqna.ipynb)

### What you have to run in the collab/jupyter notebook
Steps: 
-1) Clone the codes by running the bottom line

2) Run the pip installation of the specified modules. Alternatively you may run install the python modules from the requirements.txt file

3) cd to the correct folder i.e. cd /web-app

3) Run app.py by clicking on the Right arrow circle button (top left of the cell) next to "!python app.py" to run app.py

4) The output will print a link to the running flask application that looks like this : "Running on http://xxxx.ngrok.io". Click the link to access the application

5) Within the application, type a medical question and wait for about 10-15 seconds for a list of possible replies to be generated

### Things to note when running the notebook: 
- Ensure the runtime is in GPU 
- Clone the codes from repository to google collab: https://github.gatech.edu/rsim6/medicalqna
