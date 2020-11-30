#!/usr/bin/env python3
from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template,jsonify
from covid_bot import generate_embeddings, generate_top_k_answers
import pandas as pd

# REPLACE THIS WITH FUNCTION FROM COVID_BOT.PY
# def run_covid_bot(input):
#     # results =jsonify({"User Questions":input}) 
#     # return "THIS IS THE INPUT: " + results.get_data(as_text=True)
#     return "THIS IS THE INPUT: " + input

def create_app(sentence_embeddings=None):
    
    # Load data  
    data = pd.read_csv("../data/questionDoctorQAs.csv")
    df = data.sample(n=1000)
    df.reset_index(drop=True, inplace=True)
    
    df = generate_embeddings(df)
    
    
    # questions = data['question'].to_list()
    # answers = data['answer']

    # data['qn_biobert-nli'] = [generate_embeddings(qn) for qn in questions]

    # # load embedding module:
    # module = load_embedding_module()   
    # # create qn and response embeddings
    # response_encodings = create_qna_embeddings(data,module)


    # create and configure the app
    app = Flask(__name__)
    run_with_ngrok(app)   #starts ngrok when the app is run
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        """The index page management."""

        # # GET
        if request.method == "GET":
          return render_template("chatbot.html")

        # # POST
        input = request.form.get("textarea")
        top_5_answers_index,top_5_answers = generate_top_k_answers(input,df,5)
        response  = top_5_answers

        # response = run_covid_bot(input,qn_embedding,answer_embedding) #run_covid_bot(input,data,module,response_encodings)
        
        
        return render_template("chatbot.html", response=response)
        # response = run_covid_bot(input,sentence_embeddings)
        # return render_template("chatbot.html", response=response.get_data(as_text=True))

    return app


if __name__ == '__main__':
    # from generate_sentence_embeddings import generate_embeddings
    # sentence_embeddings = generate_embeddings()
    # print(sentence_embeddings)
    # app = create_app(sentence_embeddings)
    app = create_app()
    app.run()



