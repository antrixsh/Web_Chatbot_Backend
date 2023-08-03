import pickle
from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app=Flask(__name__)
CORS(app,origins="*")

#Deserialise / Depickle
with open("faiss_store_openai.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# @app.route("/")
# def hello():
#     return render_template("index.html")

@app.route('/check',methods=['POST','GET'])
def predict_class():
    data = request.get_json()
    question = data.get("userInput")

    #features=[x for x in request.form.values()]
    #print(features[0])

    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.chains.question_answering import load_qa_chain
    from langchain import OpenAI
    llm = OpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

    if question:
        output = chain({"question": question}, return_only_outputs=True)
        print(output.get('answer'))
        #render_template("index.html",check=output.get('answer'))
        sudh = jsonify(output.get('answer'))
        print(sudh)
        return sudh
    else:
        return print("Error") #("index.html",check="Please Enter Data !")

if __name__ == "__main__":
    app.run(debug=True) #create a flask local server
#-----------------------------------------------------------------------------

# import pickle
# from flask import Flask, render_template, request, jsonify
# import os
# #from flask_cors import CORS
#
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# app=Flask(__name__)
# #app=Flask(__name__)
# #CORS(app,origins="*")
#
# #Deserialise / Depickle
# if __name__=='__main__':
#     with open("faiss_store_openai.pkl", "rb") as f:
#         VectorStore = pickle.load(f)
#
# @app.route("/")
# def hello():
#     return render_template("index.html")
#
# @app.route('/check',methods=['POST','GET'])
# def predict_class():
#     #data = request.get_json()
#     #question = data.get("userInput")
#
#     features=[x for x in request.form.values()]
#     print(features[0])
#
#     from langchain.chains import RetrievalQAWithSourcesChain
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain import OpenAI
#     llm = OpenAI(temperature=0)
#     chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
#
#     if features[0]:
#         output = chain({"question": features[0]}, return_only_outputs=True)
#         print(output.get('answer'))
#         return render_template("index.html",check=output.get('answer'))
#         #sudh = jsonify(output.get('answer'))
#         #print(sudh)
#         #return sudh
#     else:
#         return render_template("index.html",check="Please Enter Data !")
#
# if __name__ == "__main__":
#     app.run(debug=True) #create a flask local server


#-----------------------------------------------------------------------------

# import pickle
# from flask import Flask, render_template, request, jsonify
# import os
#
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
# app = Flask(__name__)
#
# # Deserialise / Depickle
# with open("faiss_store_openai.pkl", "rb") as f:
#     vectorstore = pickle.load(f)
#
# @app.route("/")
# def hello():
#     return render_template("index.html")
#
# @app.route('/check', methods=['POST', 'GET'])
# def predict_class():
#     print("IN FUNCTION")
#     #data = request.get_json()
#     #question = data.get("userInput")
#     features = [x for x in request.form.values()]
#     print(features[0])
#     from langchain.chains import RetrievalQA
#     from langchain import OpenAI
#     llm = OpenAI(temperature=0)
#     chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#
#     if features[0]:
#         output = chain({"query": features[0]}, return_only_outputs=True)
#         answer = output.get('answer')
#         print(answer)
#         return render_template("index.html", check=output.get('answer'))
#         #return jsonify({"answer": answer})
#     else:
#         return render_template("index.html", check="Please Enter Data !")
#         #return jsonify({"error": "Please Enter Data!"})
#
# if __name__ == "__main__":
#     app.run(debug=True)
