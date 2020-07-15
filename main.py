import nltk
from nltk.stem.lancaster import LancasterStemmer
import nltk.stem.lancaster

stemmer=LancasterStemmer()
import numpy as np
import tensorflow
import tflearn

import pickle
import random
import json




with open ("intents.json") as file:
    data =json.load(file)

#print (data)

try:

    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #print("\n \n\nwords (nltk.word_tokenize())(all words in each each hardcoded user input sentances)  :  " + str(words))
    #print(" \ndocs_x (array containing hard coded user input words ) :  " + str(docs_x))
    #print("\nlength of doc_x : " + str((len(docs_x))))
    #print("\ndocs_y  :  " + str(docs_y))
    #print("\nlabels (basically tag) :  " + str(labels))

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    #print("\n\nwords  lower stemmed :  " + str(words))

    words = sorted(list(set(words)))
    #print("words  lower stemmed and sorted :  " + str(words))
    #print(len(words))

    labels = sorted(labels)
    #print("labels sorted" + str(labels) + "\n \n \n")

    words=[stemmer.stem(tokenized_word.lower()) for tokenized_word in words]
    #print("new words"+ str(words))

    # one hot encoding
    # words= ['hello','a','buddy']
    # if the word is hello buddy
    # encoded = [1,0,1]           (whether the word is in the library /not .. if yes how many?

    training = []
    output = []

    # 1hotencoding for classes (greatings,goodbye ect) : [0 1] for goodbye
    out_empty = [0 for _ in range(len(labels))]
    #print("out empty (ie 0 for _ in words length) " + str(out_empty))
    #print(len(out_empty))

    for x, doc in enumerate(docs_x):
        #print("\n\n\nx,doc = " + str(x) + "," + str(doc))

        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]
        #rint("stemmed word no " + str(x) + " : " + str(wrds))

        for w in words:
            if w in wrds:

                bag.append(1)
            else:
                bag.append(0)
            #print("\nwrds " + str(w) + "   : " + str(wrds))
        #print("bag=" + str(bag))
        output_row = out_empty[:]

        #print("docs_y[x] : " + str(docs_y[x]))
        #print("labels.index(docs_y[x]) : " + str(labels.index(docs_y[x])))
        output_row[labels.index(docs_y[x])] = 1
        #print("output row" + str(output_row))

        training.append(bag)
        #print("training=" + str(training))
        output.append(output_row)
        #print("output" + str(output))

    training = np.array(training)
    output = np.array(output)
    #print("\ntrain " + str(x) + "   : " + str(training))
    #print("\noutput " + str(x) + "   : " + str(output))

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,32,activation="relu",regularizer='L2')
net = tflearn.fully_connected(net,180,activation="relu")
net = tflearn.fully_connected(net,1800,activation="relu")
net = tflearn.fully_connected(net,1800,activation="relu")




net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net =tflearn.regression(net)
model=tflearn.DNN(net)
model = tflearn.DNN(net)


try:
    model.load("model.tflearn")


except:
    model.fit(training, output, n_epoch=625, batch_size=0, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=(1)
    return np.array(bag)




def predict(input):
    #print("chat")
    command="auto"
    inp = input
    #inp=tcglisten("en")
    # inp=input()

    #print("user : " + str(inp))
    if inp.lower == "quit":
        b=1
    # print([bag_of_words(inp,words)])
    results = model.predict([bag_of_words(inp, words)])[0]
    # print(results)
    results_index = np.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.271:

        for tg in data["intents"]:
            if tg['tag'] == tag:
                i = data["intents"].index(tg)
                responses = tg['responses']
                ran = random.choice(responses)
                j = responses.index(ran)
                name = "response_" + str(i) + "_" + str(j)
                if command == "auto":
                    file = 'audiobase\{}.mp3'.format(name)
                if command == "manual":
                    file = 'audiobase\{}.wav'.format(name)

                from playsound import playsound
                #playsound(file)
                #print("nstcg : " + str(ran))
        return ran
    else:
        ran="i didnt get it"
        return ran






from flask import Flask,url_for,render_template,request,redirect

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    #return f"<h1>Welcome iam you personalised ai chatbot: NSTCG mark 1, you can ask me anything by putting a slash (/) and the question in the browser</h1>"

@app.route("/<usr>")
def user(usr):
    #res=predict(usr)
    return f"<h1>{usr}</h1>"

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        response=predict(user)
        #print(user)
        return render_template("login.html",res=response)
    else:
        return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)

#for importing image :  <p><img src="https://drive.google.com/uc?export=view&id=1Q1dJLHK5L_zpSgf0yWpXUmp4Mcyrlfho"></p>