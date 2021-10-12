# import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import json
import pickle
import os
import random

def chat(userinput):
    print("WORKING 1 \\n")
    inp = userinput
    print(userinput)
    # a = r'intents.json'
        # file = open('intents.json')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'intents.json')) as file:
        data = json.load(file)
    print("WORKING 2 \\n")
    # try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data.pickle"), "rb") as f:
        words, labels, training, output = pickle.load(f)
    print("WORKING 3 \\n")
    # except:
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

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    print("WORKING 4 \\n")
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)
    print("WORKING 5 \\n")
    # with open("D:\\PROFESSIONAL WORK\\8. PROJECTS\\5. Major\\Artificial_Intelligence\\website\\static\\data.pickle", "wb") as f:
        # pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    print("WORKING 6 \\n")
    # try:
    model.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"model.tflearn"))
    # except:
    # model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
    # model.save("model.tflearn")
        # for x in range(0,6):
        #     print(x)
        #     print("HELLO\n")

    def bag_of_words(s, words):
        print("WORKING 8 \\n")
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
            
        return numpy.array(bag)

    print("WORKING 7 \\n")
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    print(results[0][results_index])

    if results[0][results_index] > 0.69:
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                print("TAG IS IN TAG")
                myresponse = tg['responses']
    else:
        myresponse = ['Not sure', 'Contact shivam', 'Confused']
    # return myresponse
    # print(inp)
    print(random.choice(myresponse))
    reply = random.choice(myresponse)
    return reply


# #DEBUG##
# chat("Your age")