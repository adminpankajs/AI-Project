from django.shortcuts import render
from django.http import HttpResponse
from website.static import testbot
from website import data
from website.static import images
import os
import cv2
import pytesseract
import speech_recognition as sr
import  numpy
import base64
from PIL import Image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# from website.static import test

# Create your views here.

def homepage(request):
    return render(request,"base.html")

def chatter(request):
    return render(request,"chatbot.html")

def chatbot(request, myptext):
    # print(myptext)
    myreply = testbot.chat(myptext)
    mytext = myreply
    # mytext = random.choice(myreply)
    return render(request,"chattext.html",{'text':mytext})

def facialrecognition(request):
    return render(request,"facialrecognition.html")

def processimg(request):
    global BASE_DIR

    face_cascade = cv2.CascadeClassifier('D:/PROFESSIONAL WORK/8. PROJECTS/5. Major/Artificial_Intelligence/website/data/haarcascade_frontalface_alt2.xml')
    # print(request.POST['mytext'])
    orgimg = request.POST['mytext'].split(",")[1]
    # print(orgimg)
    fcimg = os.path.join(BASE_DIR,'website/static/images/testimg.png')
    # print("\nDEBUG 1\n")
    with open(fcimg,"wb") as f:
        f.write(base64.b64decode(orgimg))
    imagepath = os.path.join(BASE_DIR,'website/static/images/testimg.png')
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # print("\nDEBUG 2\n")
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        img_item = os.path.join(BASE_DIR,'website/static/images/testing.png')
        # os.remove(img_item)
        # print("\nDEBUG 3\n")
        cv2.imwrite(img_item, roi_gray)
        

    return render(request,"processedimg.html")

def voicerecognize(request):
    global BASE_DIR
    audiopath = os.path.join(BASE_DIR,'website/static/audio/rawaudio.mp3')
    return render(request,"voicepage.html",{"audiopath" : audiopath})

def processvoice(request):
    audiofile = request.POST['myvoice'].split(",")[1]
    print("FIle Loaded")
    print(audiofile)
    print("File readed")

    saveaudio = os.path.join(BASE_DIR, 'website/static/audio/my.wav')
    # print("\nDEBUG 1\n")
    with open(saveaudio, "wb") as f:
        f.write(base64.b64decode(audiofile))

    text = "Not recognised"
    r = sr.Recognizer()
    print("Speak Anything :")
    with sr.AudioFile(saveaudio) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
        except:
            print("Sorry could not recognize what you said")
    return render(request,"processvoice.html", {'mytext' : text})

def textextract(request):
    return render(request,"mytextpage.html")

def processtext(request):
    global BASE_DIR
    myimg = request.FILES["myimage"].read()
    saveimg = os.path.join(BASE_DIR, 'website/static/images/mytestimg.png')
    # print("\nDEBUG 1\n")
    with open(saveimg, "wb") as f:
        f.write(myimg)
    # print("<<<<<<<<<<<<<<<<<<<<<<<")
    # print(type(myimg))
    # print("<<<<<<<<<<<<<<<<<<<<<<<")
    # myimg = base64.b64encode(request.FILES["myimage"].read())
    # myimg =myimg.split(",")[1]
    print(myimg)
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    myimg = os.path.join(BASE_DIR,'website/static/images/mytestimg.png')
    # myimg = cv2.imencode(myimg)
    img = cv2.imread(myimg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    # print(pytesseract.image_to_string(img))
    text = pytesseract.image_to_string(img)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    return render(request,"processtext.html",{ 'text': text })



