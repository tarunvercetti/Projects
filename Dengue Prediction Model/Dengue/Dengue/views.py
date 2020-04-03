from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.models import User , auth
import requests
import sys
from subprocess import run,PIPE
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ubidots import ApiClient
import json
import urllib.request
import urllib.parse


def button(request):
    return render(request,'home.html')


def external(request):
    inp=[]
    inp.append(request.POST.get('param1'))
    inp.append(request.POST.get('param2'))
    inp.append(request.POST.get('param3'))
    inp.append(request.POST.get('param4'))
    inp.append(request.POST.get('param5'))
    inp.append(request.POST.get('param6'))
    inp.append(request.POST.get('param7'))
    inp.append(request.POST.get('param8'))
    a=''
    b=''
    c=''
    d=''

    data_set()

    print(len(inp))
    output_data=api_client(inp,a,b,c,d)
    #output_data2 =sendSMS(apikey, numbers,sender, message)
    print (output_data)


    #print(out)
    return render(request,'home.html',{'data1':output_data})

def data_set():
    global regressor
    dataset = pd.read_csv('dengue.csv')
    dataset.head()
    columnlist = ['vomiting', 'eyepain', 'fatigue', 'fever', 'musclepain', 'skinrash', 'disease']
    for i in columnlist:
        labelencoder_X = LabelEncoder()
        dataset[i] = labelencoder_X.fit_transform(dataset[i])
    X = dataset.iloc[:, 7:-1].values
    X.shape
    Y = dataset.iloc[:, 15:].values
    Y.shape
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1.0)

    # clf.fit(x,y)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    #out = "hi %s prediction is %s" % (sys.argv[1],predictions)
    print(predictions)
    print()
    print("AUC:", roc_auc_score(Y_test, predictions))
    regressor = LogisticRegression()
    print (regressor)
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    Y_pred
    print("ACC:", roc_auc_score(Y_test, Y_pred))
    # plt.scatter(dataset['temp'], dataset['disease'])
    # plt.xlabel("Temperature")
    # plt.ylabel("diseases ")
    # #plt.show()
    # plt.plot(dataset['platelets'], dataset['disease'])
    # plt.xlabel("Platelets")
    # plt.ylabel("diseases ")
    # #plt.show()
    # plt.plot(dataset['vomiting'], dataset['platelets'])
    # plt.xlabel("Vomiting ")
    # plt.ylabel("Platelets ")
    #plt.show()
    return regressor

def api_client(s,apikey,numbers,message,username):
    api = ApiClient(token="A1E-1RiuZrHDsVkCXfXJhEYXro5JqT6gXllJfNUnRdrAOnj4zAvSFZZKCzrF")
    patient_condition='Not found'
    d=[]
    variable1 = api.get_variable("5d3daca7c03f9752cfa4d75b")
    variable2 = api.get_variable("5d550d04c03f9708d88d0db7")  # longitude
    variable3 = api.get_variable("5d550cf5c03f9708bbf50f1d")
    variable4 = api.get_variable("5d550ed0c03f9709ddb82928")
      #s = input("enter dengue inputs Symptoms : ")
   # t = input("enter dengue inputs Symptoms : ")
  #  u = input("enter dengue inputs Symptoms : ")
    #v = input("enter dengue inputs Symptoms : ")
    #w = input("enter dengue inputs Symptoms : ")
    #x = input("enter dengue inputs Symptoms : ")
   # y = input("enter dengue inputs Symptoms : ")
   # z = input("enter dengue inputs Symptoms : ")

    x = list(map(int, s))
    print(x)


    # prediction
    # 102 0 0 160000 0 1 0 0 --0(Normal)
    # 102 0 0 60000 0 1 0 0 --1(Critical)
    Y_pred = regressor.predict([x])
    Y_pred

    # apikey='q9wYxbxY1nQ-A02wlcARdNYp4R2ckoemcZX0rMQREQ'
    apikey='ce+/p8DI1mg-g6u3BB4FpqOeMGAn8q8HH7so6eYZDI'
    numbers=('3530894323227')
    sender='TXTLCL'
    message='Patient is critical due to dengue fever.'
    username = 'tarunhighv@icloud.com'

    def sendSMS(apikey, numbers, sender, message):
        data =  urllib.parse.urlencode({'apikey': apikey, 'numbers': numbers,
            'message' : message, 'sender': sender})
        data = data.encode('utf-8')
        request = urllib.request.Request("https://api.txtlocal.com/send/?")
        f = urllib.request.urlopen(request, data)
        fr = f.read()
        return(fr)

    latitude=''
    longitude=''
    city=''
    if Y_pred[0] == 1:
        a = 1
        variable1.save_value({'value': a})
        patient_condition='Patient is critical'

        #print("Patient is critical")
        send_url = "http://api.ipstack.com/31.193.221.194?access_key=76ea36786367e229e3a80033709d21e2"  # replace your API key
        geo_req = requests.get(send_url)
        geo_json = json.loads(geo_req.text)
        latitude = geo_json['latitude']
        longitude = geo_json['longitude']
        #sendSMS(apikey, numbers,sender, message)
        resp =  sendSMS(apikey, numbers,sender, message)
        print (resp)
        variable2.save_value({'value': longitude})
        variable3.save_value({'value': latitude})
        variable4.save_value({"value": 10, "context": {"lat": latitude, "lng": longitude}})
        city = geo_json['city']
        latitude =  ("Latitude : ", latitude)
        longitude = ("Longitude : ", longitude)
        print(city)
    else:
        b = 0
        #variable1.save_value({'value': b})
        #print("patient is normal ")
        patient_condition='Patient is normal'
    return patient_condition+'~'+str(city)+'~'+str(latitude)+'~'+str(longitude)
    #return latitude
