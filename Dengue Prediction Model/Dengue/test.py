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
import requests
import json
import sys
import urllib.request
import urllib.parse



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
    plt.scatter(dataset['temp'], dataset['disease'])
    plt.xlabel("Temperature")
    plt.ylabel("diseases ")
    #plt.show()
    plt.plot(dataset['platelets'], dataset['disease'])
    plt.xlabel("Platelets")
    plt.ylabel("diseases ")
    #plt.show()
    plt.plot(dataset['vomiting'], dataset['platelets'])
    plt.xlabel("Vomiting ")
    plt.ylabel("Platelets ")
    #plt.show()
    return regressor

def api_client():
    api = ApiClient(token="A1E-Y0FHvT8NFn29e0Cv9O5vN9JlkVYvQj")

    variable1 = api.get_variable("5d3daca7c03f9752cfa4d75b")
    variable2 = api.get_variable("5d550d04c03f9708d88d0db7")  # longitude
    variable3 = api.get_variable("5d550cf5c03f9708bbf50f1d")
    variable4 = api.get_variable("5d550ed0c03f9709ddb82928")
    s = input("enter dengue inputs Symptoms : ")
    t = input("enter dengue inputs Symptoms : ")
    u = input("enter dengue inputs Symptoms : ")
    v = input("enter dengue inputs Symptoms : ")
    w = input("enter dengue inputs Symptoms : ")
    x = input("enter dengue inputs Symptoms : ")
    y = input("enter dengue inputs Symptoms : ")
    z = input("enter dengue inputs Symptoms : ")

    x = list(map(int, s.split()+t.split()+u.split()+v.split()+w.split()+x.split()+y.split()+z.split()))
    print(x)

#     apikey='q9wYxbxY1nQ-A02wlcARdNYp4R2ckoemcZX0rMQREQ'
#     numbers=('9962037023')
#     sender='TXTLCL'
#     message='Patient is critical due to dengue fever'
#     username = 'yadavaprasath@gmail.com'

# def sendSMS(apikey, numbers, sender, message):
#     data =  urllib.parse.urlencode({'username':username,'apikey': apikey, 'numbers': numbers,'message' : message, 'sender': sender})   
#     data = data.encode('utf-8')
#     request = urllib.request.Request("https://api.textlocal.in/send/?")
#     f = urllib.request.urlopen(request, data)
#     fr = f.read()
    #return(fr)

    # prediction
    # 102 0 0 160000 0 1 0 0 --0(Normal)
    # 102 0 0 60000 0 1 0 0 --1(Critical)
    Y_pred = regressor.predict([x])
    Y_pred
    if Y_pred[0] == 1:
        a = 1
        variable1.save_value({'value': a})
        print("Patient is critical")
        send_url = "http://api.ipstack.com/103.81.238.70?access_key=ad7c97b207925f101642719b8ccc9f78"  # replace your API key
        geo_req = requests.get(send_url)
        geo_json = json.loads(geo_req.text)
        # sendSMS(apikey, numbers,sender, message)
        latitude = geo_json['latitude']
        longitude = geo_json['longitude']
        variable2.save_value({'value': longitude})
        variable3.save_value({'value': latitude})
        variable4.save_value({"value": 10, "context": {"lat": latitude, "lng": longitude}})
        city = geo_json['city']
        print("Latitude : ", latitude)
        print("Longitude : ", longitude)
        print(city)
    else:
        b = 0
        variable1.save_value({'value': b})
        print("patient is normal ")

if __name__ == "__main__":

    data_set()
    api_client()

