from flask import Flask, render_template, request
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk import data
from nltk.tokenize import word_tokenize
import string
import pickle

tfid = pickle.load(open('tfid.pkl', 'rb'))
model = pickle.load(open('svc.pkl', 'rb'))
app = Flask(__name__)
nltk.download('punkt') 
stopwords = StopWordRemoverFactory().get_stop_words()
stemmer = StemmerFactory().create_stemmer()

def lowercase(text):
    return text.lower()

def stemming(text):
    return stemmer.stem(text)

def data_cleansing(text):
    if('bagus' not in text):
        text = text.replace('bagu', 'bagus')
    text = text.replace(' gk ', ' tidak ')
    text = text.replace(' gak ', ' tidak ')
    text = text.replace(' g ', ' tidak ' )
    text = text.replace(' ga ', ' tidak ')
    text = text.replace(' krn ', ' karena ')
    text = text.replace(' blm ', ' belum ')
    text = text.replace(' sdh ', ' sudah ')
    text = text.replace(' klo ',' jika ')
    text = text.replace(' tdk ', ' tidak ')
    text = text.replace(' jgn ', ' jangan ')
    text = text.replace(' skrg ', ' sekarang ')
    text = text.replace(' dgn ', ' dengan ')
    text = text.replace(' emg ', ' memang ')
    text = text.replace(' dpt ', ' dapat ')
    text = text.replace(' ttp ', ' tetap ')
    text = text.replace(' utk ', ' untuk ')
    text = text.replace(' knp ', ' kenapa ')
    text = text.replace(' mkin ', ' semakin ')
    text = text.replace(' dkt ', ' dekat ')
    text = text.replace(' jg ', ' juga ')
    text = text.replace(' jga ', ' juga ')
    text = text.replace(' dr ', ' dari ')
    text = text.replace(' dri ', ' dari ')
    text = text.replace(' bbrp ', ' beberapa ')
    text = text.replace(' bikin ', ' menjadi ')
    text = text.replace("'a", '')
    text = text.replace(' kmr ', ' kamar ')
    text = text.replace(' hrs ', ' harus ')
    text = text.replace(' tmpt ', ' tempat ')
    text = text.replace(' tgl ', ' tanggal ')
    text = text.replace(' org ', ' orang ')
    text = text.replace(' sy ', ' saya ')
    text = text.replace(' sya ', ' saya ')
    text = text.replace(' lg ', ' lagi ')
    text = text.replace(' lgi ', ' lagi ')
    text = text.replace(' jd ', ' jadi ')
    text = text.replace(' jdi ', ' jadi ')
    text = text.replace(' adl ', ' adalah ')
    text = text.replace(' adlah ', ' adalah ')
    text = text.replace(' tp ', ' tapi ')
    text = text.replace(' tpi ', ' tapi ')
    text = text.replace(' mksd ', ' maksud ')
    text = text.replace(' kt ', ' kita ')
    text = text.replace(' kta ', ' kita ')
    text = text.replace(' yg ', ' yang ')
    text = text.replace(' bs ', ' bisa ')
    text = text.replace(' bsa ', ' bisa ')
    text = text.replace(' pesen ', ' pesan ')
    text = text.replace(' nyoba ', ' coba ')
    text = text.replace('tidak bagus', 'jelek')
    text = text.replace('tidak jelek', 'bagus')
    text = text.replace('tidak kecewa', 'suka')
    text = text.replace('tidak suka', 'kecewa')
    text = text.replace('muas', 'puas')
    text = text.replace('tidak puas', 'kecewa')
    word_lst = word_tokenize(text)
    word_lst = [word for word in word_lst if len(word) > 2 and word.isalnum()] # non alfa numerical removing
    word_lst = [word for word in word_lst if string.punctuation not in word] # string punctuition removing
    word_lst = [word for word in word_lst if word not in stopwords] # stopwords removing
    word_lst = [word for word in word_lst if word not in ['kok', 'nya', 'sih']]
    
    text = ' '.join(word_lst)
    return text

@app.route('/')
def main():
    return render_template('index.html', data="kosong")

@app.route('/', methods=['POST'])
def home():
    ulasan = request.form['ulasan'].lower()
    ulasan = data_cleansing(stemming(lowercase(ulasan)))
    pred = model.predict(tfid.transform([ulasan]))
    return render_template('index.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)