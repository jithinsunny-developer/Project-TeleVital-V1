from flask import Flask,render_template,redirect,request
import pandas as pd
from dig_pres_utils import *
import speech_recognition as sr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/uploadAudio', methods=["GET","POST"])
def uploadAudio():
    if request.method =="POST":
        
        text=request.form.get("text_to_send")
        pat_dets= extractPatientDetails(text)
        patient_name = pat_dets[0]
        patient_age = pat_dets[1]
        patient_gender = pat_dets[2]
        if(patient_name==''):
            patient_name = find_name(text,patient_gender)
        prescriptions=extract_medicines(text)
        patient_advice = extract_advice(text)
        patient_symptoms = extract_symps(text)
        patient_disease = get_disease(text)
        #raise Exception(text,prescriptions,patient_advice)
        return render_template("uploadAudio.html",text=text,patient_name=patient_name,patient_gender=patient_gender,patient_age=patient_age,prescriptions=prescriptions,patient_advice=patient_advice,patient_symptoms=patient_symptoms,patient_disease=patient_disease)
    else:
        return render_template("uploadAudio.html")

@app.route('/getPDF', methods=["GET","POST"])
def getPDF():
    if request.method=="POST":
        
        hospital_name = "Televitals"
        doctor_name = "Dr Tele"
        doctor_address = "Cabin 37, Super Speciality Hospitals, Pratapganj,\nNew Delhi, Delhi, 110346\nPhone: 1234567890"
        
        patient_name = request.form.get("inp_name")
        patient_age = request.form.get("inp_age")
        patient_gender = request.form.get("inp_sex")
        symptoms = request.form.get("inp_symptoms")
        prescriptions = request.form.get("inp_medicine")
        prescriptions = extract_medicines(prescriptions)
        advice=request.form.get("inp_advice")
        diseases=request.form.get("inp_diagnosis")
        #symptoms = "High Fever\nCold\nCough\nBody Ache"
        #diseases = "Actue Bronchitis"
        #prescriptions="Azithromycin 500mg once a day for 3 days\nRobitussin 5 mlthrice a day for 5days"
        #advice="Have light Dinner\nDo not drink cold water"
        #raise Exception(prescriptions)
        download_name = ("".join(patient_name.split()))+".pdf"
        
        path=createPDF(hospital_name,doctor_name,doctor_address,
                      patient_name,patient_age,patient_gender,
                   symptoms,diseases,prescriptions,advice)
        path_txt = createtxt(hospital_name,doctor_name,doctor_address,
              patient_name,patient_age,patient_gender,
              symptoms,diseases,prescriptions,advice)

       
        #raise Exception(path_hindi)
        return render_template("getPDF.html",path=path,download_name=download_name,patient_name=patient_name,path_txt=path_txt)
    else:
        return render_template("getPDF.html")

@app.route('/sendEmail',methods=["GET","POST"])
def sendEmail():
    if request.method=="POST":
        receiver_id = request.form.get("email_box")
        patient_name = request.form.get("email_patient_name")
        path_pdf = request.form.get("path_holder")
        
        #raise Exception(hindi_path)
        sendEmailfun(receiver_id,patient_name,path_pdf)
        return render_template("email_sent.html")
    else:
        return render_template("email_sent.html")
    
if __name__ == "__main__":
    app.run(debug=True)
