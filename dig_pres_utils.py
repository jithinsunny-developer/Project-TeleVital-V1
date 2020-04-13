import os
from os import path
from fpdf import FPDF
from datetime import date,datetime
import pandas as pd
import email
import smtplib
from email import encoders
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import collections
import csv
from googletrans import Translator
import codecs
import re
from nltk.tag import pos_tag
d = 256



col_names_medicine=["drug_name"]
medicines = pd.read_csv('medicine.csv',header=None, names=col_names_medicine)

col_names=["symptom"]
symptoms = pd.read_csv('symptoms.csv',header=None, names=col_names)


def lower_text(text):
    text=text.lower()
    return text

medicines['drug_name'] = medicines['drug_name'].apply(lambda x: lower_text(x))
symptoms["symptom"] = symptoms["symptom"].apply(lambda x: lower_text(x))

def extract_advice(data_text):
    data = data_text.lower()
    all_text = data.split()
    L = len(all_text)

    meds=[]
    req_index=0
    for i in range(L):
        if all_text[i] in medicines["drug_name"].to_list():
            found=0
            for j in range(i+1,L):
                if all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    meds.append((" ".join(all_text[i:j])))
                    break
            if(found==0):
                #meds.append((" ".join(all_text[i:L])))
                meds.append(all_text[i])
                req_index=i
    advice_list=[]
    for i in range(req_index+1,L):
        advice_list.append(all_text[i])

    advice = " ".join(advice_list)
    return(advice)
    
def extract_medicines(data_text):
    data = data_text.lower()
    all_text = data.split()
    L = len(all_text)

    meds=[]
    for i in range(L):
        if all_text[i] in medicines["drug_name"].to_list():
            found=0
            for j in range(i+1,L):
                if all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    meds.append((" ".join(all_text[i:j])))
                    break
            if(found==0):
                #meds.append((" ".join(all_text[i:L])))
                meds.append(all_text[i])

    stop_words_presp=["needed","daily","day","days","weekly","week","weeks","basis","monthly","yearly","once","twice","thrice","times","lunch","breakfast","morning","dinner","night","afternoon","evening","meal","food","meals","foods","sleep","up","pm","p.m.","am","a.m.","noon","or"]

    formatted_meds=[]
    for i in range(len(meds)-1):
        med_i = meds[i].split()
        len_i = len(med_i)
        for j in range(len_i-1,-1,-1):
            done=0
            if med_i[j] in stop_words_presp:
                formatted_meds.append((" ".join(med_i[0:j+1])))
                done=1
                break
            else:
                if (med_i[j])[-1]=='g' and (med_i[j])[-2]=='m':
                    formatted_meds.append((" ".join(med_i[0:j+1])))
                    done=1
                    break
            if done==0:
                internal_meds=meds[i].split()
                for me in internal_meds:
                    if me in medicines["drug_name"].to_list():
                        formatted_meds.append(me)
                        break
                break
    if len(meds)!=0:            
        formatted_meds.append(meds[len(meds)-1])
    med_text_final = ""
    for med_ok in formatted_meds:
        med_text_final+=med_ok+" \n"
    return(med_text_final)

def extract_symps(data):
    data.lower()
    all_text = data.split()
    L=len(all_text)

    symps=[]
    meds_found=0
    for i in range(L):
        if all_text[i] in symptoms["symptom"].to_list():
            found=0
            for j in range(i+1,L):
                if " ".join(all_text[j:j+3]) in symptoms["symptom"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif " ".join(all_text[j:j+2]) in symptoms["symptom"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in symptoms["symptom"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    meds_found=1
                    break
                    
        elif " ".join(all_text[i:i+2]) in symptoms["symptom"].to_list():
            found=0
            for j in range(i+1,L):
                #print(" ".join(all_text[j:j+3]))
                #print(" ".join(all_text[j:j+2]))
                #print(all_text[j])
                if " ".join(all_text[j:j+3]) in symptoms["symptom"].to_list():
                    found=1
                    #i=j+3
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif " ".join(all_text[j:j+2]) in symptoms["symptom"].to_list():
                    found=1
                    #i=j+2
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in symptoms["symptom"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    meds_found=1
                    break
                    
        elif " ".join(all_text[i:i+3]) in symptoms["symptom"].to_list():
            found=0
            for j in range(i+1,L):
                #print(" ".join(all_text[j:j+3]))
                #print(" ".join(all_text[j:j+2]))
                #print(all_text[j])
                if " ".join(all_text[j:j+3]) in symptoms["symptom"].to_list():
                    found=1
                    #i=j+3
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif " ".join(all_text[j:j+2]) in symptoms["symptom"].to_list():
                    found=1
                    #i=j+2
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in symptoms["symptom"].to_list() or all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    break
                elif all_text[j] in medicines["drug_name"].to_list():
                    found=1
                    symps.append((" ".join(all_text[i:j])))
                    meds_found=1
                    break
                    
            if(found==0):
                symps.append(all_text[i])
        if(meds_found==1):
            break
    str_sy=""
    for sy in symps:
        str_sy+=sy+" \n"

    return(str_sy)



def get_disease(text):
    def listToString(s):   
        str1 = " " 
        return (str1.join(s)) 

    sym_search=[]
    sym_ind=[]
    dis_search=[] 
    sym_durn = []   
    sym_durn_ind = []  
      
    def searchdis(pat, txt, q): 
        M = len(pat) 
        N = len(txt) 
        i = 0
        j = 0
        p = 0     
        t = 0
        h = 1
      
        for i in range(M-1): 
            h = (h * d)% q 
      
        for i in range(M): 
            p = (d * p + ord(pat[i]))% q 
            t = (d * t + ord(txt[i]))% q 
      
        for i in range(N-M + 1):
            if p == t: 
                for j in range(M): 
                    if txt[i + j] != pat[j]: 
                        break
                j+= 1
                if j == M: 
                    dis_search.append(pat)

            if i < N-M:
                t = (d*(t-ord(txt[i])*h) + ord(txt[i + M]))% q
                if t < 0: 
                    t = t + q 

    def searchsym(pat, txt, q): 
        M = len(pat) 
        N = len(txt) 
        i = 0
        j = 0
        p = 0
        t = 0
        h = 1
      
        for i in range(M-1): 
            h = (h * d)% q 
      
        for i in range(M): 
            p = (d * p + ord(pat[i]))% q 
            t = (d * t + ord(txt[i]))% q 

        for i in range(N-M + 1): 
            if p == t: 
                for j in range(M): 
                    if txt[i + j] != pat[j]: 
                        break
      
                j+= 1
                if j == M:
                    sym_search.append(pat)
                    sym_ind.append(str(i))

            if i < N-M: 
                t = (d*(t-ord(txt[i])*h) + ord(txt[i + M]))% q 
                if t < 0: 
                    t = t + q 

    def searchsymdurn(pat, txt, q): 
        M = len(pat) 
        N = len(txt) 
        i = 0
        j = 0
        p = 0    
        t = 0   
        h = 1
      
        for i in range(M-1): 
            h = (h * d)% q 

        for i in range(M): 
            p = (d * p + ord(pat[i]))% q 
            t = (d * t + ord(txt[i]))% q 
      
        for i in range(N-M + 1): 
            if p == t: 
                for j in range(M): 
                    if txt[i + j] != pat[j]: 
                        break
      
                j+= 1
                if j == M: 
                    sym_durn.append(pat)
                    sym_durn_ind.append(i)
     
            if i < N-M: 
                t = (d*(t-ord(txt[i])*h) + ord(txt[i + M]))% q 
                if t < 0: 
                    t = t + q 

    file="sih_diseasess1.csv"
    list=[""]

    filey_n="Names.csv"
    listy=[""]

    filey_s="surnames.csv"
    listy_s=[""]

    filey_sy="Symptom.csv"
    listy_sy=[""]

    with open(file) as f:
        reader = csv.reader(f)
        for i in reader:
            x=i[0]
            list.append(x)

    with open(filey_sy) as f:
        reader = csv.reader(f)
        for i in reader:
            x=i[0]
            listy_sy.append(x)

    listy_sy_edited = []

    for i in range(len(listy_sy)):
            m=listy_sy[i]
            m=m.strip()
            listy_sy_edited.append(m)

    with open(filey_n) as fy:
        reader = csv.reader(fy)
        for i in reader:
            m=listToString(i)
            m=m.strip()
            listy.append(m)

    with open(filey_s) as fy:
        reader = csv.reader(fy)
        for i in reader:
            m=listToString(i)
            m=m.strip()
            listy_s.append(m)

    listLower = [item.lower() for item in list]
    listyLower= [item.lower() for item in listy]
    listyLower_s=[item.lower() for item in listy_s]
    listyLower_sy=[item.lower() for item in listy_sy_edited]

    g = text
    g = g.lower()
    g = g.replace("'", "")
    #g = t2d.convert(g)
    wordList = re.sub("[^\w]", " ",  g).split()
    wordListLower= [item.lower() for item in wordList]
    length=len(wordListLower)

    # NAME
    print("Name >> ")
    stri=""
    for i in range(length):
        if wordListLower[i] in listyLower:
            stri=wordListLower[i]
            if wordListLower[i+1] in listyLower_s:
                stri=stri+' '+wordListLower[i+1]
                break
    print(stri, "\n")
    name_index = g.index(stri)
    #print(name_index)

    # AGE
    print("Age >> ")
    age_factors=["month", "months", "day", "days"]
    str_age=""
    for i in range(length):
        if (wordListLower[i].isnumeric())==True:
            if(wordListLower[i+1]) in age_factors:
                str_age=wordListLower[i]+" "+wordListLower[i+1]
                age_index = g.index(str_age)
                break
            else:
                str_age=wordListLower[i]
                age_index = g.index(str_age)
                str_age += " "+"years"
                break
    print(str_age, "\n")

    # GENDER
    print("Gender >> ")
    gen_factors=["male", "female", "m", "f", "neutral", "genderqueer", "unidentified", "transgender", "unknown", "third"]
    gen=""
    for i in range(length):
        if wordListLower[i] in gen_factors:
            gen=wordList[i]
            break
    print(gen, "\n")
    gender_index = g.index(gen)

    # SYMPTOMS
    for pat in listyLower_sy:
        txt=g
        searchsym(pat, g, 101)

    for i in range(len(sym_ind)):
        sym_ind[i] = int(sym_ind[i])
    # PAUSE

    # DISEASE
    print("Disease >> ")
    for pat in listLower:
        txt=g
        searchdis(pat, g, 101)
    dis_search_u=[]
    for k in dis_search:
        k=k.strip()
        dis_search_u.append(k) 
    dis_search_update=[]
    for w in dis_search_u:
        if w not in sym_search:
            dis_search_update.append(w)
    print(dis_search_update)

    hello = g.index(dis_search_update[0])
    g = g[0:hello+len(dis_search_update[0])]

    wordList = re.sub("[^\w]", " ",  g).split()
    wordListLower= [item.lower() for item in wordList]
    length=len(wordListLower)
    unique_list = [] 
    def unique(list1): 
  
    # intilize a null list 
        
      
    # traverse for all elements 
        for x in list1: 
        # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 


    unique(dis_search_update)
    str_di=""
    for di in unique_list:
        str_di+=di+" \n"
    return(str_di)


today = date.today()
date = today.strftime("%d/%m/%Y")

def listToString(s):   
    str1 = " " 
    return (str1.join(s))


def extractPatientDetails(text_work):
    filey_n="Names.csv"
    listy=[""]

    filey_s="surnames.csv"
    listy_s=[""]

    with open(filey_n) as fy:
        reader = csv.reader(fy)
        for i in reader:
            m=listToString(i)
            m=m.strip()
            listy.append(m)

    with open(filey_s) as fy:
        reader = csv.reader(fy)
        for i in reader:
            m=listToString(i)
            m=m.strip()
            listy_s.append(m)

    listyLower= [item.lower() for item in listy]
    listyLower_s=[item.lower() for item in listy_s]

    
    g=text_work.lower()


    wordList = re.sub("[^\w]", " ",  g).split()
    wordListLower= [item.lower() for item in wordList]
    length=len(wordListLower)

    stri=""

    for i in range(length):
        if wordListLower[i] in listyLower:
            stri=wordListLower[i]
            if wordListLower[i+1] in listyLower_s:
                stri=stri+' '+wordListLower[i+1]
                break


    age_factors=["month", "months", "day", "days"]

    str_age=""

    for i in range(length):
        if (wordListLower[i].isnumeric())==True:
            if(wordListLower[i+1]) in age_factors:
                str_age=wordListLower[i]+" "+wordListLower[i+1]
                break
            else:
                str_age=wordListLower[i]+" "+"yrs"
                break
    
    gen_factors=["male", "female", "m", "f", "neutral", "genderqueer", "unidentified", "transgender", "unknown", "third"]

    gen=""

    for i in range(length):
        if wordListLower[i] in gen_factors:
            gen=wordList[i]
            break

    names_list = stri.split()
    cap_names=[]
    for i in range(len(names_list)):
        cap_names.append(names_list[i].capitalize())

    name=" ".join(cap_names)

    patient_details=[]
    patient_details.append(name)
    patient_details.append(str_age)
    patient_details.append(gen)

    return(patient_details)


def find_name(sentence,gen):
  #Method 1
  pos_gen=sentence.find(gen)
  sentence=sentence[:pos_gen]
  tagged_sent=pos_tag(sentence.split())
  proper_nouns=[w for w,pos in tagged_sent if pos=='NNP']
  name=''
  for w in proper_nouns:
    name=name+w+' '
  if name!='':
    return name

  #Method 2
  name_f=0
  sur_f=0
  name=''
  for w,pos in tagged_sent:
    if pos=='JJ' and name_f==0:
      name_f=1
      name=w
    elif pos=='NN' and name_f==1:
      name_f=0
      name=name+' '+w
      break
    else:
      name_f=0
      name=''
  if name!='':
    return name

  return ''

    
def sendEmailfun(receiver_id,patient_name,path_send):
    smtp_ssl_host = 'smtp.gmail.com'  # smtp.mail.yahoo.com
    smtp_ssl_port = 465
    username = 'televitalapp@gmail.com'
    password = 'whatTHEshit1'
    sender = 'televitalapp@gmail.com'
    targets = []
    targets.append(receiver_id)
    msg = MIMEMultipart()
    msg['Subject'] = 'Televital: Your Prescription '+date
    msg['From'] = sender
    msg['To'] = ', '.join(targets)

    mime_text = 'Dear '+patient_name+',\nGreetings from Televital!\nWe have attached your prescription in the mail. Kindly find it below.\n\nHave a nice day and take care!\nRegards,\n'
    txt = MIMEText(mime_text)
    msg.attach(txt)

    filepath = path_send
    with open(filepath, 'rb') as opened:
        openedfile = opened.read()
    attachedfile = MIMEApplication(openedfile, _subtype = "pdf", _encoder=encoders.encode_base64)

    file_name_=patient_name+".pdf"
    attachedfile.add_header('content-disposition', 'attachment', filename = file_name_)
    msg.attach(attachedfile)


    server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
    server.login(username, password)
    server.sendmail(sender, targets, msg.as_string())
    server.quit()


    
def createPDF(hospital_name,doctor_name,doctor_address,
              patient_name,patient_age,patient_gender,
              symptoms,diseases,prescriptions,advice):
    pdf = FPDF()
    pdf.add_page()
    #pdf.set_draw_color(70,225,250)
    pdf.set_line_width(2)
    pdf.rect(10,10,190,275)
    #pdf.set_text_color(70,225,250)
    pdf.set_font("Helvetica",style="I", size=40)
    pdf.cell(200, 20, txt=hospital_name, ln=1, align="C")
    pdf.set_font("Arial", size=20)
    pdf.cell(200, 10, txt=doctor_name, ln=1, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200,5, txt=doctor_address, align="C")
    pdf.set_line_width(0.5)
    pdf.line(60,60,160,60)
    pdf.ln(10)
    pdf.set_left_margin(20)
    pdf.set_font("Arial", size=15)
    pdf.cell(300,5.5,txt="Patient's Name:  "+patient_name,ln=1)
    pdf.cell(300,5.5,txt="Age:  "+patient_age,ln=1)
    pdf.cell(300,5.5,txt="Sex:  "+patient_gender,ln=1)
    pdf.cell(300,5.5,txt="Date: "+date,ln=1)
    pdf.ln(7)
    pdf.set_font("Arial", size=20)
    pdf.cell(300,7,txt="Symptoms",ln=1)
    pdf.ln(3)
    pdf.set_left_margin(40)
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(200,6,txt=symptoms)

    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.cell(300,1,txt="",ln=1)
    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.set_font("Arial", size=20)
    pdf.cell(300,1,txt="Diagnosis",ln=1)
    pdf.ln(5)
    pdf.set_left_margin(40)
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(200,6,txt=diseases)

    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.cell(300,1,txt="",ln=1)
    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.set_font("Arial", size=20)
    pdf.cell(300,1,txt="Prescription",ln=1)
    pdf.ln(5)
    pdf.set_left_margin(40)
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(200,6,txt=prescriptions)

    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.cell(300,1,txt="",ln=1)
    pdf.ln(3)
    pdf.set_left_margin(20)
    pdf.set_font("Arial", size=20)
    pdf.cell(300,1,txt="Advice",ln=1)
    pdf.ln(5)
    pdf.set_left_margin(40)
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(200,6,txt=advice)
    temp_list = doctor_name.split()
    folder_name = "".join(temp_list)
    
    if not os.path.exists("static/"+folder_name):
        os.makedirs("static/"+folder_name)

    temp_list = patient_name.split()
    file_name = "".join(temp_list)
    month=datetime.now().month
    month=str(month)
    year=datetime.now().year
    year=str(year)
    day=datetime.now().day
    day=str(day)
    hour=datetime.now().hour
    hour=str(hour)
    minutes=datetime.now().minute
    minutes=str(minutes)
    seconds=datetime.now().second
    seconds=str(seconds)

    time_stamp=year+"_"+month+"_"+day+"_"+hour+"_"+minutes+"_"+seconds+"_file"
    file_name+=time_stamp
    
    path="static/"+folder_name+"/"+file_name+".pdf"
    pdf.output(path)
    return path

def createtxt(hospital_name,doctor_name,doctor_address,
              patient_name,patient_age,patient_gender,
              symptoms,diseases,prescriptions,advice):

    temp_list = doctor_name.split()
    folder_name = "".join(temp_list)
    
    if not os.path.exists("static/"+folder_name):
        os.makedirs("static/"+folder_name)

    temp_list = patient_name.split()
    
    file_name = "".join(temp_list)
    
    month=datetime.now().month
    month=str(month)
    year=datetime.now().year
    year=str(year)
    day=datetime.now().day
    day=str(day)
    hour=datetime.now().hour
    hour=str(hour)
    minutes=datetime.now().minute
    minutes=str(minutes)
    seconds=datetime.now().second
    seconds=str(seconds)

    time_stamp=year+"_"+month+"_"+day+"_"+hour+"_"+minutes+"_"+seconds+"_file"
    file_name+=time_stamp
    path_txt="static/"+folder_name+"/"+file_name+"text.txt"
    f=open(path_txt,"w+")
    f.write("HOSPITAL NAME: "+hospital_name+"\n")
    f.write("DOCTOR NAME: "+doctor_name+"\n")
    f.write("DOCTOR ADDRESS: "+doctor_address+"\n")
    f.write("\n\n")
    f.write("PATIENT NAME: "+patient_name+"\n")
    f.write("PATIENT AGE: "+patient_age+"\n")
    f.write("PATIENT SEX: "+patient_gender+"\n")
    f.write("\n\n")
    f.write("SYMPTOMS\n")
    f.write(symptoms+"\n\n")
    f.write("DIAGNOSIS\n")
    f.write(diseases+"\n\n")
    f.write("PRESCRIPTION\n")
    f.write(prescriptions+"\n\n")
    f.write("ADVICE\n")
    f.write(advice+"\n\n")
    f.close()
    return path_txt
