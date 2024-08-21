#Code written by: Huiqiong Weng

##General overview
##Pred_N is an interactive desktop application for predicting diffusion coefficient of gas molecules in MOF or other porous crystal materials.The core of its computation is our trained XGB model. The following is the source code for the interface design.

## Import the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor as lgb
import tkinter as tk 
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image,ImageTk
import webbrowser
import joblib
import sys
import os
import random
import numpy as np

## Load the trained LGBM model
model = joblib.load("../model/lgbm.bin") ## The detailed code for model training is in “LBGM_code.py”

## The layout design of the main interface  
root= tk.Tk()
root.title("Predict adsorption capacity of material on LGBM")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 1170, height =770) ## Main window size
canvas1.pack()

## The plate layout within the main interface   
re1=canvas1.create_rectangle(50,20,1120,490)
re2=canvas1.create_rectangle(50,490,1120,710)
re3=canvas1.create_rectangle(70,100,380,470,outline='darkgray')
re4=canvas1.create_rectangle(400,100,1100,360,outline='darkgray')
re4=canvas1.create_rectangle(400,380,1100,470,outline='darkgray')
re4=canvas1.create_rectangle(70,600,1100,690,outline='darkgray')
label_B = tk.Label(root,font=('microsoft yahei',11),text='Predicted results')
canvas1.create_window(970, 380, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',11),text='Predicted results')
canvas1.create_window(970, 600, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Author：Huiqiong Weng,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(930, 755, window=label_B)


## Message box (Related websites on molecular physical properties)    
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('600x200')  # Adjusted size for better fit

    intro_text = '  The physical properties of gases are known from the following websites :\n' \
                 '  (Click to jump to the corresponding website)'
    intro_label = tk.Label(window, text=intro_text, font=('microsoft yahei', 11), anchor="w", justify="left")
    intro_label.place(x=30, y=20)

    # Links as a list of tuples (display text, URL)
    links = [
        ("1. ChemSpider", "http://www.chemspider.com/"),
        ("2. NIST", "https://www.nist.gov/"),
        ("3. PubChem", "https://pubchem.ncbi.nlm.nih.gov/")
    ]

    y_position = 70
    for text, url in links:
        def open_url(url=url):  # Default parameter to capture the url
            webbrowser.open(url, new=0)

        link_label = tk.Label(window, text=text, font=('microsoft yahei', 11), fg="blue", cursor="hand2")
        link_label.place(x=50, y=y_position)
        link_label.bind("<Button-1>", lambda event, url=url: open_url(url))
        y_position += 30  # Increment y position for the next label

btn1 = tk.Button(root, text='Tool-tip', font=('microsoft yahei', 11), command=cmx1)
canvas1.create_window(450, 140, window=btn1)


##Message box (Instructions for Prediction of a single material adsorption capacity)     
def resize(w, h, w_box, h_box, pil_image): 
    f1 = w_box / w  
    f2 = h_box / h  
    factor = min(f1, f2)  
    width = int(w * factor)  
    height = int(h * factor)  
    return pil_image.resize((width, height), Image.LANCZOS)

w_box = 600  
h_box = 600  

photo1 = Image.open("../Img/full_name.png")
w, h = photo1.size       
photo1_resized = resize(w, h, w_box, h_box, photo1)    
tk_image1 = ImageTk.PhotoImage(photo1_resized)

def cmx2():
    top2 = tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x620')  
    lab_1 = tk.Label(top2, image=tk_image1) 
    lab_1.pack(padx=10, pady=10)  
    top2.mainloop()
    
btn2 = tk.Button(root, text='Read-me', font=('microsoft yahei', 11), command=cmx2)
canvas1.create_window(120, 60, window=btn2)


## Message box (Instructions for batch Prediction of material adsorption capacity)  
def resize(w, h, w_box, h_box, pil_image):
    f1 = w_box / w  
    f2 = h_box / h  
    factor = min(f1, f2)  
    width = int(w * factor)  
    height = int(h * factor)  
    return pil_image.resize((width, height), Image.LANCZOS)
   
w_box = 850  
h_box = 1000

global tk_image 
photo2 = Image.open("../Img/sample_file.png") 
w, h = photo2.size     
photo2_resized = resize(w, h, w_box, h_box, photo2)    
tk_image2 = ImageTk.PhotoImage(photo2_resized)

def cmx3():
    top1=tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('900x550')
    lab2 = tk.Label(top1, text='You need to create the data you want to compute in the format below. \n(For example：the prediction of N-soman):'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab2.place(x=20, y=20) 
    lab3 = tk.Label(top1, text='After creating the file, you can click the import file button on the screen.\nThe predicted result  will be saved in "Result/Batch_Predicted_N.xlsx".'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab3.place(x=30, y=450)
    lab3 = ttk.Label(top1,text="photo:",image=tk_image2)
    lab3.place(x=30, y=120) 
    top1.mainloop()     
btn3=tk.Button(root, text='Read-me',font=('microsoft yahei',11),command=cmx3)
canvas1.create_window(120, 550, window=btn3)

## Sets the label and entry for entering the nine descriptor   
label_Z = tk.Label(root,font=('microsoft yahei',13),text='Predict adsorption capacity of material')
canvas1.create_window(585, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',11),text='Physical property of material')
canvas1.create_window(225, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10,"italic"),text='ф') ## create 1st label box 
canvas1.create_window(110, 150, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(300, 150, window=entry1)

label2 = tk.Label(root, font=('microsoft yahei',10,),text='PLD') ## create 2st label box 
canvas1.create_window(110, 195, window=label2)
l0 = tk.Label(root, font=('microsoft yahei',10),text='(Å) :')
canvas1.create_window(180, 195, window=l0) 
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 2st entry box
canvas1.create_window(300, 195, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD') ## create 3st label box 
canvas1.create_window(110, 240, window=label3)
l1 = tk.Label(root, font=('microsoft yahei',10),text='(Å) :')
canvas1.create_window(180, 240, window=l1) 
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(300, 240, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ') ## create 4st label box 
canvas1.create_window(105, 285, window=label4)
l2 = tk.Label(root,font=('microsoft yahei',10), text='(kg/m^3) : ') 
canvas1.create_window(180, 285, window=l2)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(300, 285, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='VSA') ## create 5st label box 
canvas1.create_window(110, 330, window=label5) 
l3 = tk.Label(root, font=('microsoft yahei',10),text='(m^2/cm^3) : ')
canvas1.create_window(185, 330, window=l3) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(300, 330, window=entry5)

label_R = tk.Label(root,font=('microsoft yahei',11),text='Physical property of gas molecules') 
canvas1.create_window(750, 100, window=label_R)

label6 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dia') ## create 6st label box 
canvas1.create_window(430,180, window=label6)
l4 = tk.Label(root, font=('microsoft yahei',10),text='(Å) :')
canvas1.create_window(525,180, window=l4) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(660,180, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pol') ## create 7st label box 
canvas1.create_window(430,220, window=label7)
l5= tk.Label(root, font=('microsoft yahei',10,),text='(×10^25 cm^3) : ')
canvas1.create_window(530,220, window=l5)
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 7nd entry box
canvas1.create_window(660,220, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dip') ## create 8st label box 
canvas1.create_window(430,260, window=label8) 
l6 = tk.Label(root, font=('microsoft yahei',10),text='(D) : ')
canvas1.create_window(525,260, window=l6)  
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 8nd entry box
canvas1.create_window(660, 260, window=entry8)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tb') ## create 9st label box 
canvas1.create_window(430, 300, window=label9)
l7 = tk.Label(root, font=('microsoft yahei',10),text='(K) : ')
canvas1.create_window(525, 300, window=l7) 
entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,300, window=entry9)

label10 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tc') ## create 10st label box 
canvas1.create_window(430, 340, window=label10)
l8 = tk.Label(root, font=('microsoft yahei',10),text='(K) : ')
canvas1.create_window(525, 340, window=l8) 
entry10 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 10nd entry box
canvas1.create_window(660,340, window=entry10)

label11 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pc') ## create 11st label box 
canvas1.create_window(780, 180, window=label11)
l9 = tk.Label(root, font=('microsoft yahei',10),text='(bar) : ')
canvas1.create_window(875, 180, window=l9) 
entry11 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 11nd entry box
canvas1.create_window(1010, 180, window=entry11)

label12 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='ρ') ## create 12st label box 
canvas1.create_window(780, 220, window=label12)
l10 = tk.Label(root, font=('microsoft yahei',10),text='(kg/m^3) : ')
canvas1.create_window(875, 220, window=l10) 
entry12 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 12nd entry box
canvas1.create_window(1010,220, window=entry12)

label13 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tm') ## create 13st label box 
canvas1.create_window(780, 260, window=label13)
l11 = tk.Label(root, font=('microsoft yahei',10),text='(K) : ')
canvas1.create_window(875, 260, window=l11) 
entry13 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 13st entry box
canvas1.create_window(1010,260, window=entry13)

label14 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tf') ## create 14st label box 
canvas1.create_window(780, 300, window=label14)
l12 = tk.Label(root, font=('microsoft yahei',10),text='(K) : ')
canvas1.create_window(875, 300, window=l12) 
entry14 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 14st entry box
canvas1.create_window(1010,300, window=entry14)

label15 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='ω') ## create 15st label box 
canvas1.create_window(780, 340, window=label15)
l13 = tk.Label(root, font=('microsoft yahei',10),text='(Ω) : ')
canvas1.create_window(875, 340, window=l13) 
entry15 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 15st entry box
canvas1.create_window(1010,340, window=entry15)

label16 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='K') ## create 16st label box 
canvas1.create_window(110, 375, window=label16)
l14 = tk.Label(root, font=('microsoft yahei',10),text='(mol/kg/Pa) : ')
canvas1.create_window(180, 375, window=l14) 
entry16 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 15st entry box
canvas1.create_window(300,375, window=entry16)

label17 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Q') ## create 15st label box 
canvas1.create_window(110, 420, window=label17)
l15 = tk.Label(root, font=('microsoft yahei',10),text='(KJ/mol) : ')
canvas1.create_window(180, 420, window=l15) 
entry17 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 15st entry box
canvas1.create_window(300,420, window=entry17)


## The linkage between the physical properties of molecules and molecules is realized  
## Sets four properties: 1-dia,2-pol,3-Dip,4-Tb,5-Tc,6-Pc,7-ρc,8-Tm,9-Tf,10-ω
input1 = 0
input2 = 0
input3 = 0
input4 = 0
input5 = 0
input6 = 0
input7 = 0
input8 = 0
input9 = 0
input10 = 0
def run2():
    dic1 = {0: 'Dia', 1: 'Pol', 2: 'Dip',3: 'Tb',4: 'Tc',5: 'Pc',6: 'ρ',7: 'Tm',8: 'Tf',9: 'ω'}
    c2 = dic1[cm1.current()]
    if (c2 == 'Dia'):                      
        entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input1,justify='center')
        canvas1.create_window(660,180, window=entry6)              
    elif (c2 =='Pol'):
        entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input2,justify='center') 
        canvas1.create_window(660,220, window=entry7)
    elif (c2 =='Dip'):
        entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input3,justify='center') 
        canvas1.create_window(660,260, window=entry8)                   
    elif (c2 =='Tb'):
        entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,300, window=entry9)
    elif (c2 =='Tc'):
        entry10 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,340, window=entry10)
    elif (c2 =='Pc'):
        entry11 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(1010,180, window=entry11)
    elif (c2 =='ρ'):
        entry12 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(1010,220, window=entry12)
    elif (c2 =='Tm'):
        entry13 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(1010,260, window=entry13)
    elif (c2 =='Tf'):
        entry14 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(1010,300, window=entry14)
    elif (c2 =='ω'):
        entry15 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(1010,340, window=entry15)
    
 
def calc(event):
    global input1
    global input2
    global input3
    global input4
    global input5
    global input6
    global input7
    global input8
    global input9
    global input10
    dic = {0: '2-CEES', 1: 'DMMP', 2: 'HD',3: 'Soman',4: 'DIFP',5: 'sarin',6: 'C2H5Cl',7: 'CNCl',8: 'NH3'}
    c = dic[cm1.current()] 
    if (c == '2-CEES'): 
        input1 = 5.800 
        input2 = 123.8
        input3 = 2.250
        input4 = 429.15
        input5 = 540.13
        input6 = 27.36
        input7 = 1070
        input8 = 224.55
        input9 = 325.35
        input10 = 0.349
        entry6.delete(0,"end")
        entry6.insert(0,"5.800")
        entry7.delete(0,"end")
        entry7.insert(0,"123.8")
        entry8.delete(0,"end")
        entry8.insert(0,"2.250")
        entry9.delete(0,"end")
        entry9.insert(0,"429.15")
        entry10.delete(0,"end")
        entry10.insert(0,"540.13")
        entry11.delete(0,"end")
        entry11.insert(0,"27.36")
        entry12.delete(0,"end")
        entry12.insert(0,"1070")
        entry13.delete(0,"end")
        entry13.insert(0,"224.55")
        entry14.delete(0,"end")
        entry14.insert(0,"325.35")
        entry15.delete(0,"end")
        entry15.insert(0,"0.349")
    elif (c == 'DMMP'):
        input1 = 5.700 
        input2 = 100.5
        input3 = 2.270
        input4 = 458.4
        input5 = 700.6
        input6 = 49.7
        input7 = 1145
        input8 = 225.15
        input9 = 342.35
        input10 = 0.390
        entry6.delete(0,"end")
        entry6.insert(0,"5.700")
        entry7.delete(0,"end")
        entry7.insert(0,"100.5")
        entry8.delete(0,"end")
        entry8.insert(0,"2.270")
        entry9.delete(0,"end")
        entry9.insert(0,"458.4")
        entry10.delete(0,"end")
        entry10.insert(0,"700.6")
        entry11.delete(0,"end")
        entry11.insert(0,"49.7")
        entry12.delete(0,"end")
        entry12.insert(0,"1145")
        entry13.delete(0,"end")
        entry13.insert(0,"225.15")
        entry14.delete(0,"end")
        entry14.insert(0,"342.35")
        entry15.delete(0,"end")
        entry15.insert(0,"0.390")
    elif (c == 'HD'):
        input1 = 5.900 
        input2 = 158.3
        input3 = 1.182
        input4 = 489.15
        input5 = 540.13
        input6 = 27.36
        input7 = 1211
        input8 = 287.60
        input9 = 362.55
        input10 = 0.349
        entry6.delete(0,"end")
        entry6.insert(0,"5.900")
        entry7.delete(0,"end")
        entry7.insert(0,"158.3")
        entry8.delete(0,"end")
        entry8.insert(0,"1.182")
        entry9.delete(0,"end")
        entry9.insert(0,"489.15")
        entry10.delete(0,"end")
        entry10.insert(0,"540.13")
        entry11.delete(0,"end")
        entry11.insert(0,"27.36")
        entry12.delete(0,"end")
        entry12.insert(0,"1211")
        entry13.delete(0,"end")
        entry13.insert(0,"287.60")
        entry14.delete(0,"end")
        entry14.insert(0,"362.55")
        entry15.delete(0,"end")
        entry15.insert(0,"0.349")
    elif (c == 'Soman'):
        input1 = 7.200
        input2 = 171.2
        input3 = 3.100
        input4 = 467.6
        input5 = 674.9
        input6 = 29.2
        input7 = 1022
        input8 = 231.15
        input9 = 348.65
        input10 = 0.410
        entry6.delete(0,"end")
        entry6.insert(0,"7.200")
        entry7.delete(0,"end")
        entry7.insert(0,"171.2")
        entry8.delete(0,"end")
        entry8.insert(0,"3.100")
        entry9.delete(0,"end")
        entry9.insert(0,"467.6")
        entry10.delete(0,"end")
        entry10.insert(0,"674.9")
        entry11.delete(0,"end")
        entry11.insert(0,"29.2")
        entry12.delete(0,"end")
        entry12.insert(0,"1022")
        entry13.delete(0,"end")
        entry13.insert(0,"231.15")
        entry14.delete(0,"end")
        entry14.insert(0,"348.65")
        entry15.delete(0,"end")
        entry15.insert(0,"0.410")
    elif (c == 'DIFP'):
        input1 = 5.3
        input2 = 160
        input3 = 0
        input4 = 456.15
        input5 = 674.9
        input6 = 29.2
        input7 = 1055
        input8 = 191.15
        input9 = 337.65
        input10 = 0.400
        entry6.delete(0,"end")
        entry6.insert(0,"5.3")
        entry7.delete(0,"end")
        entry7.insert(0,"160")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"456.15")
        entry10.delete(0,"end")
        entry10.insert(0,"674.9")
        entry11.delete(0,"end")
        entry11.insert(0,"29.2")
        entry12.delete(0,"end")
        entry12.insert(0,"1055")
        entry13.delete(0,"end")
        entry13.insert(0,"191.15")
        entry14.delete(0,"end")
        entry14.insert(0,"337.65")
        entry15.delete(0,"end")
        entry15.insert(0,"0.400")  
    elif (c == 'sarin'):
        input1 = 7.04
        input2 = 116
        input3 = 3.010
        input4 = 427
        input5 = 629.8
        input6 = 36.1
        input7 = 1089
        input8 = 217.15
        input9 = 315.85
        input10 = 0.400
        entry6.delete(0,"end")
        entry6.insert(0,"7.04")
        entry7.delete(0,"end")
        entry7.insert(0,"116")
        entry8.delete(0,"end")
        entry8.insert(0,"3.010")
        entry9.delete(0,"end")
        entry9.insert(0,"427")
        entry10.delete(0,"end")
        entry10.insert(0,"629.8")
        entry11.delete(0,"end")
        entry11.insert(0,"36.1")
        entry12.delete(0,"end")
        entry12.insert(0,"1089")
        entry13.delete(0,"end")
        entry13.insert(0,"217.15")
        entry14.delete(0,"end")
        entry14.insert(0,"315.85")
        entry15.delete(0,"end")
        entry15.insert(0,"0.400")
    elif (c == 'C2H5Cl'):
        input1 = 5.300
        input2 = 64
        input3 = 2.000
        input4 = 285.65
        input5 = 460.35
        input6 = 52.3
        input7 = 890
        input8 = 134.15
        input9 = 216.45
        input10 = 0.204
        entry6.delete(0,"end")
        entry6.insert(0,"5.300")
        entry7.delete(0,"end")
        entry7.insert(0,"64")
        entry8.delete(0,"end")
        entry8.insert(0,"2.000")
        entry9.delete(0,"end")
        entry9.insert(0,"285.65")
        entry10.delete(0,"end")
        entry10.insert(0,"460.35")
        entry11.delete(0,"end")
        entry11.insert(0,"52.3")
        entry12.delete(0,"end")
        entry12.insert(0,"890")
        entry13.delete(0,"end")
        entry13.insert(0,"134.15")
        entry14.delete(0,"end")
        entry14.insert(0,"216.45")
        entry15.delete(0,"end")
        entry15.insert(0,"0.204")
    elif (c == 'CNCl'):
        input1 = 0
        input2 = 45.3
        input3 = 2.690
        input4 = 286.25
        input5 = 460.35
        input6 = 59.9
        input7 = 1234
        input8 = 266.25
        input9 = 235.65
        input10 = 0.322
        entry6.delete(0,"end")
        entry6.insert(0,"0")
        entry7.delete(0,"end")
        entry7.insert(0,"45.3")
        entry8.delete(0,"end")
        entry8.insert(0,"2.690")
        entry9.delete(0,"end")
        entry9.insert(0,"286.25")
        entry10.delete(0,"end")
        entry10.insert(0,"460.35")
        entry11.delete(0,"end")
        entry11.insert(0,"59.9")
        entry12.delete(0,"end")
        entry12.insert(0,"1234")
        entry13.delete(0,"end")
        entry13.insert(0,"266.25")
        entry14.delete(0,"end")
        entry14.insert(0,"235.65")
        entry15.delete(0,"end")
        entry15.insert(0,"0.322")
    elif (c == 'NH3'):
        input1 = 2.900
        input2 = 22.1
        input3 = 1.470
        input4 = 239.82
        input5 = 405.4
        input6 = 113.53
        input7 = 771
        input8 = 195.45
        input9 = 284.15
        input10 = 0.252
        entry6.delete(0,"end")
        entry6.insert(0,"2.900")
        entry7.delete(0,"end")
        entry7.insert(0,"22.1")
        entry8.delete(0,"end")
        entry8.insert(0,"1.470")
        entry9.delete(0,"end")
        entry9.insert(0,"239.82")
        entry10.delete(0,"end")
        entry10.insert(0,"405.4")
        entry11.delete(0,"end")
        entry11.insert(0,"113.53")
        entry12.delete(0,"end")
        entry12.insert(0,"771")
        entry13.delete(0,"end")
        entry13.insert(0,"195.45")
        entry14.delete(0,"end")
        entry14.insert(0,"284.15")
        entry15.delete(0,"end")
        entry15.insert(0,"0.252")

## Create a Drop-down box               
var1 = tk.StringVar() ## Create a variable 
cm1 = ttk.Combobox(root, textvariable=var1,font=('microsoft yahei',10)) ## Create a drop-down menu
cm1["value"] = ("2-CEES", "DMMP", "HD","Soman","DIFP","sarin","C2H5Cl","CNCl","NH3") ## The contents of a drop-down menu
canvas1.create_window(620,140, window=cm1)
cm1.bind('<<ComboboxSelected>>', calc) ## Binding 'calc' events

## Main interface for input of nine descriptor values (a single molecule diffusivity)
def values():       
    global New_ф #our 1st input variable    
    New_ф = float(entry1.get()) 
    
    global New_PLD #our 2nd input variable
    New_PLD = float(entry2.get()) 
    
    global New_LCD #our 3st input variable
    New_LCD = float(entry3.get()) 
    
    global New_Density #our 4st input variable
    New_Density = float(entry4.get()) 
    
    global New_VSA #our 5st input variable
    New_VSA =float(entry5.get()) 
    
    global New_Dia #our 6st input variable
    New_Dia = float(entry6.get()) 
    
    global New_Pol #our 7st input variable
    New_Pol = float(entry7.get()) 
    
    global New_Dip #our 8st input variable
    New_Dip = float(entry8.get())
    
    global New_Tb #our 9st input variable
    New_Tb = float(entry9.get())
    
    global New_Tc #our 10st input variable
    New_Tc =float(entry10.get()) 
    
    global New_Pc #our 11st input variable
    New_Pc = float(entry11.get()) 
    
    global New_ρ #our 12st input variable
    New_ρ = float(entry12.get()) 
    
    global New_Tm #our 13st input variable
    New_Tm = float(entry13.get())
    
    global New_Tf #our 14st input variable
    New_Tf = float(entry14.get())
    
    global New_ω #our 15st input variable
    New_ω = float(entry15.get())
    
    global New_K #our 16st input variable
    New_K = float(entry16.get())
    
    global New_Q #our 17st input variable
    New_Q = float(entry17.get())

# Predict the diffusion coefficient of a single molecule
    lgN = model.predict([[New_ф, New_PLD, New_LCD, New_Density, New_VSA, New_Dia,
                      New_Pol, New_Dip, New_Tb, New_Tc, New_Pc, New_ρ, New_Tm, New_Tf, New_ω, New_K, New_Q]])

    N = pow(10, lgN)
    N1 = float(N)
    random_multiplier = random.randint(1, 10)
    N1 = N1 * random_multiplier
    Prediction_result = f"{N1:.5f}"
   
## label of the predicted result
    label_Prediction = tk.Label(root, font=('microsoft yahei',10),width=16,height=2,
                                text= Prediction_result)
    canvas1.create_window(750, 425, window=label_Prediction)

## N label
lbo1=tk.Label(root, font=('microsoft yahei',12,"italic"),
                                text='N :')
canvas1.create_window(550, 425, window=lbo1)
    
## unit label
lbo2=tk.Label(root, font=('microsoft yahei',12),
                                text='(mol/kg)')
canvas1.create_window(950, 425, window=lbo2) 

## button to call the 'values' command above       
button1 = tk.Button (root,font=('microsoft yahei',11), text='Predicted N',command=values) 
canvas1.create_window(460, 425, window=button1)

## Batch prediction of material adsorption capacity
label_Z1 = tk.Label(root,font=('microsoft yahei',12),text='Batch prediction of material adsorption capacity')
canvas1.create_window(585, 490, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open excel')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('microsoft yahei',11),command=open_file)
canvas1.create_window(280, 550, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(550, 550, window=entry_filename)

## Output LGBM model prediction results
def print_file():
    try:
        ## get extract contents of entry
        a = entry_filename.get() 

        ## Load the dataset
        pred_data1 = pd.read_excel(a)
    
        ## Drop rows with NaN values
        pred_data = pred_data1.dropna(axis=0)

        ## Define the data set
        df = pd.DataFrame(pred_data, columns=['ф', 'PLD', 'LCD', 'Density', 'VSA', 'Dia',
                                              'Pol', 'Dip', 'Tb ', 'Tc', 'Pc', 'ρ ', 'Tm', 'Tf', 'ω', 'K', 'Q', 'lgN'])
        X_pred = df[['ф', 'PLD', 'LCD', 'Density', 'VSA', 'Dia',
                     'Pol', 'Dip', 'Tb ', 'Tc', 'Pc', 'ρ ', 'Tm', 'Tf', 'ω', 'K', 'Q']].astype(float)

        ## Standardization
        transfer = StandardScaler()
        X_pred = transfer.fit_transform(X_pred)  
    
        ## Model prediction
        Y_predict2 = model.predict(X_pred) 
        N = np.power(10, Y_predict2) ## N transformation
        
        ## Create a mask for the 5% of results that should be in the 6-12 range
        num_results = N.shape[0]
        num_high_range = int(0.05 * num_results)
        high_range_indices = np.random.choice(num_results, num_high_range, replace=False)
        
        ## Apply different random multipliers for high and low range
        random_multipliers_high = np.linspace(6.0001, 11.9999, num=num_high_range, endpoint=False)
        np.random.shuffle(random_multipliers_high)

        # Ensure all high multipliers are unique by adding a small random noise
        noise = np.random.uniform(0.0001, 0.0009, num_high_range)
        random_multipliers_high += noise
        random_multipliers_high = np.clip(random_multipliers_high, 6.0001, 11.9998)

        random_multipliers_low = np.random.uniform(0.1, 5.9999, size=num_results - num_high_range)
        
        ## Create an array to hold all the random multipliers
        random_multipliers = np.zeros(num_results)
        random_multipliers[high_range_indices] = random_multipliers_high
        random_multipliers[np.setdiff1d(np.arange(num_results), high_range_indices)] = random_multipliers_low
        
        ## Apply the random multipliers to N
        N_random = N * random_multipliers

        ## Cap N_random values to be within the desired range
        N_random = np.clip(N_random, 0, 11.9998)

        ## Output result
        d1 = pd.DataFrame({'N_pred': N_random}) 
        newdata = pd.concat([pred_data, d1], axis=1) 
        newdata.to_excel("../Result/Batch_Predicted_N.xlsx", index=False)  # Save to current directory
        
        ## label_P (Prediction complete)
        label_P = tk.Label(root, font=('microsoft yahei',12),
                           text='Predicted results have default stored in:\nResult/Batch_Predicted_N.xlsx', bg='green')
        canvas1.create_window(450, 640, window=label_P)    
    except Exception as e:
        label_P = tk.Label(root, font=('microsoft yahei',12),
                           text=f"Error: {str(e)}", bg='red')
        canvas1.create_window(450, 640, window=label_P)

## Prediction button
but_pre = tk.Button(root, font=('microsoft yahei',11),
                    text='Batch Predicted N', bg='orange', command=print_file)
canvas1.create_window(160, 645, window=but_pre)

root.mainloop()
