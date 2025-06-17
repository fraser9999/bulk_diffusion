# Bulk Diffusion 0.1a
#
# Massive and Continous Picture Rendering 
# Author: Hermann Knopp
# Version: 0.1a Date: 31.7.2023
# Contact: hermann.knopp@gmx.at
# Use: Python 3.10.10 x64/amd64
# with additional libraries
# you should make an virtual
# python environment with venv
# if you have more python setups
# look at: requirements.txt
 

# System Init
import os
os.system("cls")
os.system("title GPT/Diffusion Continous Picture Renderer")
os.system('mode con: cols=100 lines=40')


print("Importing Libs... please wait")
import sys


# Window System Libs
from tkinter import *
from PIL import Image,ImageTk


# for utf8 txt encoding
import codecs


# Libs for Data Structures
import numpy as np 
import pandas as pd


# String Libs
import time
import glob
import os.path
import random
from random import randrange


# GPT Neo Libs
from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
from aitextgen.TokenDataset import TokenDataset, merge_datasets
from aitextgen.utils import build_gpt2_config
from aitextgen.tokenizers import train_tokenizer


# Torch Libs
import torch
from torch import autocast


# Clipboard Libs - not used
# import pyperclip 


# Diffuser Libs
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,EulerAncestralDiscreteScheduler
from datetime import datetime


# Image Libs
from PIL import Image


# Huggingface Space Libs
#from huggingface_hub import hf_hub_download
#import joblib


# clear screen
os.system("cls")


# Setup Standard Parameters for Diffuser Model
prompt = "A Photo of  Anne Hathaway, Jeans, white Shirt" 
negative_prompt = "" 
num_samples = 1 
guidance_scale = 7.5 
num_inference_steps = 16 
height = 512 
width = 512 
seed = 1000


# Display Window
main=Tk()
main.title("GPT/Diffusers Mass Picture Render Preview")
canvas = Canvas(main,width=768,height=768)
canvas.pack()


# Main Title Text
print("GPT/Diffusion Continous Picture Renderer (Infinity Batch)")
print("")
print("")


# Select Save Path
fod=input("Do you want to change Image Out folder (Y/N)")

if fod =="":
   fod="N"

if fod=="Y" or fod=="y" or fod=="j" or fod=="J":
   fod_path=input("Enter new Path:")
   img_out_folder=fod_path
   
   # Check whether the specified path exists or not
   isExist = os.path.exists(img_out_folder)
   if not isExist:
   
       # Create a new directory because it does not exist
       os.makedirs(img_out_folder)
       print("The new directory " + img_out_folder + " is created!")   
       print("")
       print("I am using: "+img_out_folder+" for Picture saving")

if fod=="N" or fod=="n":
   
    # Make Custom Picture Dir in .Py Working Folder

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = "pictures"
    filepath= dir_path + "\\" + path
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(filepath)
    if not isExist:
    
        # Create a new directory because it does not exist
        os.makedirs(filepath)
        print("The new directory 'ideas' is created!")

    # Select new path Variable
    img_out_folder=filepath
    print("I am using: "+img_out_folder+" for Picture saving")
    print("")

# Check Filepath String of Slash Errors
if img_out_folder[:-1] !="/":
    img_out_folder = img_out_folder + "/"

print("")
print("")
# Select GPT Engine
gpt=input("Do you want to use GPT Prompt Engine (Y/N)")
if gpt =="":
    gpt="N"
if gpt=="Y" or gpt=="y" or gpt=="j" or gpt=="J":
    gpt_flag=1
    print("I am using GPT Prompt Engine")
if gpt=="N" or gpt=="n":
    gpt_flag=0
    print("I am using Style Prompt only")
print("")


# Wait for User
a=input("Wait Key..for selecting Models")
os.system("cls")


# Select Diffuser Binary Format Diffusion Models
print("Please Select SD-Model from List (1-7)")
print("")
print("Model (1) = Stable Diffusion 1.5 - GPT Lexart")
print("Model (2) = Dreamlike Diffusion - Photorealistic GPT Lexart")
print("Model (3) = Dreamlike Diffusion - Creative Art GPT Lexart")
print("Model (4) = Stable Diffusion 1.5 - GPTNeo Custom")
print("Model (5) = Dreamlike Diffusion - Photorealistic GPTNeo Custom")
print("Model (6) = Dreamlike Diffusion - Creative Art GPTNeo Custom")
print("Model (7) = Inkpunk Diffusion - GPTNeo Custom")
print("")
print("for selecting GPTNeo Custom you have to download github release")
print("with pytorch_model.bin 'model' folder/files (Size 500MB)")

print("")


# Test Model Nr. Input
modelnr = input("Model Nr: (1-7)")
if modelnr==None:
    modelnr=1
if modelnr=="":
    modelnr=1
modelnr=int(modelnr)
if modelnr==0:
    modelnr=1
if modelnr>7:
    modelnr=1


# Clear Init Prefix Prompt
prefix_prompt=""


# Select Diffuser Model from Nr. Input
# scheduler_flag=0 ... Euler Scheduler - epsilon_prediction
# scheduler_flag=1 ... Euler Ancestral Scheduler - v_prediction

if modelnr ==1:
    model_path="runwayml/stable-diffusion-v1-5"
    gpt_path="AUTOMATIC/promptgen-lexart"
    gpt_flag=0
    model_prompt="4k, 8k, award winning, amazing"
    prefix_prompt="photo of a "
    scheduler_flag=0

if modelnr ==2:
    model_path="dreamlike-art/dreamlike-photoreal-2.0"
    gpt_path="AUTOMATIC/promptgen-lexart"
    gpt_flag=0
    model_prompt="4k, 8k, award winning, amazing"
    prefix_prompt="photo of a "
    scheduler_flag=0

if modelnr ==3:
    model_path="dreamlike-art/dreamlike-photoreal-2.0"
    gpt_path="AUTOMATIC/promptgen-lexart"
    gpt_flag=0
    model_prompt=""
    prefix_prompt="picture of a "
    scheduler_flag=0

if modelnr ==4:
    model_path="runwayml/stable-diffusion-v1-5"
    gpt_path="model"
    gpt_flag=1
    model_prompt="4k, 8k, award winning, amazing"
    prefix_prompt="photo of a "
    scheduler_flag=0

if modelnr ==5:
    model_path="dreamlike-art/dreamlike-photoreal-2.0"
    gpt_path="model"
    gpt_flag=1
    model_prompt="4k, 8k, award winning, amazing"
    prefix_prompt="photo of a "
    scheduler_flag=0

if modelnr ==6:
    model_path="dreamlike-art/dreamlike-photoreal-2.0"
    gpt_path="model"
    gpt_flag=1
    model_prompt=""
    prefix_prompt="picture of a "
    scheduler_flag=0

if modelnr ==7:
    model_path="Envvi/Inkpunk-Diffusion"
    gpt_path="model"
    gpt_flag=1
    model_prompt=""
    prefix_prompt="nvinkpunk photo of a "
    scheduler_flag=0




# System Message
print("")
print("Please check Model Path and File Format, if error is shown here...")
print("Diffusion Model is loading from huggingface.co or from harddrive/.cache")
print("... please wait")
print("")
print("")


# Select GPU Mode always (CPU not supported today)
mode=1

# cuda or cpu
if mode==1:
   device = "cuda" 
else:
   device = "cpu"


# Import Custom Libs for Log Window Text
import logging

# disable warnings
logging.disable(logging.WARNING)  


# Setup Diffuser Pipeline
if 'pipe' not in locals():

    if scheduler_flag==0:
        scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    if scheduler_flag==1:
        scheduler = EulerAncestralDiscreteScheduler( beta_start=0.0001, beta_end=0.02, beta_schedule="scaled_linear", prediction_type = 'v_prediction' ) 

    if device=='cuda':  
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None,torch_dtype=torch.float32).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None,torch_dtype=torch.float32).to(device)


# use Faster Render with Xformers
g_cuda = None
g_cuda = torch.Generator(device=device)
pipe.enable_xformers_memory_efficient_attention()


# Clear Console
os.system("cls")


# Load Main Words File (English Words 60k)
wordspath=os.path.dirname(os.path.realpath(__file__))
wordspath=wordspath + "\\words.txt"

print("Loading Wordlist Files..")
print("")
print("Wordsfile: "+ wordspath)

filename=wordspath
with open(filename, encoding="utf8") as f:
    words = f.readlines()
    words = [x.strip() for x in words]
    anz = len(words) 
    random.shuffle(words)


# Load Main Categories File  (English Picture Categories)
wordspath2=os.path.dirname(os.path.realpath(__file__))
wordspath2=wordspath2 + "\\categories.txt"

print("Categoryfile: "+ wordspath2)

filename2=wordspath2
with open(filename2, encoding="utf8") as f2:
    words2 = f2.readlines()
    words2 = [x.strip() for x in words2]
    anz2 = len(words2) 
    random.shuffle(words2)


# Load Main Moods File  (English Words about Picture Mood)
wordspath3=os.path.dirname(os.path.realpath(__file__))
wordspath3=wordspath3 + "\\moods.txt"

print("Moodsfile: "+ wordspath3)

filename3=wordspath3
with open(filename3, encoding="utf8") as f3:
    words3 = f3.readlines()
    words3 = [x.strip() for x in words3]
    anz3 = len(words3) 
    random.shuffle(words3)


# Load Model und generate prompt for Stable Diffusion 


# Select Custom Harddrive Model Path or download from huggingface.co 

if gpt_flag==0:
   # Load Static GPT Model (promptgen-lexart with 350MB)
   # model = "AUTOMATIC/promptgen-lexart"
   model=gpt_path  

if gpt_flag==1:
   # Load Model from harddrive "github repository" at .py working folder
   model=os.path.dirname(os.path.realpath(__file__))
   model = model + "\\" + gpt_path




# Status Message Model Download
print("")
print("")
print("Loading the GPT Model from huggingface.co or from harddrive/.cache")
print("Model is 300-500MB download... please wait...will take some time")
print("")

# select/load/init GPT Model
prompt_ai = aitextgen(model, to_gpu=True)



# Clear Init Negative Prompt
negative_prompt=""


# Start Batch Render J/N
bat=input("Do you want Start Batch Rendering... (Press Return Key)")


# Clear Window
os.system("cls")


# Init Variable Picture Counter
banz=0


# Main Render Loop
while True:


    # Set Object Type of Picture Randomly
    
    # Find Random Picture Category Word
    catrn=randrange(1,anz2)
    cat_prompt = words2[catrn]

    # Find Random Moods Word
    moodrn=randrange(1,anz3)
    mood_prompt = words3[moodrn]


    # Status Message wich Category/Moods Word selected
    print("RND Nr.",str(catrn)," ",str(moodrn))
    print("Category is:  " , cat_prompt)
    print("Mood is:  " , mood_prompt)
   
    # Randomly select Picture describing Words from one to three
    we=randrange(1,5)
    print("Word Limit:",str(we))
    

    # No Word
    if we==1:
            
        # add Strings to Prompt
        prmpt = prefix_prompt + " " + cat_prompt + ", " + mood_prompt + ", " + model_prompt + ","    

    # One Word
    if we==2:
       
        # find random Word #1
        wo=randrange(anz)
        prmpt=words[wo] 
        print("Random Word:  " + words[wo])       

        # add Strings to Prompt
        prmpt = prefix_prompt + " " + cat_prompt + ", " + mood_prompt + ", " + prmpt + ", " + model_prompt + ","

    # Two Words
    if we==3:
        
        # find random Word #1
        wo=randrange(anz)
        prmpt=words[wo] 
        print("Random Word #1:  " + words[wo])       

        # find random Word #2
        wo2=randrange(anz)
        prmpt2=words[wo2] 
        print("Random Word #2:  " + words[wo2])       

        # add Strings to Prompt
        prmpt = prefix_prompt + " " + cat_prompt + ", " + mood_prompt + ", " + prmpt + " " + prmpt2 + ", " + model_prompt + ","
    
    # Three Words
    if we>=4:
    
        # find random Word #1
        wo=randrange(anz)
        prmpt=words[wo] 
        print("Random Word #1:  " + words[wo])       

        # find random Word #2
        wo2=randrange(anz)
        prmpt2=words[wo2] 
        print("Random Word #2:  " + words[wo2])       

        # find random Word #3
        wo3=randrange(anz)
        prmpt3=words[wo3] 
        print("Random Word #3:  " + words[wo3])    


        # add Strings to Prompt
        prmpt = prefix_prompt + " " + cat_prompt + ", " + mood_prompt + ", " + prmpt + " " + prmpt2 +" with " + prmpt3  + ", " + model_prompt + ","

      
    # Select GPT Engine Enhanced Prompt or Wordlist Prompt only
    
    if gpt_flag==0:
        print("Random Prompt: ",prmpt)
        txtprmpt=prmpt

    if gpt_flag==1:

        # generate Neuronal Model Prompt  
        txtprmpt = prompt_ai.generate_one(prompt=prmpt)

        # add activation Token "Photo of" to prompt
        txtprmpt = "" + txtprmpt

        # delete last 2 Chars
        txtprmpt = txtprmpt[:-2]

        # copy prompt to clipboard - not used
        # pyperclip.copy(txtprmpt)

        # Status Message Print Prompt 
        print("Random GPT Prompt: " + txtprmpt)
        print("")
    
    
    # Set System Variable MAX Seed
    max=4294967295


    # generate random Seed Value
    seed=random.randint(1,max)


    # Status/Set Random Seed. always another Picture 
    print("Set Seed to: " + str(seed))
    g_cuda = torch.Generator(device).manual_seed(seed)


    # Test if last possible Picture is rendered than exit
    if banz==sys.maxsize:
        print("Your have reached end of rendering , no more images")
        a=input("Wait Key")
        exit()


    # Variable Picture Counter
    banz=banz+1

    
    # Status Message Rendering actual Picture Number
    bmax = sys.maxsize
    print("Rendering Image Nr: " + str(banz) + " from " + str(bmax)) 

 
    # GPU Select Test
    if device=='cuda':
           
        # Render Picture with Prompt
        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                      prompt=txtprmpt,
                      height=height,
                      width=width,
                      negative_prompt=negative_prompt,
                      num_images_per_prompt=num_samples,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale,
                      generator=g_cuda
                   ).images
       
    
            # Save Images/Display Window with File
            for img in images:
                # datetime object containing current date and time
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H%M%S")
                filename=img_out_folder + "test_" + dt_string +".png"
                img.save(filename)
                
                # Save Prompt to txt File
                textfilename=img_out_folder + "test_" + dt_string +".txt"
                textprompt="Prompt: " + txtprmpt + "  Seed: " + str(seed)
               
                try:
                    file = codecs.open(textfilename, "w", "utf-8")
                    file.write("Hermanns GPT Mass Picture Prompt renderer")                   
                    file.write("\n")
                    file.write("(C)2023 - 29.7.2023")
                    file.write("\n")
                    file.write("Date:" + str(dt_string))
                    file.write("\n")
                    file.write("Category:" + cat_prompt)
                    file.write("\n")
                    file.write("Mood:"+ mood_prompt)
                    file.write("\n")
                    file.write("Seed:" + str(seed))
                    file.write("\n")
                    file.write("Image:" +str(height) + "x" + str(width))
                    file.write("\n")
                    file.write("Config Scale:"+str(guidance_scale))
                    file.write("\n")
                    file.write("Num Samples."+str(num_samples))
                    file.write("\n")
                    file.write("Negative Prompt:")
                    file.write("\n")
                    file.write(str(negative_prompt))
                    file.write("\n")
                    file.write("\n") 
                    file.write("Positive Prompt:")
                    file.write("\n")
                    file.write(textprompt)
                    file.close()
                  
              
                    # check for window-reopen
                    if 'normal' != main.state():
                        #Display Window
                        main=Tk()
                        canvas = Canvas(main,width=768,height=768)
                        canvas.pack()
                
                    image = Image.open(filename)
                    image2 = image.resize((768,768))

                    tk_image = ImageTk.PhotoImage(image2)
                    canvas.create_image(1,1, anchor=NW, image=tk_image)
                    canvas.update()
 
                except:
                    main=Tk()
                    canvas = Canvas(main,width=768,height=768)
                    canvas.pack() 
                    image = Image.open(filename)
                    image2 = image.resize((768,768))
                    tk_image = ImageTk.PhotoImage(image2)
                    canvas.create_image(1,1, anchor=NW, image=tk_image)
                    canvas.update()                     
                    continue


            # Slow down for display Picture one Second
            time.sleep(1)


   
    # Cpu Select Code

    # Cpu Code not ready today (comming soon)   
    else:
        print("CPU Selected,but not supporting this feature today")
        a=input("Wait Key, to end")
        exit()

   
