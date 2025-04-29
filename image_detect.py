# -*- coding: utf-8 -*-
print("Running image_detect")
bot_name = 'image_detect'

import sys
try:
    mother_folder = sys.argv[1] ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1
except:
    mother_folder = r"D:\test\Bots-2024\bot-others" ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1

sys.path.append(f"{mother_folder}/{bot_name}")  ## 指到自己的Folder
import setting_config

test = setting_config.test

## SQL 位址
env = setting_config.env
server = setting_config.DB_info['address']
database = ''
username = setting_config.DB_info['uid']  
password = setting_config.DB_info['pwd'] 

## ip
host_ip = setting_config.ip
port_ip = setting_config.port

import tensorflow as tf
import tensorflow_hub as hub
try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    
    tf.config.list_physical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(["GPU:0"])
except:
    pass

import torch
torch.cuda.is_available()

import ast
import string
import requests
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
## import ollama
import base64
from datasets import DatasetDict, Dataset, load_dataset, Features, concatenate_datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel

import numpy as np

import os
from transformers import pipeline
try:
    from qreader import QReader
except:
    os.remove(r"C:\Users\admin\AppData\Local\Temp\tfhub_modules")
    
qreader = QReader()
from deep_translator import GoogleTranslator

from paddleocr import PaddleOCR
en_ocr = PaddleOCR(lang="ch")
ct_ocr = PaddleOCR(lang="chinese_cht")
vi_ocr = PaddleOCR(lang="vi")

import cv2
from opencc import OpenCC
s2t = OpenCC('s2t')

import re
import pyodbc
cnxn = pyodbc.connect(f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={server}; DATABASE={database};UID={username};PWD={password}')
import pandas as pd
import json
import threading
import time

'''
keyword = ["gash","line","facebook","www","http","com","qr","yoe","wechat","bet","充值","儲","網","微信","支付","下載","登錄","在線","登入","註冊","投注", 
           "申請","賭場","娛樂","大獎","彩票","現金","回饋","百家樂","優惠","開戶","二維碼","掃碼","送","發貨","郵箱","信箱","福袋","禮",
           ".net",".app",".ip",".io",".play",".shop",".in",".jo",".co",".cc",".info",".game",".news",".es",".fish",".world",".life",".live",".org"]
'''

def reload():
    global keyword_cn_t, keyword_cn_s, keyword_en
    global threshold_porn, threshold_ads, threshold_QRcode, threshold_text
    data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
    keyword_cn_t = list(pd.read_sql(f"SELECT * FROM {data['Synoemotion']}", cnxn)['Text'])
    keyword_cn_s = list(pd.read_sql(f"SELECT * FROM {data['SynoemotionS']}", cnxn)['Text'])
    keyword_en = list(pd.read_sql(f"SELECT * FROM {data['SynoemotionE']}", cnxn)['Text'])
    
    threshold = pd.read_sql(f"SELECT * FROM {data['Option']}", cnxn)
    threshold_porn = float(threshold['Value'][threshold['Key']=='SimilarityImageDetectPorn'].values[0])
    threshold_ads = float(threshold['Value'][threshold['Key']=='SimilarityImageDetectAds'].values[0])
    threshold_QRcode = float(threshold['Value'][threshold['Key']=='SimilarityImageDetectQRcode'].values[0])
    threshold_text=0
    
    return {'threshold_ads': threshold_ads, 'threshold_QRcode': threshold_QRcode, 'threshold_porn': threshold_porn}

reload()

nsfw_model = "giacomoarienti/nsfw-classifier"

def save_result(func, args, result, j):
    result[j] = func(*args)
    
def four_detector(image, func_list, run_all, model_name):
    global threshold_porn, threshold_ads, threshold_QRcode, threshold_text  
    '''
    image_url = "https://blog.accupass.com/wp-content/uploads/2022/08/Cover-Tiny_facebook-ad-material.png"
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    test = four_detector(image, [], 1)
    '''
    if func_list==[]:
        func_list = ['ads', 'QRcode', 'text', 'porn']
        
    if run_all:
        results = [None for i in range(len(func_list))]
        for i,j in zip(func_list, range(len(func_list))):
            t = threading.Thread(target=save_result, args=(globals()[f'{i}_detector'], (image, model_name, globals()[f'threshold_{i}']), results, j))
            t.start()
            t.join()
    else:
        results = []
        for i in func_list:
            result = globals()[f'{i}_detector'](image, model_name, globals()[f'threshold_{i}'])
            results.append(result)
            if result['accept']:
                break
    return results

def porn_detector(image, model_name, threshold=0.8):
    ## porn, LukeJacob2023/nsfw-image-detector, giacomoarienti/nsfw-classifier
    detector_name='porn'
    image = image.convert("RGB")
    if model_name=='local':
        try:
            pipe = pipeline("image-classification", model=f"{mother_folder}/{bot_name}/Models/nsfw_model")(image)
        except:
            return "Model loaded fail"
            ## pipe = pipeline("image-classification", model=nsfw_model)(image)
            
        image = image.resize((331, 331))
        if (pipe[0]['label']=='porn' and pipe[0]['score']>=threshold) or (pipe[0]['label']=='hentai' and pipe[0]['score']>=threshold):
            if round(pipe[0]['score'],3)==1:
                return {'name': detector_name, 'accept': True, 'value': [{'label': pipe[0]['label'], 'score': 1}]}
            else:
                return {'name': detector_name, 'accept': True, 'value': [{'label': i['label'], 'score': round(i['score'], 3)} for i in pipe]}
        else:
            if round(pipe[0]['score'],3)==1:
                return {'name': detector_name, 'accept': False, 'value': [{'label': pipe[0]['label'], 'score': 1}]}
            else:
                return {'name': detector_name, 'accept': False, 'value': [{'label': i['label'], 'score': round(i['score'], 3)} for i in pipe]}
    elif model_name=='ollama':
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        image_bytes = output_buffer.getvalue()
        
        try:
            response = ollama.chat(model='llama3.2-vision', messages=[{'role': 'user',
                                                                       'content': 'Label this image, if it is NSFW, pick one label from porn or hentai, \
                                                                           if not pick one label from neutral, sexy or drawings. Just return only the label.',
                                                                       'images': [image_bytes]}])
        except:
            return "Ollama loaded fail"
            
        extract_str = "".join(i for i in response['message']['content'].lower() if i not in string.punctuation)
        if extract_str in ['porn', 'hentai']:
            return {'name': detector_name, 'accept': True, 'value': [{'label': extract_str, 'score': 1}]}
        elif extract_str in ['sexy', 'drawings', 'neutral']:
            return {'name': detector_name, 'accept': False, 'value': [{'label': extract_str, 'score': 1}]}
        else:
            porn_detector(image, "local")

def ads_detector(image, model_name, threshold=0.6):  
    ## ad
    detector_name = 'ads'
    image = image.convert("RGB")
    if model_name=="local":
        pipe = pipeline("image-classification", model="dhruv0808/autotrain-ad_detection_ver_1-1395053127")(image)
        if pipe[0]['label']=='Ads' and pipe[0]['score']>=threshold:
            if round(pipe[0]['score'],3)==1:
                return {'name': detector_name, 'accept': True, 'value': [{'label': pipe[0]['label'], 'score': 1}]}
            else:
                return {'name': detector_name, 'accept': True, 'value': [{'label': i['label'], 'score': round(i['score'], 3)} for i in pipe]}
        else:
            if round(pipe[0]['score'],3)==1:
                return {'name': detector_name, 'accept': False, 'value': [{'label': pipe[0]['label'], 'score': 1}]}
            else:
                return {'name': detector_name, 'accept': False, 'value': [{'label': i['label'], 'score': round(i['score'], 3)} for i in pipe]}
    elif model_name=='ollama':
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        image_bytes = output_buffer.getvalue()
        
        try:
            response = ollama.chat(model='llama3.2-vision', messages=[{'role': 'user',
                                                                       'content': 'Label this image, is it an advertisment? Return just only 0 or 1',
                                                                       'images': [image_bytes]}])
        except:
            return "Ollama loaded fail"
        
        extract_str = re.search(r'\d+', response['message']['content'])
        extract_str = int(extract_str.group()) if extract_str else None
        if extract_str == 1:
            return {'name': detector_name, 'accept': True, 'value': [{'label': "Ads", 'score': 1}]}
        elif extract_str == 0:
            return {'name': detector_name, 'accept': False, 'value': [{'label': "non_ads", 'score': 1}]}
        else:
            return ads_detector(image, "local")
    
def QRcode_detector(image, model_name, threshold=0.75):
    ## qrcode
    detector_name = 'QRcode'
    if model_name=="local":
        qr_image = image.convert('RGB')
        try:
            pipe = round(qreader.detect(image= np.array(qr_image))[0]['confidence'], 3)
            if pipe==0:
                return {'name': detector_name, 'accept': False, 'value': []}
            elif pipe >= threshold:
                return {'name': detector_name, 'accept': True, 'value': [{'label': detector_name, 'score':pipe}]}
            else:
                return {'name': detector_name, 'accept': False, 'value': [{'label': detector_name, 'score':pipe}]}
        except:
            return {'name': detector_name, 'accept': False, 'value': []}
    elif model_name=='ollama':
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        image_bytes = output_buffer.getvalue()
        
        try:
            response = ollama.chat(model='llama3.2-vision', messages=[{'role': 'user',
                                                                       'content': 'Label this image, does it contains QR Code? Return just only 0 or 1',
                                                                       'images': [image_bytes]}])
        except:
            return "Ollama loaded fail"
        
        extract_str = re.search(r'\d+', response['message']['content'])
        extract_str = int(extract_str.group()) if extract_str else None
        
        if extract_str == 1:
            return {'name': detector_name, 'accept': True, 'value': [{'label': detector_name, 'score': 1}]}
        elif extract_str == 0:
            return {'name': detector_name, 'accept': False, 'value': [{'label': detector_name, 'score': 0}]}
        else:
            return QRcode_detector(image, "local")
            
def text_detector(image, model_name, threshold):
     ## ocr
    detector_name = 'text'
    
    width, height = image.size
    if (width < 1024) and (height < 1024):
        if width >= height:
            image = image.resize((1024, int(1024*height/width)))
        else:
            image = image.resize((int(1024*width/height), 1024))
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.erode(image, kernel, iterations=1)
        
    try:
        pipe =  [[i[1][0].lower(), i[1][1]] for i in en_ocr.ocr(image)[0]]
        
        ## numbers
        pipe_n = [[''.join(re.findall(re.compile(r'\d{2,}'), i[0])), i[1]] for i in pipe]
        for i in pipe_n:
            if len(i[0]) >=6:
                return {'name': detector_name, 'accept': True, 'value': [{'label':"numbers", 'score': round(i[1], 3)}]}
            
        ## .com
        pipe_c = [[i[0], i[1]] for i in pipe if re.search(re.compile(r'\b\w{2,}\.[a-zA-Z]{2,}\b'), i[0])]
        if len(pipe_c) !=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"website", 'score': round(pipe_c[0][1], 3)}]}
        
        ## @
        pipe_a = [[i[0], i[1]] for i in pipe if re.search(re.compile(r'@\w{2,}\b'), i[0])]
        if len(pipe_a) !=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"contact", 'score': round(pipe_a[0][1], 3)}]}
        
        ## en_ocr
        pipe_en = [kw for kw in keyword_en if any(kw.lower() in i[0] for i in pipe)]
        if len(pipe_en)!=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"en_ocr", 'score': 1}]}
        
        ## cn_s        
        pipe_cn = [kw for kw in keyword_cn_s if any(kw in i[0] for i in pipe)]
        if len(pipe_cn)!=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"cns_ocr", 'score': 1}]}
    except:
        pass
    
    ## vi_ocr
    try:
        pipe =  [i[1][0] for i in vi_ocr.ocr(image)[0]]
        pipe = [kw for kw in keyword_cn_s if any(kw in i for i in pipe)]
        if len(pipe)!=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"vi_ocr", 'score': 1}]}
    except:
        pass
    
    ## cn_t
    try:
        pipe =  [i[1][0] for i in ct_ocr.ocr(image)[0]]
        pipe = [kw for kw in keyword_cn_t if any(kw in i for i in pipe)]
        if len(pipe)!=0:
            return {'name': detector_name, 'accept': True, 'value': [{'label':"cnt_ocr", 'score': 1}]}
        else:
            return {'name': detector_name, 'accept': False, 'value': []}
    except:
        return {'name': detector_name, 'accept': False, 'value': []}
    
def image_recognition(image):
    image = np.array(image)
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    
    probabilities = tf.nn.softmax(classifier(image)).numpy()
    
    top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()   
    
    temp = [[GoogleTranslator(source='en', target='zh-TW').translate(classes[i]), str(round(j,2))] for i,j in zip(top_5, probabilities[0][top_5])]
    return temp
    
def model_train(processor, model, dataset, target_folder):
    ## pytorch
    ## load data
    from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)

    from transformers import DefaultDataCollator
    data_collator = DefaultDataCollator()

    ## evaluate
    import evaluate
    accuracy = evaluate.load("accuracy")
    import numpy as np
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    ## train
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=f"{mother_folder}/{bot_name}/Models",
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.05,
        weight_decay=0,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{mother_folder}/{bot_name}/Models/{target_folder}")
    
    
'''
image regconition
'''
model_handle_map = {
  "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
  "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2",
  "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2",
  "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2",
  "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2",
  "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2",
  "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2",
  "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2",
  "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2",
  "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2",
  "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2",
  "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2",
  "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2",
  "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2",
  "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2",
  "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2",
  "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2",
  "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2",
  "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2",
  "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2",
  "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2",
  "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2",
  "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2",
  "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
  "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/classification/1",
  "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/classification/1",
  "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/classification/1",
  "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/classification/1",
  "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/classification/1",
  "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/classification/1",
  "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/classification/1",
  "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1",
  "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
  "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4",
  "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4",
  "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4",
  "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/classification/4",
  "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
  "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4",
  "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4",
  "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/classification/4",
  "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/classification/4",
  "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/classification/4",
  "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
  "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4",
  "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4",
  "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5",
  "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
  "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5",
  "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5",
}

model_image_size_map = {
  "efficientnetv2-s": 384,
  "efficientnetv2-m": 480,
  "efficientnetv2-l": 480,
  "efficientnetv2-b0": 224,
  "efficientnetv2-b1": 240,
  "efficientnetv2-b2": 260,
  "efficientnetv2-b3": 300,
  "efficientnetv2-s-21k": 384,
  "efficientnetv2-m-21k": 480,
  "efficientnetv2-l-21k": 480,
  "efficientnetv2-xl-21k": 512,
  "efficientnetv2-b0-21k": 224,
  "efficientnetv2-b1-21k": 240,
  "efficientnetv2-b2-21k": 260,
  "efficientnetv2-b3-21k": 300,
  "efficientnetv2-s-21k-ft1k": 384,
  "efficientnetv2-m-21k-ft1k": 480,
  "efficientnetv2-l-21k-ft1k": 480,
  "efficientnetv2-xl-21k-ft1k": 512,
  "efficientnetv2-b0-21k-ft1k": 224,
  "efficientnetv2-b1-21k-ft1k": 240,
  "efficientnetv2-b2-21k-ft1k": 260,
  "efficientnetv2-b3-21k-ft1k": 300, 
  "efficientnet_b0": 224,
  "efficientnet_b1": 240,
  "efficientnet_b2": 260,
  "efficientnet_b3": 300,
  "efficientnet_b4": 380,
  "efficientnet_b5": 456,
  "efficientnet_b6": 528,
  "efficientnet_b7": 600,
  "inception_v3": 299,
  "inception_resnet_v2": 299,
  "mobilenet_v2_100_224": 224,
  "mobilenet_v2_130_224": 224,
  "mobilenet_v2_140_224": 224,
  "nasnet_large": 331,
  "nasnet_mobile": 224,
  "pnasnet_large": 331,
  "resnet_v1_50": 224,
  "resnet_v1_101": 224,
  "resnet_v1_152": 224,
  "resnet_v2_50": 224,
  "resnet_v2_101": 224,
  "resnet_v2_152": 224,
  "mobilenet_v3_small_100_224": 224,
  "mobilenet_v3_small_075_224": 224,
  "mobilenet_v3_large_100_224": 224,
  "mobilenet_v3_large_075_224": 224,
}

model_name = "nasnet_large"
model_handle = model_handle_map[model_name]

try:
    classifier = hub.load(model_handle)
except Exception as e:
    try:
        file_path = re.search(r'(D:.*?tfhub_modules)', e.args[0]).group(1)
    except:
        file_path = re.search(r'(C:.*?tfhub_modules)', e.args[0]).group(1)
    
    try:
        [os.remove(f"{file_path}\{i}") for i in os.listdir(file_path)]
    except:
        print(f'請注意: Access Denied, please delete files in {file_path}')
    
image_size = model_image_size_map[model_name]

labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)
classes = []
with open(downloaded_file) as f:
    labels = f.readlines()
    classes = [l.strip() for l in labels]

'''
api
'''
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import JSONResponse
app = FastAPI()
print("API:Ready to GO")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

@app.get('/')
def hello():
    return "I am NSFW & IR 20250321"

## reload
@app.get('/updatefaq')
def reload_threshold():
    temp = reload()
    return temp

## nfsw
@app.get('/nsfw_url')
def nsfw_url_get(request: Request):
    global check
    image_url = request.query_params.get('url', '')
    func_list = ast.literal_eval(request.query_params.get('func_list', '[]'))
    run_all = int(request.query_params.get('run_all', '1'))
    model_name = request.query_params.get('model_name', 'local')
        
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    
    check = four_detector(image, func_list, run_all, model_name)
    return check
    
@app.post('/nsfw_url')
async def nsfw_url_post(request: Request):
    global check
    data = await request.json()
    image_url = data.get('url', '')
    func_list = ast.literal_eval(data.get('func_list', '[]'))
    run_all = int(data.get('run_all', '1'))
    model_name = data.get('model_name', 'local')
    
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    
    check = four_detector(image, func_list, run_all, model_name)
    return check

@app.get('/nsfw_base')
def nsfw_base_get(request: Request):
    global check
    image_base = request.query_params.get('base', '')
    func_list = ast.literal_eval(request.query_params.get('func_list', '[]'))
    run_all = int(request.query_params.get('run_all', '1'))
    model_name = request.query_params.get('model_name', 'local')
    
    image_data = base64.b64decode(image_base)
    image = Image.open(BytesIO(image_data))
    
    check = four_detector(image, func_list, run_all, model_name)
    return check
    
@app.post('/nsfw_base')
async def nsfw_base_post(request: Request):
    global check
    try:
        data = await request.json()
        image_base = data.get('base', '')
        func_list = ast.literal_eval(data.get('func_list', '[]'))
        run_all = int(data.get('run_all', '1'))
        model_name = data.get('model_name', 'local')
            
        try:
            image_data = base64.b64decode(image_base)
            image = Image.open(BytesIO(image_data))
            check = four_detector(image, func_list, run_all, model_name)

            return check
        except Exception as e:
            return f"Error processing image: {str(e)}"

    except Exception as e:
        return f"Internal server error: {str(e)}"


@app.get('/nsfw_file')
def nsfw_file_get(request: Request):
    global check
    file_path = request.query_params.get('file_path', '')
    func_list = ast.literal_eval(request.query_params.get('func_list', '[]'))
    run_all = int(request.query_params.get('run_all', '1'))
    model_name = request.query_params.get('model_name', 'local')
    
    image = Image.open(file_path)
    
    check = four_detector(image, func_list, run_all, model_name)
    return check
    
@app.post('/nsfw_file')
async def nsfw_file_post(request: Request):
    global check
    data = await request.json()
    file_path = data.get('file_path', '')
    func_list = ast.literal_eval(data.get('func_list', '[]'))
    run_all = int(data.get('run_all', '1'))
    model_name = data.get('model_name', 'local')
    
    image = Image.open(file_path)
    
    check = four_detector(image, func_list, run_all, model_name)
    return check


## ir
@app.get('/ir_url')
def ir_url_get(request: Request):
    global check
    image_url = request.query_params.get('url', '')
        
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    
    check = image_recognition(image)
    return check
    
@app.post('/ir_url')
async def ir_url_post(request: Request):
    global check
    data = await request.json()
    image_url = data.get('url', '')
    
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    
    check = image_recognition(image)
    return check


@app.get('/ir_base')
def ir_base_get(request: Request):
    global check
    image_base = request.query_params.get('base', '')
        
    image_data = base64.b64decode(image_base)
    image = Image.open(BytesIO(image_data))
    
    check = image_recognition(image)
    return check
    
@app.post('/ir_base')
async def ir_base_post(request: Request):
    global check
    data = await request.json()
    image_base = data.get('base', '')
    
    image_data = base64.b64decode(image_base)
    image = Image.open(BytesIO(image_data))
    
    check = image_recognition(image)
    return check


@app.get('/ir_file')
def ir_file_get(request: Request):
    global check
    file_path = request.query_params.get('file_path', '')
    
    image = Image.open(file_path)
    
    check = image_recognition(image)
    return check
    
@app.post('/ir_file')
async def ir_file_post(request: Request):
    global check
    data = await request.json()
    file_path = data.get('file_path', '')
    
    image = Image.open(file_path)
    
    check = image_recognition(image)
    return check

@app.get('/train_nsfw_model')
def train_nsfw_model(request: Request):
    url_folder = request.query_params.get('file_path', '')
    
    try:
        data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
        check = pd.read_sql(f"SELECT * FROM {data['imageIndexDetail']}", cnxn)
        normal = list(set(check[(check['IsBot']==1) & (check['Index']=='porn') & (check['IsAccept']==0) & (check['FinalState']=='Accept')]['ImageUrl']))
        nsfw = list(set(check[(check['IsBot']==0) & (check['Index']=='porn') & (check['FinalState']=='Reject')]['ImageUrl']))
    except:
        pass
    
    labels = ['hentai', 'drawings', 'porn', 'sexy', 'neutral']
    
    for i in labels:
        for j in os.listdir(f'{mother_folder}/{bot_name}/Models/nsfw/{i}'):
            try:
                Image.open(f'{mother_folder}/{bot_name}/Models/nsfw/{i}/{j}')
            except:
                os.remove(f'{mother_folder}/{bot_name}/Models/nsfw/{i}/{j}')
            
    ## load data
    dataset = load_dataset("imagefolder", data_dir=f"{mother_folder}/{bot_name}/Models/nsfw", split='train')
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(dataset.features["label"].names):
        label2id[label] = i
        id2label[i] = label
    
    try:
        new_data = {"image": [f"{url_folder}/{i}" for i in normal]+[f"{url_folder}/{i}" for i in nsfw],
                    "label": [label2id['normal'] for i in normal]+[label2id['nsfw'] for i in nsfw]}
        
        dataset = concatenate_datasets([dataset, Dataset.from_dict(new_data, features=dataset.features)])
    except:
        pass
    
    dataset = dataset.train_test_split(test_size=0.05)

    ## load model
    processor = AutoImageProcessor.from_pretrained(nsfw_model)
    model = AutoModelForImageClassification.from_pretrained(nsfw_model,
                                                            label2id=label2id,
                                                            id2label=id2label,
                                                            ignore_mismatched_sizes = True)
    target_folder = 'nsfw_model'
    model_train(processor, model, dataset, target_folder)
    return "train nsfw_model complete"


import uvicorn
if test:
    uvicorn.run(app, host='192.168.2.209', port=4090)
else:
    uvicorn.run(app, host=host_ip, port=port_ip)

