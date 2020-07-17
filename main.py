from flask import Flask, render_template, url_for, request
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
import re
import numpy as np
import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import nltk
from nltk import tokenize
from nltk.tokenize import sent_tokenize
dt= datetime.datetime.now().strftime('%Y-%b-%d-%H-%M-%S')


app = Flask(__name__)



def literaryFormat(string):
    string = string.lower()
    string = string.replace(" ", '-')
    return string

def historyFormat(string):
    string = string.title()
    string = string.replace(' ', '_')
    return string

def ProcessText(path, fNameOut):
    for file in os.scandir(path):
        if (file.path.endswith('.txt')):
            fo = open(fNameOut,'a',encoding='utf=8')
            with open(file.path,'r',encoding='utf=8') as f:
                for line in f:
                    line = re.sub(r'"','',line) #remove ""
                    line = line.replace('[', '<')
                    line = line.replace(']', '>')
                    line = line.replace('. ', '.\n')
                    line = re.sub(r" ?\([^)]+\)", "", line)#remove parenthese
                    line = re.sub(r'<[^>]*>', "", line) #remove audience or other speakers
                    line = re.sub(r'D.','',line) #remove D.
                    line = re.sub(r'E.','',line) #remove E.
                    line = re.sub(r'  ','',line) #remove double spaces
                    line = re.sub(r'^ +','',line) #remove space at start of line
                    if line != '\n':
                        fo.write(line)
            fo.close()
            print('file cleaned')

class SpeechData(Dataset):
    def __init__(self, fname):
        super().__init__()
        self.data = []
        self.eol = "\n"

        with open(fname, 'r', encoding="utf=8") as f:
            i=0
            for line in f:
                
                for sent in tokenize.sent_tokenize(line): #break up into sentences
                    self.data.append(sent)

               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

def Train(model,loader,fname):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learningRate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmupSteps, num_training_steps=trainingSteps, last_epoch = -1)
        
    sumLoss = 0
    step = 0
    model.train()
    for epoch in range(epochs):
        for l in tqdm(loader):
            inputs = torch.tensor(tokenizer.encode(' '.join(l))).unsqueeze(0).to(device) #tensor size 1 at dim 0
            output = model(inputs, labels=inputs)
            loss, logits = output[:2]                        
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            tb.add_scalar('Loss', loss,step)
            step +=1
            scheduler.step() 

    tb.flush()
    tb.close()
    model.to('gpu')
    torch.save(model.state_dict(), "%s_%s_%s.pt"%(fname,modelName,epochs))

def EvaluateModel(model,tokenizer, prompts, attempts, words, search):
    model.eval()
    for prompt in prompts:
        with torch.no_grad():
            for i in range(attempts):
                tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
                for j in range(words):
                    inputs = tokens
                    output = model(inputs, labels=inputs)
                    loss, logits = output[:2]
                    softmaxLogits = torch.softmax(logits[0,-1],dim=0) # gets final column?
                    newToken = search(softmaxLogits)
                    tokens =torch.cat((tokens,newToken.unsqueeze(0)),1)
                print(tokenizer.decode(list(tokens.squeeze().to('cpu').numpy())))
                print("\n")
                return tokenizer.decode(list(tokens.squeeze().to('cpu').numpy()))


def TopKChoice(prob,k=10):
    prob, indx = (prob.sort(descending=True)) #sort by probability 
    nProb, nIndx = prob[:k], indx[:k] #grab k items
    normProb = nProb/torch.sum(nProb) #normalize
    choiceIdx = torch.multinomial(normProb, 1) #choose 1 based on norm probability
    return nIndx[choiceIdx] #get tokenID



driver = webdriver.Chrome("chromedriver.exe")
literarywebsite = "https://literatureessaysamples.com/category/"
historicalwebsite = "https://en.wikipedia.org/wiki/"



tb = SummaryWriter('runs/%s'%dt)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


modelName = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(modelName)
essayModel = GPT2LMHeadModel.from_pretrained(modelName).to(device)

learningRate = 0.0001
warmupSteps = 10
trainingSteps = 100
epochs = 4
batchSize = 15

prompts = []
words = 300
attempts = 2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/write', methods=['POST'])
def write():
    if request.method == 'POST':
        subject = request.form['subject']
        topic = request.form['topic']
        prompts = request.form['starts']
        prompts = sent_tokenize(prompts)
        print(subject + ', ' + topic)

        if subject.lower() == 'literature':
            literaryq = topic
            literaryq = literaryFormat(literaryq)
            website = literarywebsite + literaryq + '/'
            driver.get(website)
            corpus = driver.find_elements_by_css_selector('p')
            c = open('corpus.txt', 'w', encoding='utf=8')
            for text in corpus:
                c.write(text.text)
            try:
                if driver.find_element_by_link_text('2') == True:
                    cookies = driver.find_element_by_xpath("//*[@id='moove_gdpr_cookie_info_bar']/div/div/div[2]/button")
                    cookies.click()
                    element = driver.find_element_by_link_text('2')
                    element.click()
                    corpus = driver.find_elements_by_css_selector('p')
                    c = open('corpus.txt', 'w', encoding='utf=8')
                    for text in corpus:
                        c.write(text.text)
                if driver.find_element_by_link_text('3') == True:
                    element = driver.find_element_by_link_text('3')
                    element.click()
                    corpus = driver.find_elements_by_css_selector('p')
                    c = open('corpus.txt', 'w', encoding='utf=8')
                    for text in corpus:
                        c.write(text.text)
                if driver.find_element_by_link_text('4') == True:
                    element = driver.find_element_by_link_text('4')
                    element.click()
                    corpus = driver.find_elements_by_css_selector('p')
                    c = open('corpus.txt', 'w', encoding='utf=8')
                    for text in corpus:
                        c.write(text.text)
                if driver.find_element_by_link_text('5') == True:
                    element = driver.find_element_by_link_text('5')
                    element.click()
                    corpus = driver.find_elements_by_css_selector('p')
                    c = open('corpus.txt', 'w', encoding='utf=8')
                    for text in corpus:
                        c.write(text.text)
            except NoSuchElementException:
                print("...")
            c.close()
            print('file ready')
        elif subject.lower() == 'history' or subject.lower() == 'history.':
            historyq = topic
            historyq = historyFormat(historyq)
            website = historicalwebsite + historyq
            driver.get(website)
            corpus = driver.find_elements_by_css_selector('p')
            c = open('corpus.txt', 'w', encoding='utf=8')
            for text in corpus:
                c.write(text.text)
            c.close()
            print('file ready')
        ProcessText("C:/Users/bashi/", "corpus_new.txt")

        essayText = SpeechData("corpus_new.txt")
        essaySpeechLoader = DataLoader(essayText, batch_size=batchSize, shuffle=True)
        try:
            Train(essayModel, essaySpeechLoader, 'Essay Writer')
        except:
            print("Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu device type at start of device string: gpu")
        essay = EvaluateModel(essayModel.to(device), tokenizer, prompts, attempts, words, TopKChoice)
        os.remove("corpus_new.txt")
        os.remove("corpus.txt")

        return render_template('results.html', essay=essay)



if __name__ == '__main__':
    app.run()