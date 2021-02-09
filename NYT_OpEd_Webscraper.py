#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:40:15 2021

@author: samueljocas

Description: This program reads in a list of New York Times columnists, and then scrapes 
each Op-Ed article off the web through BeautifulSoup. It then creates a dataframe that
records each article URL, its headline, its author, and its article text. This is then exported
to a desired folder as a .csv so that it can be used as ML training data
"""
############ Scrapes all the data for NYT columnists 
import requests #url extraction
from bs4 import BeautifulSoup #webscraper
import pandas as pd #dataframes 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #sentiment analyzer
import time #to check runtime speed 
import datetime as dt

data_path = '/Users/samueljocas/Desktop/Data Science Portfolio/New York Times ML Algorithm  /Op-Ed Training Datasets'

#converts list of proper-written names to ones that fit in the website link for the webscraper
def nameToWeb(names):
    names1 = names.copy()
    names1 = [i.lower() for i in names1]
    names1 = [i.replace(' ','-') for i in names1]
    names1 = [i.replace('.','') for i in names1] #Paul R. Krugman becomes paul-r-krugman
    return names1

#ensures all scraped urls can be accessed via Beautiful Soup               
def ensureURL(newlist):
    new_urls =[]
    for i in newlist:
        if i[0] != "h":
            i = i.replace('', 'https://www.nytimes.com', 1)
            new_urls.append(i)
        else:
            new_urls.append(i)
    return new_urls 

#loop through each columnist 
def scrapeArticles(journos):
    #establish sets of blank lists to extract info from New York Times 
    name_list = [] #tells which article was written by who
    url_list = [] #all scraped article URLs
    news_text = [] #text from each news article 
    headlines = [] #headlines
    df = pd.DataFrame() #dataframe that contains all scraped info  
    webnames =  nameToWeb(journos) #make the journalist names url-friendly
    
    #chunk that extracts all webpages from website 
    for name in webnames:
        url = "https://www.nytimes.com/column/"+ name 
        print("Scraping data for " + str(name) + ".....")
        request = requests.get(url)
        soup = BeautifulSoup(request.text, "html.parser")
        for links in soup.find_all('div', {'class': 'css-13mho3u'}):  #parse webpage for all article links 
            for info in links.find_all('a'):
                if info.get('href') not in url_list:
                    url_list.append(info.get('href'))
                    name_list.append(name)
    url_list = ensureURL(url_list) #ensure all scraped urls are complete
    df["URLs"]  = url_list #add URLs to df
    
     #parse through each url to scrape headlines and append to list 
    for url in url_list:  
        if url.endswith('.html'): #makes the headlines 
            url = url[:-5]
        headlines.append(url.split("/")[-1].replace("-", " "))
    df["Headlines"] = headlines
    df["Names"] = name_list
    
    df.drop(df[df.Headlines == ''].index, inplace=True) #drop any rows that could not gather headlines 
    df = df.reset_index(drop=True) #reset index to accomodate dropped rows 
    
    #store the text of the news in df 
    for i in range(0,len(df["URLs"])):
        temp = []
        url = df["URLs"][i]   
        print("Fetching article text for " + str(url))
        request = requests.get(url)
        soup1 = BeautifulSoup(request.text, "html.parser")
        for news in soup1.find_all('p'): #goes through all text for each link 
            temp.append(news.text) #appends each block of text to temp list 
          #identify the first and last line of the news article to get rid of "fluff" content  
        for first_sentence in temp:
            if first_sentence.split(" ")[-1] == "Columnist" and first_sentence.split(" ")[0] == "Opinion":
                break
            elif first_sentence == 'Opinion' or first_sentence == "Comentario": 
                break
        for last_sentence in reversed(temp):
            if last_sentence.split(" ")[0]== "The" and last_sentence.split(" ")[-1] == "letters@nytimes.com.":
                break
            
        joined_text = ' '.join(temp[temp.index(first_sentence)+1:temp.index(last_sentence)-1])
        news_text.append(joined_text) #end of loop cycle appends all text from url to the df 
    df["News"] = news_text #append joined text to df 
    df.to_csv(data_path + "/Training-Dataset-" + str(dt.datetime.now())+ ".csv") #save training data to .csv file to be used in future ML training 
    return df  
                
######### OUTPUT #################################
start = time.time()
#Initiate webscraping function 
print("Starting Program....")
columnists = ["David Brooks", "Paul Krugman", "Thomas L. Friedman", "Ezra Klein", "Bret Stephens", "Jamelle Bouie", "Michelle Goldberg", "Farhad Manjoo", "Jennifer Senior", "Maureen Dowd", "Ross Douthat"]
df = scrapeArticles(columnists) 
print("Program Executed in: " + str(time.time()-start) + " seconds\n\n")
print("Program finished, training dataset saved to folder:\n" + data_path ) #saves in same folder as input file 
