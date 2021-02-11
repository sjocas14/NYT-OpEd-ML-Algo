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
import time #to check runtime speed 
import datetime as dt
from selenium import webdriver #infinite scroll component 

data_path = 'File/Pathname' #pathname where script is located, where training data will be saved 

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
        print("Scraping data for " + str(name) + ".....")
        page_source = fullScroll(name) #makes sure all articles are visible BEFORE scraping 
        soup = BeautifulSoup(page_source, "lxml") #uses lmxl page source of scrolled page to create the soup
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
    
    df.drop(df[df.News == ''].index, inplace=True) #drop any rows that could not gather news headlines 
    df = df.reset_index(drop=True) #reset index to accomodate dropped rows 
    
    df.to_csv(data_path + "/Training-Dataset-" + str(dt.datetime.now())+ ".csv") #save training data to .csv file to be used in future ML training 
    return df  

def fullScroll(name):
    url = "https://www.nytimes.com/column/"+ name 
    driver = webdriver.Chrome('/Users/samueljocas/Desktop/chromedriver') #activate path for chrome webdriver
    driver.get(url)
    time.sleep(1)  # Time for webpage to load before scrolling (should be increased if computer is slower)
    scroll_pause_time = .25 #time in between scrolls (slower for slower webpages)
    screen_height = (driver.execute_script("return window.screen.height;"))   # get the screen height of the webpage
    
    counter = 0 #both loop counters that determine when scroll if finished set to zero
    max_scroll = 0
    
    while True: #continuously scrolls until loop broken
        driver.execute_script("window.scrollTo(0,{screen_height});".format(screen_height = screen_height)) #scroll height of screen
        time.sleep(scroll_pause_time)
        screen_height += 300 #add 300 to the screen height each loop
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
    
        if scroll_height > max_scroll:  
            max_scroll =  scroll_height #max height set to largest current scroll height 
            counter = 0 #if max_scroll changed, counter reset  
        elif scroll_height == max_scroll: 
            counter+=1  #counter increases each time the scroll height is the same as before
        
        if counter == 15: #wait half a second to load the webpage in case it is being slow to ensure that has actually reached the end 
            time.sleep(1) 
        elif counter == 20: #after 20 continuous loops (5 seconds) without an increase in scroll height, it's likely the end of the webpage is reached. 
            break #breaks while loop
    print("Done scrolling all the way down for " + name)
    
    page_source = driver.page_source #returns page source that has scrolled all the way to the bottom 
    return page_source 
    
                
######### OUTPUT #################################
start = time.time()
#Initiate webscraping function 
print("Starting Program....")
columnists = ["David Brooks", "Paul Krugman"] #add any of the NYT columnists to the list 
df = scrapeArticles(columnists) 
print("************************************************************************")
print("Program Executed in: " + str(time.time()-start) + " seconds\n\n")

print("Program finished, training dataset saved to folder:\n" + data_path ) #saves in same folder as input file 

