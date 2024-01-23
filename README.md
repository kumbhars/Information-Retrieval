
# APPENDIX

### Part-1 Code for Search engine

import requests

from bs4 import BeautifulSoup

import time

from collections import defaultdict

import re

import csv

from urllib.parse import urljoin

import os

import string

import pandas as pd

from urllib.parse import urljoin

from urllib.robotparser import RobotFileParser

import schedule

from sklearn.feature\_extraction.text import TfidfVectorizer

import itertools

from spellchecker import SpellChecker

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word\_tokenize

from nltk.corpus import stopwords

# Downloading necessary resources for pre-processing

nltk.download('stopwords')

nltk.download('punkt')

res=requests.get('https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/')

print(res)

soup=BeautifulSoup(res.text,'html.parser')

#print(soup.prettify)

# Fetch the robots.txt file

rp = RobotFileParser()

root\_url = 'https://pureportal.coventry.ac.uk'

rp.set\_url(urljoin(root\_url, "/robots.txt"))

rp.read()

def is\_allowed(url):

return rp.can\_fetch("\*", url)

def crawl\_persons():

csm\_member\_links=set()

url='https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/persons/'

profile=requests.get(url)

if is\_allowed(url):

profile\_soup=BeautifulSoup(profile.text,'html.parser')

tiles=profile\_soup.select('.result-container')

for profiles in tiles:

profile\_links=profiles.find("a", class\_="link person").get('href')

csm\_member\_links.add(profile\_links)

return csm\_member\_links

def crawl\_author\_pub\_and\_count(pub\_author\_url\_count):

author\_pub\_and\_count = dict()

for key in list(pub\_author\_url\_count.keys()):

authors\_portal\_page = requests.get('https://pureportal.coventry.ac.uk/en/persons/'+key)

author\_soup = BeautifulSoup(authors\_portal\_page.text, "html.parser")

author\_name=author\_soup.find('h1').text

author\_pub\_and\_count[author\_name] = pub\_author\_url\_count[key]

#print(author\_pub\_and\_count)

with open('article\_counts.csv', 'w', newline='') as csvfile:

writer = csv.writer(csvfile)

writer.writerow(['Author Name', 'No. of Articles Published'])

for author, count in author\_pub\_and\_count.items():

writer.writerow([author, count])

def crawler(root, page\_count):

search\_list=[]

visited\_urls = set()

queue = [root] #this is the queue which initially contains the given seed URL

count = 0

csm\_member\_links=crawl\_persons()

while(queue!=[] and count \< page\_count):

url = queue.pop(0)

if is\_allowed(url):

if url in visited\_urls:

continue

visited\_urls.add(url)

print("fetching " + url)

count +=1

page = requests.get(url)

soup = BeautifulSoup(page.text, "html.parser")

tiles=soup.select('.result-container')

c=1

for subsections in tiles:

list\_of\_records={}

members = subsections.find\_all("a", class\_="link person")

is\_member\_csm=False

for member in members:

#print(member.get('href'))

if member.get('href') in csm\_member\_links:

#print('----------')

#print(member.get('href'))

is\_member\_csm=True

break

#print(is\_member\_csm)

#count+=1

#member\_link=members.find('a',class\_="link person").get('href')

#print(member\_link)

if is\_member\_csm:

p\_title=subsections.find('a',{'class':'link'}).get\_text()

p\_link=subsections.find('a',{'class':'link'}).get('href')

p\_date=subsections.find('span',{'class':'date'}).get\_text()

p\_auth\_names=[]

p\_auth\_portals=[]

for auth\_info in subsections.findAll('a',{'class':'link person'}):

p\_auth\_name=auth\_info.get\_text()

p\_auth\_portal\_link=auth\_info.get('href')

p\_auth\_names.append(p\_auth\_name)

p\_auth\_portals.append(p\_auth\_portal\_link)

##for author and their publishing count

p\_auth\_portal\_link=p\_auth\_portal\_link.split('/')[-1]

if p\_auth\_portal\_link in pub\_author\_url\_count:

pub\_author\_url\_count[p\_auth\_portal\_link] += 1

else:

pub\_author\_url\_count[p\_auth\_portal\_link] = 1

list\_of\_records['Name of Publication']=p\_title

list\_of\_records['Publication Link']=p\_link

list\_of\_records['Publication Date']=p\_date

list\_of\_records['List of Authors']=p\_auth\_names

list\_of\_records['Author Pureportal Profile Link']=p\_auth\_portals

search\_list.append(list\_of\_records)

c=c+1

else:

print(f'{url} is not allowed to be crawled')

#print(author\_pub\_and\_count)

for nextpage in soup.findAll('a',attrs={"class": "step"}):

new\_url = nextpage.get('href')

if(new\_url != None and new\_url != '/'):

new\_url = urljoin(url, new\_url)

#print("new\_url is : ", new\_url) #uncomment the print statement to see the urls

queue.append(new\_url)

# Sleep for a few seconds to avoid hitting the server too fast

time.sleep(rp.crawl\_delay('\*'))

headers=['Name of Publication','Publication Link','Publication Date','List of Authors','Author Pureportal Profile Link']

with open('output.csv', mode='w', newline='', encoding='utf-8') as output\_file:

writer = csv.DictWriter(output\_file, fieldnames=headers)

writer.writeheader()

for record in search\_list:

writer.writerow(record)

#print(pub\_author\_url\_count)

crawl\_author\_pub\_and\_count(pub\_author\_url\_count)

def run\_crawler():

crawl\_persons()

print("Crawler running at", time.strftime("%Y-%m-%d %H:%M:%S"))

crawler('https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/',10)

#if it is running on friday

while True:

schedule.run\_pending()

time.sleep(60) # Wait for 1 minute

while True:

x = input('Select the following options\n1. To Run Crawler \n2. Schedule the crawler to run every Friday\n3. Exit\n')

if x == '1':

pub\_author\_url\_count=dict()

crawl\_persons()

crawler('https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/', 10)

print("Crawler running at", time.strftime("%Y-%m-%d %H:%M:%S"))

print('crawling completed')

#crawl\_author\_pub\_and\_count(auth\_dict)

print()

print('2 files generated')

current\_dir = os.path.abspath(os.curdir)

file\_path = os.path.join(current\_dir, 'output.csv')

print("CSM website crawled output File path:", file\_path)

current\_dir = os.path.abspath(os.curdir)

file\_path = os.path.join(current\_dir, 'article\_counts.csv')

print("check Authors and their publication counts File path:", file\_path)

print()

elif x == '2':

schedule.every().friday.at('01:00').do(run\_crawler)

print('crawler scheduled to run every friday at 01:00 AM')

print()

elif x=='3':

break

else:

print('Choose the correct option')

print()

os.path.isfile('output.csv')

output=pd.read\_csv("output.csv")

output.head(2)

processed\_pub\_names = []

def preprocess\_text(pub\_name):

# Removing non-ASCII characters

pub\_name = pub\_name.encode('ascii', 'ignore').decode()

# Removing mentions (starting with '@')

pub\_name = re.sub(r'@\w+', '', pub\_name)

# Converting to lowercase

pub\_name = pub\_name.lower()

# Removing punctuation marks

pub\_name = pub\_name.translate(str.maketrans('', '', string.punctuation))

# Removing stop words

stop\_words = set(stopwords.words('english'))

pub\_name = ' '.join(word for word in pub\_name.split() if word not in stop\_words)

# Stemming words

stemmer = PorterStemmer()

pub\_name = ' '.join(stemmer.stem(word) for word in pub\_name.split())

return pub\_name

# apply the pre-processing function to each publication name and store the results in the processed\_pub\_names list

for pub\_name in output['Name of Publication']:

processed\_pub\_name = preprocess\_text(pub\_name)

processed\_pub\_names.append(processed\_pub\_name)

# create an empty dictionary to hold the inverted index

inverted\_index = {}

# iterate through each publication name and tokenize it

for i, pub\_name in enumerate(processed\_pub\_names):

tokens = pub\_name.split()

# iterate through each token and update the inverted index

for token in tokens:

if token in inverted\_index:

inverted\_index[token].append(i)

else:

inverted\_index[token] = [i]

# print top 5 entries in the inverted index

top\_five = dict(itertools.islice(inverted\_index.items(), 5))

print(top\_five)

tfid = TfidfVectorizer(use\_idf=True, smooth\_idf=True, norm=None, binary=False)

tfid\_matrix = tfid.fit\_transform(processed\_pub\_names)

terms = tfid.get\_feature\_names()

dense\_matrix = tfid\_matrix.toarray()

dense\_list = dense\_matrix.tolist()

# Create a dataframe with the tf-idf values

tf\_idf\_df = pd.DataFrame(dense\_list, columns=terms)

from autocorrect import Speller

spell = Speller(lang='en')

inputquery = input('Enter Publication Name: ')

enteredquery=inputquery

corrected\_query=[]

for word in inputquery.split(' '):

corrected\_word = spell(word)

corrected\_query.append(corrected\_word)

inputquery=' '.join(corrected\_query)

clean\_inputquery = preprocess\_text(inputquery).split()

print(clean\_inputquery)

# Calculate match scores

match\_scores = [sum(tf\_idf\_df.iloc[i][j] for j in clean\_inputquery if j in tf\_idf\_df.columns) for i in range(len(processed\_pub\_names))]

# Check if there are any matches

if all(score == 0 for score in match\_scores):

print("No related research paper found.")

else:

# Sort results by match score and select top 6

top\_results = sorted(enumerate(match\_scores), key=lambda x: x[1], reverse=True)[:6]

# Print results

print('Entered Query:',enteredquery)

print('showing results for',' '.join(clean\_inputquery))

print()

pd.set\_option('display.max\_colwidth', None)

for i, score in top\_results:

row = output.iloc[i]

publication\_name = row['Name of Publication'].strip()

publication\_link = row['Publication Link'].strip()

publication\_date = row['Publication Date'].strip()

authors = row['List of Authors'].strip()

authors\_portal\_links = row['Author Pureportal Profile Link'].strip()

print('Publication Name:', publication\_name)

print('Publication Link:', publication\_link)

print('Publication Date:', publication\_date)

print('Authors:', authors)

print('Author Portal Links:', authors\_portal\_links)

print("\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*")

### Part-2 Code for Document clustering

import feedparser

import random

import nltk

import re

import string

from nltk.tokenize import word\_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette\_score

# Define the URLs for the news feeds

urls = {

"sports": ["https://feeds.bbci.co.uk/sport/rss.xml", "https://www.espn.com/espn/rss/news"],

"technology": ["https://feeds.bbci.co.uk/news/technology/rss.xml", "https://www.wired.com/feed/rss"],

"climate": ["https://feeds.bbci.co.uk/news/science-environment-56837908/rss.xml", "https://theconversation.com/uk/topics/climate-change-27"]

}

# Define the number of clusters

num\_clusters = 3

# Define the preprocessing function

def preprocess\_text(text):

stop\_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

tokens = word\_tokenize(text.lower())

filtered\_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop\_words and token.isalpha()]

return ' '.join(filtered\_tokens)

# Fetch and preprocess the news articles

news\_articles = []

for category, feed\_urls in urls.items():

for url in feed\_urls:

feed = feedparser.parse(url)

for entry in feed.entries:

news\_articles.append((category, preprocess\_text(entry.title)))

print('Total documents collected: ',len(news\_articles))

# Shuffle the news articles

random.shuffle(news\_articles)

# Split the news articles into training and testing sets

train\_size = int(0.8 \* len(news\_articles))

train\_data = [article[1] for article in news\_articles[:train\_size]]

test\_data = [article[1] for article in news\_articles[train\_size:]]

# Vectorize the news articles using TF-IDF

tfidf\_vectorizer = TfidfVectorizer()

tfidf\_X = tfidf\_vectorizer.fit\_transform(train\_data)

# Train a KMeans model on the training data

tfidf\_km = KMeans(n\_clusters=num\_clusters)

tfidf\_km.fit(tfidf\_X)

# Test the model on the testing data

test\_vectors = tfidf\_vectorizer.transform(test\_data)

predicted\_labels = tfidf\_km.predict(test\_vectors)

# Print the performance metrics

print("TF-IDF Performance Metrics:")

print("Silhouette Score:", silhouette\_score(test\_vectors, predicted\_labels))

def classify\_doc(doc, vectorizer, km):

# Preprocess the document

preprocessed\_doc = preprocess\_text(doc)

# Convert the document into a numerical vector using the vectorizer

doc\_vec = vectorizer.transform([preprocessed\_doc])

# Predict the cluster of the document using the KMeans model

cluster = km.predict(doc\_vec)[0]

# Return the predicted label of the document

if cluster == 0:

return "Sports"

elif cluster == 1:

return "Technology"

else:

return "Climate"

while True:

x = input('Select the following options\n1. To Enter Query \n2. Exit\n')

if x=='1':

inputquery=input('Enter Query: ')

out=classify\_doc(inputquery, tfidf\_vectorizer, tfidf\_km)

print('result: ',out)

elif x=='2':

break

else:

print('enter correct option')

print()

