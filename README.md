APPENDIX 
Part-1 Code for Search engine
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
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from spellchecker import SpellChecker
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
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
root_url = 'https://pureportal.coventry.ac.uk'
rp.set_url(urljoin(root_url, "/robots.txt"))
rp.read()
def is_allowed(url):
    return rp.can_fetch("*", url)

def crawl_persons():
    csm_member_links=set()
    url='https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/persons/'
    profile=requests.get(url)
    if is_allowed(url):
        profile_soup=BeautifulSoup(profile.text,'html.parser')
        tiles=profile_soup.select('.result-container')
        for profiles in tiles:
            profile_links=profiles.find("a", class_="link person").get('href')
            csm_member_links.add(profile_links)
        return csm_member_links

def crawl_author_pub_and_count(pub_author_url_count):
    author_pub_and_count = dict()
    for key in list(pub_author_url_count.keys()):
        authors_portal_page = requests.get('https://pureportal.coventry.ac.uk/en/persons/'+key)
        author_soup = BeautifulSoup(authors_portal_page.text, "html.parser")
        author_name=author_soup.find('h1').text  
        author_pub_and_count[author_name] = pub_author_url_count[key]
    #print(author_pub_and_count)    
    with open('article_counts.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Author Name', 'No. of Articles Published'])
        for author, count in author_pub_and_count.items():
            writer.writerow([author, count])

def crawler(root, page_count):
    search_list=[]
    visited_urls = set()
    queue = [root] #this is the queue which initially contains the given seed URL
    count = 0
    csm_member_links=crawl_persons()
    while(queue!=[] and count < page_count):
        url = queue.pop(0)
        if is_allowed(url):
            if url in visited_urls:
                continue
            visited_urls.add(url)
            print("fetching " + url)
            count +=1
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")

            tiles=soup.select('.result-container')

            c=1
            for subsections in tiles:

                list_of_records={}
                members = subsections.find_all("a", class_="link person")
                is_member_csm=False
                for member in members:
                    #print(member.get('href'))
                    if member.get('href') in csm_member_links:
                        #print('----------')
                        #print(member.get('href'))
                        is_member_csm=True
                        break
                #print(is_member_csm)
                #count+=1
                    #member_link=members.find('a',class_="link person").get('href')
                    #print(member_link)


                if is_member_csm:
                    p_title=subsections.find('a',{'class':'link'}).get_text()
                    p_link=subsections.find('a',{'class':'link'}).get('href')
                    p_date=subsections.find('span',{'class':'date'}).get_text()
                    p_auth_names=[]
                    p_auth_portals=[]


                    for auth_info in subsections.findAll('a',{'class':'link person'}):
                        p_auth_name=auth_info.get_text()
                        p_auth_portal_link=auth_info.get('href')
                        p_auth_names.append(p_auth_name)
                        p_auth_portals.append(p_auth_portal_link)
                        
                        ##for author and their publishing count
                        p_auth_portal_link=p_auth_portal_link.split('/')[-1]
                        if p_auth_portal_link in pub_author_url_count:
                            pub_author_url_count[p_auth_portal_link] += 1
                        else:
                            pub_author_url_count[p_auth_portal_link] = 1
                        
                    list_of_records['Name of Publication']=p_title
                    list_of_records['Publication Link']=p_link
                    list_of_records['Publication Date']=p_date
                    list_of_records['List of Authors']=p_auth_names
                    list_of_records['Author Pureportal Profile Link']=p_auth_portals
                    search_list.append(list_of_records)
                    c=c+1
        
        else:
            print(f'{url} is not allowed to be crawled')

        #print(author_pub_and_count)        
        for nextpage in soup.findAll('a',attrs={"class": "step"}):
            new_url = nextpage.get('href')
            if(new_url != None and new_url != '/'):
                new_url = urljoin(url, new_url)
                #print("new_url is : ", new_url)  #uncomment the print statement to see the urls
                queue.append(new_url)

        # Sleep for a few seconds to avoid hitting the server too fast
        time.sleep(rp.crawl_delay('*'))

        headers=['Name of Publication','Publication Link','Publication Date','List of Authors','Author Pureportal Profile Link']
        with open('output.csv', mode='w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=headers)
            writer.writeheader()
            for record in search_list:
                writer.writerow(record)
    #print(pub_author_url_count)
    crawl_author_pub_and_count(pub_author_url_count)

def run_crawler():
    crawl_persons()
    print("Crawler running at", time.strftime("%Y-%m-%d %H:%M:%S"))
    crawler('https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/',10)
    #if it is running on friday
    while True:
        schedule.run_pending()
        time.sleep(60) # Wait for 1 minute
while True:
    x = input('Select the following options\n1. To Run Crawler \n2. Schedule the crawler to run every Friday\n3. Exit\n')
    if x == '1':
        pub_author_url_count=dict()
        
        crawl_persons()
        crawler('https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/', 10)

        print("Crawler running at", time.strftime("%Y-%m-%d %H:%M:%S"))
        print('crawling completed')
        #crawl_author_pub_and_count(auth_dict)
        print()
        print('2 files generated')
        current_dir = os.path.abspath(os.curdir)
        file_path = os.path.join(current_dir, 'output.csv')
        print("CSM website crawled output File path:", file_path)
        current_dir = os.path.abspath(os.curdir)
        file_path = os.path.join(current_dir, 'article_counts.csv')
        print("check Authors and their publication counts File path:", file_path)
        print()
    elif x == '2':
        schedule.every().friday.at('01:00').do(run_crawler)
        print('crawler scheduled to run every friday at 01:00 AM')
        print()
    elif x=='3':
        break
    else:
        print('Choose the correct option')
        print()
		
os.path.isfile('output.csv')
output=pd.read_csv("output.csv") 
output.head(2)


processed_pub_names = []
def preprocess_text(pub_name):
    # Removing non-ASCII characters
    pub_name = pub_name.encode('ascii', 'ignore').decode()
    # Removing mentions (starting with '@')
    pub_name = re.sub(r'@\w+', '', pub_name)
    # Converting to lowercase
    pub_name = pub_name.lower()
    # Removing punctuation marks
    pub_name = pub_name.translate(str.maketrans('', '', string.punctuation))
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    pub_name = ' '.join(word for word in pub_name.split() if word not in stop_words)
    # Stemming words
    stemmer = PorterStemmer()
    pub_name = ' '.join(stemmer.stem(word) for word in pub_name.split())
    return pub_name

# apply the pre-processing function to each publication name and store the results in the processed_pub_names list
for pub_name in output['Name of Publication']:
    processed_pub_name = preprocess_text(pub_name)
    processed_pub_names.append(processed_pub_name)
    
# create an empty dictionary to hold the inverted index
inverted_index = {}

# iterate through each publication name and tokenize it
for i, pub_name in enumerate(processed_pub_names):
    tokens = pub_name.split()
    # iterate through each token and update the inverted index
    for token in tokens:
        if token in inverted_index:
            inverted_index[token].append(i)
        else:
            inverted_index[token] = [i]

# print top 5 entries in the inverted index
top_five = dict(itertools.islice(inverted_index.items(), 5))

print(top_five)

tfid = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, binary=False)
tfid_matrix = tfid.fit_transform(processed_pub_names)
terms = tfid.get_feature_names()
dense_matrix = tfid_matrix.toarray()
dense_list = dense_matrix.tolist()

# Create a dataframe with the tf-idf values
tf_idf_df = pd.DataFrame(dense_list, columns=terms)

from autocorrect import Speller

spell = Speller(lang='en')

inputquery = input('Enter Publication Name: ')
enteredquery=inputquery
corrected_query=[]
for word in inputquery.split(' '):
    corrected_word = spell(word)
    corrected_query.append(corrected_word)
inputquery=' '.join(corrected_query)

clean_inputquery = preprocess_text(inputquery).split()
    
print(clean_inputquery)

# Calculate match scores
match_scores = [sum(tf_idf_df.iloc[i][j] for j in clean_inputquery if j in tf_idf_df.columns) for i in range(len(processed_pub_names))]

# Check if there are any matches
if all(score == 0 for score in match_scores):
    print("No related research paper found.")
else:
    # Sort results by match score and select top 6
    top_results = sorted(enumerate(match_scores), key=lambda x: x[1], reverse=True)[:6]
    
    # Print results
    print('Entered Query:',enteredquery)
    print('showing results for',' '.join(clean_inputquery))
    print()
    pd.set_option('display.max_colwidth', None)
    for i, score in top_results:
        row = output.iloc[i]
        publication_name = row['Name of Publication'].strip()
        publication_link = row['Publication Link'].strip()
        publication_date = row['Publication Date'].strip()
        authors = row['List of Authors'].strip()
        authors_portal_links = row['Author Pureportal Profile Link'].strip()
        print('Publication Name:', publication_name)
        print('Publication Link:', publication_link)
        print('Publication Date:', publication_date)
        print('Authors:', authors)
        print('Author Portal Links:', authors_portal_links)
        print("***********************************************")
		
Part-2 Code for Document clustering
import feedparser
import random
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the URLs for the news feeds
urls = {
    "sports": ["https://feeds.bbci.co.uk/sport/rss.xml", "https://www.espn.com/espn/rss/news"],
    "technology": ["https://feeds.bbci.co.uk/news/technology/rss.xml", "https://www.wired.com/feed/rss"],
    "climate": ["https://feeds.bbci.co.uk/news/science-environment-56837908/rss.xml", "https://theconversation.com/uk/topics/climate-change-27"]
}
# Define the number of clusters
num_clusters = 3

# Define the preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words and token.isalpha()]
    return ' '.join(filtered_tokens)

# Fetch and preprocess the news articles
news_articles = []
for category, feed_urls in urls.items():
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            news_articles.append((category, preprocess_text(entry.title)))

print('Total documents collected: ',len(news_articles))
# Shuffle the news articles
random.shuffle(news_articles)

# Split the news articles into training and testing sets
train_size = int(0.8 * len(news_articles))
train_data = [article[1] for article in news_articles[:train_size]]
test_data = [article[1] for article in news_articles[train_size:]]

# Vectorize the news articles using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_X = tfidf_vectorizer.fit_transform(train_data)

# Train a KMeans model on the training data
tfidf_km = KMeans(n_clusters=num_clusters)
tfidf_km.fit(tfidf_X)

# Test the model on the testing data
test_vectors = tfidf_vectorizer.transform(test_data)
predicted_labels = tfidf_km.predict(test_vectors)

# Print the performance metrics
print("TF-IDF Performance Metrics:")
print("Silhouette Score:", silhouette_score(test_vectors, predicted_labels))

def classify_doc(doc, vectorizer, km):
    # Preprocess the document
    preprocessed_doc = preprocess_text(doc)
    
    # Convert the document into a numerical vector using the vectorizer
    doc_vec = vectorizer.transform([preprocessed_doc])
    
    # Predict the cluster of the document using the KMeans model
    cluster = km.predict(doc_vec)[0]
    
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
        out=classify_doc(inputquery, tfidf_vectorizer, tfidf_km)
        print('result: ',out)
    elif x=='2':
        break
    else:
        print('enter correct option')
        print()

References

https://files.coventry.aula.education/85a3ae7e148d70bd0132ff5ede9abab4crawler_sample_code_to_start_with.txt
JCharisTech. (2019). Data Science Tools - Spell Checker and Auto Correction with Python[2019]
 https://www.youtube.com/watch?v=rjXeG0aT-7w&t=275s
GeeksforGeeks. (2020). Create Inverted Index for File using Python.
  https://www.geeksforgeeks.org/create-inverted-index-for-file-using-python/


		
		
		
