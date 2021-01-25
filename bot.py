import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from time import sleep
from newspaper import Article
import urllib.request
import requests
from bs4 import BeautifulSoup


def extractText(url):
    #this functions extracts the text that's the article from the link


    #get requests to website
    r1 = requests.get(url)
    page = r1.content

    #create a soup reading the HTML of the page
    soup = BeautifulSoup(page, 'html5lib')

    #specific class is the class name that contains the paragraph
    #websites this doesn't support:
    #latimes.com because it blocks requests and has paywalls
    specific_class = '' #will be used if all of text is in repeated classes w same class name
    parent_class = '' # will be used if all of the text is under a single class. and the elements w the text does not have a class name
    class_type = '' #this will change depending on the class type holding the words. can be p, span, div, etc.
    method_type = 0 #method type 0 is paragraphs have class name. method type 1 is paragraphs don't have class name
    if 'nytimes.com' in url:
        specific_class = 'css-axufdj evys1bk0'
        method_type = 0
        class_type = 'p'
    elif 'medium.com' in url:
        specific_class = 'sm sn pb so b pz sp sq sr qc ss st su sv sw sx sy sz ta tb tc td te tf tg th ou by'
        method_type = 0
        class_type = 'p'
    elif 'foxnews.com' in url:
        parent_class = 'article-body'
        method_type = 1
        class_type = 'div'
    elif 'tmz.com' in url:
        parent_class = 'article__blocks clearfix'
        method_type = 1
        class_type = 'div'
    elif 'buzzfeed.com' in url:
        specific_class = 'js-subbuzz__title-text'
        method_type = 0
        class_type = 'span'
    elif 'cnn.com' in url:
        specific_class = 'zn-body__paragraph'
        method_type = 0
        class_type = 'div'
    elif 'bbc.com' in url:
        parent_class = 'css-83cqas-RichTextContainer e5tfeyi2'
        method_type = 1
        class_type = 'div'
    elif 'economist.com' in url:
        parent_class = 'article__body-text'
        method_type = 1
        class_type = 'p'
    elif 'reuters.com' in url:
        specific_class = 'Paragraph-paragraph-2Bgue ArticleBody-para-TD_9x'
        method_type = 0
        class_type = 'p'
        


    entire_text = ''
    if method_type == 0:
        #assigns all of the class objects with the specific class names
        page_content = soup.find_all(class_type , class_ = specific_class)
        
        #for each paragraph in the page_content, combine them together
        for para in page_content:
            entire_text = entire_text + para.get_text()
    if method_type == 1:
        #this combines all of the text in the children of the parent class (the div is parent and pargraphs r children nodes)
        soup = BeautifulSoup(r1.text, 'html.parser')
        elements = soup.find_all(class_type , parent_class)
        for elem in elements:
            entire_text = entire_text +elem.get_text()


    return entire_text

print(extractText('https://www.reuters.com/article/us-health-coronavirus-australia/australia-approves-pfizer-vaccine-warns-of-limited-global-astrazeneca-supply-idUSKBN29T0T4'))

#PAC model taken from youtube, mainly https://www.youtube.com/watch?v=z_mNVoBcMjM&ab_channel=SATSifaction and then a few other videos


#df is the training csv
df = pd.read_csv('fake-news/train.csv')

#convert the 0 in the labels to real and the 1s to fake for easier readability
conversion_dict = {0: 'Real',1: 'Fake'}
df['label'] = df['label'].replace(conversion_dict)

#makes sure that the training data is relatively balanced
df.label.value_counts()

#trains the model to find the relationship between the text and the label of true or false given
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.8, random_state = 7, shuffle = True)
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df = 0.75)

#converts pandas object into readble strings
vec_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = tfidf_vectorizer.transform(x_test.values.astype('U'))

#passave aggressive classifier basically creates a hyperplane between true or false and then adjusts itself depending on the article and correct itself
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)

#run the model against the given train data that was originally used to make the PAC
y_pred = pac.predict(vec_test)
score=accuracy_score(y_test,y_pred)
print(f'PAC Accuracy: {round(score*100,2)}%')

#this is an elaboration of the previous (currently not printing thought). the values in the array goes as real true, fake true, fake fake, real fake
confusion_matrix(y_test,y_pred, labels=['Real','Fake'])

#gives kfold accuracy
#x=tfidf_vectorizer.transform(df['text'].values.astype('U'))
#scores = cross_val_score(pac,x,df['label'].values,cv = 5)
#print(f'K Fold Accuracy: {round(scores.mean()*100,2)}%')

#reads in more data not related to the test data and sees how well the PAC does
df_true=pd.read_csv('True.csv')
df_true['label']='Real'
df_true_rep=[df_true['text'][i].replace('WASHINGTON (Reuters) - ','').replace('LONDON (Reuters) - ','').replace('(Reuters) - ','') for i in range(len(df_true['text']))]
df_true['text']=df_true_rep
df_fake=pd.read_csv('Fake.csv')
df_fake['label']='Fake'
df_final=pd.concat([df_true,df_fake])
df_final=df_final.drop(['subject','date'], axis=1)

def findlabel(newtext):
    vec_newtest=tfidf_vectorizer.transform([newtext])
    y_pred1=pac.predict(vec_newtest)
    return y_pred1[0]



print(findlabel(extractText('https://www.economist.com/business/2021/01/21/the-secrets-of-successful-listening')))




#gives label 1 if it predicts true, 0 if it predicts false. get the 1s divided by the whole thing
#print(sum([1 if findlabel((df_true['text'][i]))=='Real' else 0 for i in range(len(df_true['text']))])/df_true['text'].size)

#gives label 1 if it predicts false, 0 if it predicts true. get the 1s divided by the whole thing
#print(sum([1 if findlabel((df_fake['text'][i]))=='Fake' else 0 for i in range(len(df_fake['text']))])/df_fake['text'].size)