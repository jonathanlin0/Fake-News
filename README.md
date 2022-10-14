# Fake News Classifier Twitter Bot
### By: Jonathan Lin

## About
This project was my first project involving machine learning and Twitter's API. I used a Passive Aggressive Classifier, utilizing over 25,000 sample news articles to create the model. I then used the tweepy wrapper for the Twitter Developer API to integrate it.

## Usage
This program is designed so that when you mention the bot, such as doing "@fakenewschecker" (the handle you mention will depend on your bot's Twitter handle), and then also type the link, within 10 seconds the bot will reply with a conclusion of whether the model determines the article is real or fake. I will upload a list soon of supported news websites and unsupported websites.

## Running
Go the keys.txt file and change the keys according to what the text file says. For example, the first line is CONSUMER_KEY, second is CONSUMER_SECRET, etc. Do not put spaces or anything else on the lines except for your respective token/key/secret text.\
<br/>

This program was coded in Python 3.8, so you will have to do the following commands to install the packages:\
`pip install tweepy`\
`pip install requests`\
`pip install pandas`\
`pip install sklearn`\
`pip install urllib3`\
`pip install beautifulsoup4`\
\
Note: you may have to do `pip3 install` instead of `pip install` for installing packages depending on your OS, comptuer, etc
