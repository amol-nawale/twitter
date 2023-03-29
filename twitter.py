import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# import re
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk
import streamlit as st
import re
from textblob import TextBlob


nltk.download('vader_lexicon') #required for Sentiment Analysis

st.title('Twitter Sentiment Analysis')
side_bar=st.sidebar
side_bar.header('Input Space')


with st.container():

    now = dt.date.today()
    now = now.strftime('%m-%d-%Y')



    company=side_bar.text_input('Enter The Company Name')
    number=side_bar.number_input('Enter The Number Of Tweets You Want To Analyze')
    day=side_bar.number_input('Enter Number Of Days History From Current Date')
    button=side_bar.button('submit')

    if button:
    
        #Get user input
            query = company

#As long as the query is valid (not empty or equal to '#')...
            if query != "":
                noOfTweet = number
                if noOfTweet != '' :
                    noOfDays = day
                    if noOfDays != '':
                #Creating list to append tweet data
                            tweets_list = []
                            now = dt.date.today()
                            now = now.strftime('%Y-%m-%d')
                            yesterday = dt.date.today() - dt.timedelta(days = int(noOfDays))
                            yesterday = yesterday.strftime('%Y-%m-%d')
                            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' lang:en since:' +  yesterday + ' until:' + now + ' -filter:links -filter:replies').get_items()):
                                if i > int(noOfTweet):
                                    break
                                tweets_list.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username])

                #Creating a dataframe from the tweets list above 
                            df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

                            print(df)
     

with st.container():
    try:
    
        def clean_tweet(tweet):
            if type(tweet) == np.float:
                return ""
            temp = tweet.lower()
            temp = re.sub("'", "", temp) # to avoid removing contractions in english
            temp = re.sub("@[A-Za-z0-9_]+","", temp)
            temp = re.sub("#[A-Za-z0-9_]+","", temp)
            temp = re.sub(r'http\S+', '', temp)
            temp = re.sub('[()!?]', ' ', temp)
            temp = re.sub('\[.*?\]',' ', temp)
            temp = re.sub("[^a-z0-9]"," ", temp)
            temp = temp.split()
    # temp = [w for w in temp if not w in stopwords]
            temp = " ".join(word for word in temp)
            return temp


        df['clean_text']=df['Text'].apply(clean_tweet)
        df.head()

        

        b=[]
        a=df['clean_text']
        for elm in a:
            b.append(TextBlob(elm).sentiment.polarity)
        

        a=pd.DataFrame(a)
        a['polarity']=b
        





        d=[]
        for elm in a['polarity']:
            if elm> 0:
                d.append(1)
            elif elm<0:
                d.append(-1)
            else:
                d.append(0)
        a['sentiment_value']=d


        x=[]
        for elm in a['sentiment_value']:
            if elm==0:
                x.append('neutral') 
            elif elm==-1:
                x.append('negative')
            else:
                x.append('positive')

        a['sentiment']=x


        positive=a['sentiment'].loc[(a['sentiment']=='positive')]
        negative=a['sentiment'].loc[(a['sentiment']=='negative')]
        neutral=a['sentiment'].loc[(a['sentiment']=='neutral')] 

        a=len(positive)
        print(f"total positive sentiments: {a} ")
        b=len(negative)
        print(f"total negative sentiments: {b} ")
        c=len(neutral)
        print(f"total neutral sentiments: {c} ")




        col1,col2=st.columns(2)

        with col1:
            # st.markdown('pie chart')

            labels = ['Positive','Negative','Neutral']
            sizes = [a,b,c]
            colors = ['green', 'red','grey']
            plt.pie(sizes,colors=colors,startangle=90,autopct='%1.0f%%')
            plt.style.use('default')
            plt.legend(labels,loc='upper right')
            plt.title("Sentiment Analysis Result" )
            plt.axis('equal')
            plt.show()

    
            fig, ax = plt.subplots()
            ax.pie(sizes,colors=colors,startangle=90,autopct='%1.0f%%',labels=['positive','negative','neutral'])
            ax.set_title('Pie Chart')
            
            st.pyplot(fig)

        with col2:
            # st.markdown('bar chart')
            fig, ax = plt.subplots()
        # show counts
            z={'positive_count':a,'negative_count':b,'neutral_count':c}
            key=list(z.keys())
            value=list(z.values())
            plt.bar(range(len(z)), value, tick_label=key,color=['green','red','grey'])
            ax.set_title('Bar Chart')

            
            st.pyplot(fig)





           

    
    except Exception as e:
        print(e)


#show counts

try:
   dict={'positive_count':[a],'negative_count':[b],'neutral_count':[c],'Total_count':[a+b+c]}
   st.dataframe(dict)
   st.dataframe(df.head())
   st.dataframe(df.tail())

except Exception as e:
    print(e)
        


    
    
 







    
    





    



    

    
    







    















   

