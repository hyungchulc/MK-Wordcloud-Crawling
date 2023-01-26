#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pytimekr
#pip install wordcloud
#pip install nltk
#pip install konlpy


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import urllib.parse
import re
import requests as req
import datetime
import time
from bs4 import BeautifulSoup as bs 
import matplotlib as mpl
from PIL import Image as MImage
from IPython.display import Image
from matplotlib import rc
from matplotlib import font_manager
from tqdm import tqdm # for문 진행상황 눈으로 확인 (loading bar)
from datetime import date
from pytimekr import pytimekr
from wordcloud import WordCloud,ImageColorGenerator
from collections import Counter

plt.rc('font', family='NanumGothicBold')

mpl.rc('axes', unicode_minus=False)


# In[3]:


# 명사 추출
import nltk
#nltk.download('averaged_perceptron_tagger')

# 영어 불용어(stopwords) 사전
#nltk.download('stopwords')
from nltk.corpus import stopwords    
stopwords = set(stopwords.words('English')) 

# 단어/문장 단위로 토큰화
#nltk.download('punkt')
from nltk.tokenize import word_tokenize,sent_tokenize


# In[4]:


from konlpy.tag import Kkma
kkma = Kkma() #Java download needed
from konlpy.tag import Okt


# In[5]:


day = datetime.date.today() 
today = day.strftime("%Y/%m/%d")
today_url = 'https://www.mk.co.kr/today-paper/01/'+today+'/'
    
print(day)
    


# In[6]:


url = today_url
response=urllib.request.urlopen(url)
soup=bs(response,'html.parser')


# In[7]:


html = soup.select('dd.news_tt.first > a') # 원하는 부분의 CSS 선택자 지정
html[2] # 리스트의 3번째 요소 내용 확인


# In[8]:


title = [x.text for x in html]
print(title)


# In[9]:


len(title)


# In[10]:


a=[]
for i in range(len(title)):
    text = soup.select('dd.news_tt.first > a')[i].text
    a.append(text)
print(a)


# In[11]:


a


# In[12]:


kws = pd.DataFrame(a)
kws.to_csv(str(day)+' ''매경head.csv',header = ['제목'])


# In[13]:


b=str(a)


# In[14]:


print(type(b))


# In[15]:


b


# In[16]:


b_filtered = b.replace('\n','').replace('\u200b','')
b_filtered


# In[17]:


pattern = r'\[.*?\]'
re.sub(pattern, '', b_filtered)


# In[18]:


b_filtered = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', b_filtered)
b_filtered


# In[19]:


b_filtered = re.sub('\d',' ',b_filtered)
b_filtered


# In[20]:


okt = Okt()  
okt.morphs(b_filtered)


# In[21]:


#b_filtered = re.sub('[''[포토]'',[표],[매경e신문],[영상]]',' ',b_filtered)
#b_filtered


# In[22]:


okt.pos(b_filtered)


# In[23]:


c=okt.nouns(b_filtered)
c


# In[24]:


d = [word for word in c if len(word)>1]
d


# In[25]:


list(stopwords)[:10] # 집합 자료형으로 리스트로 변환해 슬라이싱


# In[26]:


result = d


# In[27]:


from collections import Counter
word_freq = Counter(d)
word_freq.most_common()


# In[28]:


stopwords2 = ['지난해','달린다','매일경제','목소리','오늘','이제','년새','억원','억이','밸리','이름','달라','천만원','이후','이슈',
             '기업','사설','매경','단독','증시','뉴스','레이더','주요','기자','논의','포토','우리','이번','개월','사태','문제','회장',
             '꼽아보','필동정담','진짜',]
result = [w for w in result if w not in stopwords2]
result


# In[29]:


word_freq = Counter(result)
word_freq.most_common()
AT=word_freq.most_common()


# In[30]:


print(AT)


# In[31]:


DF = pd.DataFrame(AT)
DF.shape


# In[32]:


DF.to_csv(str(day)+' ''매경.csv',header = ['단어','횟수'])


# In[33]:


words = ' '.join(result)


# In[34]:


#mask = np.array(MImage.open(""))
#mask = mask[:,:,0]


# In[35]:


wc = WordCloud(background_color="white")
wc = wc.generate(words)


# In[36]:


wc.words_


# In[38]:


wordcloud = WordCloud(
    font_path = '/Library/Fonts/AppleGothic.ttf',    # 맥에선 한글폰트 설정 잘해야함.
    background_color='white',                            # 배경 색깔 정하기
    #mask=mask
    colormap = 'Accent_r',                               # 폰트 색깔 정하기
    width = 800,
    height = 800
)

wordcloud_words = wordcloud.generate_from_frequencies(wc.words_)


# In[41]:


array = wordcloud.to_array()
print(type(array)) # numpy.ndarray
print(array.shape) # (800, 800, 3)


fig = plt.figure(figsize=(10,10))

#def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    #return("hsl(0,0%%, %d%%)" % np.random.randint(1,100))

plt.imshow(
    array,
    #wordcloud_words.recolor(color_func = grey_color_func),
    interpolation="bilinear")
plt.axis('off')
plt.show()
fig.savefig('/Users/dtive/Desktop/Python/매경워드클라우드/일별/'+str(day)+' ''매경워드클라우드.png')


# In[ ]:




