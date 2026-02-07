#!/usr/bin/env python
# coding: utf-8

# ## Business Statement
# 
# Flipkart sells hundreds of TV models across dozens of brands.
# Pricing, specs and features vary wildly, and shoppers drown in noise.
# A data-driven view of the TV market can help buyers, retailers and analysts understand what actually matters.

# ## Problem Statement
# 
# The marketplace lists televisions with inconsistent pricing, duplicate brands, unclear feature advantages and no obvious explanation for why one model costs twice as much as another.
# Without analysis, it’s impossible to compare brands, identify value for money, or spot what drives cost and performance.

# ## Objective
# 
# + Use scraped product data to:
# + Clean, structure and validate TV listings
# + Explore trends across brands, screen sizes, launch years, ratings and sound output
# + Identify relationships between product features and price
# + Compare brands on value and performance metrics
# + Provide insights that explain why TVs are priced the way they are

# In[2]:


import pandas as pd
from selenium import webdriver
import time
import re
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import requests


# In[2]:


url = "https://www.flipkart.com/q/tv-under-100000?page=4"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)


# In[3]:


page = requests.get(url)


# In[4]:


soup = BeautifulSoup(driver.page_source)
soup


# In[5]:


soup.find_all('div',class_='RG5Slk')[0].text.split()[0]


# In[7]:


a = soup.find_all('div',class_='RG5Slk')


# In[8]:


brand = []
for i in a:
    brand.append(i.text.split()[0])
brand


# In[9]:


b = soup.find_all('div',class_ = "hZ3P6w DeU9vF")
b


# In[10]:


price = []
for i in b:
    price.append(i.text)
price    


# In[11]:


c = soup.find_all('div',class_='kRYCnD gxR4EY')
c


# In[12]:


Actual_price = []
for i in c:
    Actual_price.append(i.text)
Actual_price    


# In[13]:


len(Actual_price)


# In[14]:


d =soup.find_all('div',class_= 'MKiFS6')
d


# In[15]:


d = soup.find_all('div', class_="MKiFS6")

rating = []
for i in range(len(price)):  # loop by price length
    try:
        rating.append(d[i].text)
    except:
        rating.append(None)   # or np.nan

rating


# In[16]:


len(rating)


# In[17]:


features = []
e = soup.find_all('div', class_='CMXw7N')
for i in e:
    features.append(i.text)
features    


# In[18]:


year = []
for i in features:
    y = re.findall(r"20\d{2}", str(i))
    year.append(y[0] if y else None)


# In[19]:


year


# In[20]:


len(year)


# In[21]:


ids = []
for i in features:
    x = re.findall(r"ID:\s*([\w\d\-]+)\s*Launch", str(i))
    ids.append(x[0] if x else None)


# In[22]:


ids


# In[23]:


len(ids)


# In[24]:


len(price)


# In[37]:


sound_in_Watts = []
for i in features:
    z = re.findall(r"Output:\s*([\d]+)", str(i))
    sound_in_Watts.append(z[0] if z else None)


# In[38]:


sound_in_Watts


# In[39]:


len(sound_in_Watts)


# In[28]:


cm = soup.find_all('div',class_='RG5Slk')
cm


# In[29]:


cm_list = []
for i in cm:
    x = re.findall(r"(\d+|\d+\.\d+)\s*cm", str(i))
    cm_list.append(x[0] if x else None)


# In[30]:


cm_list


# In[31]:


models = []
for i in features:
    x = re.findall(r"\|\s*([\w ]+?)\s*Model", str(i))
    models.append(x[0] if x else None)


# In[32]:


len(models)


# In[61]:


Brand = []
Cm_list = []
Price = []
Features = []
Year = []
Ids = []
Sound_in_Watts = []
Models = []
Rating = []

for page in range(1, 21):
    url = f"https://www.flipkart.com/q/tv-under-100000?page={page}"
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

   
    a = soup.find_all('div', class_='RG5Slk')
    for x in a:
        text = x.text
        Brand.append(text.split()[0] if text else None)
        cm = re.findall(r"(\d+|\d+\.\d+)\s*cm", text)
        Cm_list.append(cm[0] if cm else None)

    
    b = soup.find_all('div', class_="hZ3P6w DeU9vF")
    for x in b:
        Price.append(x.text)

    
    e = soup.find_all('div', class_='CMXw7N')
    for x in e:
        ftxt = x.text
        Features.append(ftxt)

        y = re.findall(r"20\d{2}", ftxt)
        Year.append(y[0] if y else None)

        fid = re.findall(r"ID:\s*([\w\d\-]+)\s*Launch", ftxt)
        Ids.append(fid[0] if fid else None)

        snd = re.findall(r"Output:\s*([\d]+)", ftxt)
        Sound_in_Watts.append(snd[0] if snd else None)

        mdl = re.findall(r"\|\s*([\w ]+?)\s*Model", ftxt)
        Models.append(mdl[0] if mdl else None)

    
    d = soup.find_all('div', class_="MKiFS6")
    target = max(len(a), len(b), len(e))
    for i in range(target):
        Rating.append(d[i].text if i < len(d) else None)

lists = [Brand, Cm_list, Price, Features, Year, Ids, Sound_in_Watts, Models, Rating]
max_len = max(len(x) for x in lists)

for L in lists:
    L.extend([None] * (max_len - len(L)))


limit = 500
for L in lists:
    del L[limit:]
driver.quit()


# In[63]:


dic = {'ID':Ids,'Brand':Brand,'Model':Models,'Length(CM)':Cm_list,'Launch Year':Year,'Sound(Watts)':Sound_in_Watts,'Price':Price,'Rating':Rating}


# In[64]:


df = pd.DataFrame(dic)


# In[65]:


df


# In[66]:


df.to_csv("flipkart_tvs.csv", index=False)


# # Importing DataSet

# In[4]:


df = pd.read_csv(r"C:\Users\K ROHITH REDDY\Web Scracping\flipkart_tvs.csv")


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# ### Converting Price column to Numeric

# In[9]:


df['Price'] = df['Price'].str.replace(r'₹|,', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')


# In[10]:


df.info()


# # Converting Launch Year, Sound(Watts), Price to int

# In[11]:


df['Launch Year'] = df['Launch Year'].astype('Int64')
df['Sound(Watts)'] = df['Sound(Watts)'].astype('Int64')
df['Price'] = df['Price'].astype('Int64')


# In[12]:


df.info()


# In[13]:


df = df.dropna(how='all')


# In[14]:


df.shape


# In[15]:


df = df.dropna(subset=['Brand', 'Price'])


# In[16]:


df.shape


# In[17]:


df = df.drop_duplicates()


# In[18]:


df.shape


# In[19]:


df.info()


# In[20]:


df.isna().sum()


# In[21]:


df['ID'] = df['ID'].fillna('Unknown')


# In[22]:


df.isna().sum()


# In[23]:


df['Model'] = df['Model'].fillna('Unspecified')


# In[24]:


df.isna().sum()


# In[25]:


df['Length(CM)'] = df['Length(CM)'].fillna(df.groupby('Brand')['Length(CM)'].transform('median'))


# In[26]:


df['Length(CM)'] = df['Length(CM)'].fillna(df['Length(CM)'].median())


# In[27]:


df.isna().sum()


# In[28]:


df['Launch Year'] = df['Launch Year'].fillna(df['Launch Year'].median())


# In[29]:


df.isna().sum()


# In[30]:


df['Rating'] = df['Rating'].fillna(df.groupby('Brand')['Rating'].transform('median'))


# In[31]:


df['Rating'] = df['Rating'].fillna(df['Rating'].median())


# In[32]:


df.isna().sum()


# In[33]:


df['Sound(Watts)'] = df['Sound(Watts)'].fillna(df.groupby('Brand')['Sound(Watts)'].transform('median'))
df['Sound(Watts)'] = df['Sound(Watts)'].fillna(df['Sound(Watts)'].median())


# In[34]:


df.isna().sum()


# In[35]:


df.info()


# In[36]:


df.describe(include='all')


# In[37]:


df.isna().sum()


# In[38]:


df


# In[39]:


df.to_csv("flipkart_tvs_sales.csv", index=False)


# # Visualization

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt


# # Univariant 

# In[41]:


plt.figure(figsize=(12,5))
top10 = df['Brand'].value_counts().head(10)

sns.barplot(x=top10.index, y=top10.values)
plt.title('Top 10 Brands by Count')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[42]:


plt.figure(figsize=(15,12))

# 1 Price Histogram
ax1 = plt.subplot(331)
sns.histplot(df['Price'], kde=False, ax=ax1,color='red')
ax1.set_title('Price Distribution')

# 2 Length KDE
ax2 = plt.subplot(332)
sns.kdeplot(df['Length(CM)'], ax=ax2)
ax2.set_title('Length Distribution')

# 3 Sound Boxplot
ax3 = plt.subplot(333)
sns.boxplot(x=df['Sound(Watts)'], ax=ax3)
ax3.set_title('Sound Spread')

# 4 Launch Year Violin
ax4 = plt.subplot(334)
sns.violinplot(x=df['Launch Year'], ax=ax4)
ax4.set_title('Launch Year Spread')



plt.tight_layout()
plt.show()


# # Bivariant

# ### Relationship between num and num columns

# In[43]:


plt.figure(figsize=(15,12))

# 1 Length vs Price
ax1 = plt.subplot(331)
sns.scatterplot(data=df, x='Length(CM)', y='Price', ax=ax1)
ax1.set_title('Length vs Price')

# 2 Sound vs Price
ax2 = plt.subplot(332)
sns.scatterplot(data=df, x='Sound(Watts)', y='Price', ax=ax2, color='red')
ax2.set_title('Sound vs Price')

# 3 Rating vs Price
ax3 = plt.subplot(333)
sns.scatterplot(data=df, x='Rating', y='Price', ax=ax3, color='green')
ax3.set_title('Rating vs Price')

# 4 Rating vs Sound
ax4 = plt.subplot(334)
sns.scatterplot(data=df, x='Sound(Watts)', y='Rating', ax=ax4, color='orange')
ax4.set_title('Sound vs Rating')

# 5 Length vs Rating
ax5 = plt.subplot(335)
sns.scatterplot(data=df, x='Length(CM)', y='Rating', ax=ax5, color='purple')
ax5.set_title('Length vs Rating')

# 6 Length vs Sound
ax6 = plt.subplot(336)
sns.scatterplot(data=df, x='Length(CM)', y='Sound(Watts)', ax=ax6, color='brown')
ax6.set_title('Length vs Sound')

# 7 Launch Year vs Price
ax7 = plt.subplot(337)
sns.scatterplot(data=df, x='Launch Year', y='Price', ax=ax7, color='blue')
ax7.set_title('Year vs Price')

plt.tight_layout()
plt.show()


# ## Top 10 

# In[47]:


plt.figure(figsize=(10,10))
ax1 = plt.subplot(321)
top_brand = df['Brand'].value_counts().head(10)
sns.barplot(x=top_brand.index, y=top_brand.values, ax=ax1)
ax1.set_title('Top 10 Selling Brands (Model Count)')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2 Top 10 Most Expensive TVs
ax2 = plt.subplot(322)
top_price = df.sort_values('Price', ascending=False).head(5)
ax2.pie(top_price['Price'], labels=top_price['Model'], autopct='%1.1f%%')
ax2.set_title('Top 5 Most Expensive TVs (Pie)')

# 3 Top 10 Highest Rated TVs
ax3 = plt.subplot(323)
top_rating = df.sort_values('Rating', ascending=False).head(10)
sns.barplot(x=top_rating['Model'], y=top_rating['Rating'], ax=ax3)
ax3.set_title('Top 3 Highest Rated TVs')
ax3.set_ylabel('Rating')
ax3.tick_params(axis='x', rotation=45)

# 4 Top 10 Loudest TVs
ax4 = plt.subplot(324)
ax4 = plt.subplot(324)
top_sound = df.sort_values('Sound(Watts)', ascending=False).head(5)
ax4.pie(top_sound['Sound(Watts)'], labels=top_sound['Model'], autopct='%1.1f%%', wedgeprops=dict(width=0.4))
ax4.set_title('Top 5 Loudest TVs (Donut)')

# 5 Top 10 Largest TVs
ax5 = plt.subplot(325)
top_length = df.sort_values('Length(CM)', ascending=False).head(10)
sns.barplot(x=top_length['Model'], y=top_length['Length(CM)'], ax=ax5, color='green')
ax5.set_title('Top 3 Largest TVs')
ax5.set_ylabel('Length (cm)')
ax5.tick_params(axis='x', rotation=45)

# 6 Top 10 Brands by Average Price (bonus)
ax6 = plt.subplot(326)
top_avg = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_avg.index, y=top_avg.values, ax=ax6, color='red')
ax6.set_title('Top 10 Most Expensive Brands (Avg Price)')
ax6.set_ylabel('Average Price')
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# # Multivariant

# In[50]:


#Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df[['Price','Length(CM)','Sound(Watts)']].corr(),annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[49]:


#pairplot
plt.figure(figsize=(2,5))
sns.pairplot(df[['Price','Length(CM)','Sound(Watts)','Rating','Launch Year']],diag_kind='kde')
plt.show()


# # Hypothesis Testing

# ### 1.Does screen size affect price?

# In[183]:


from scipy.stats import pearsonr

print("H0: Screen size and price have NO relationship")
print("H1: Bigger TVs cost more\n")

corr, p = pearsonr(df['Length(CM)'], df['Price'])
print("Correlation:", corr)
print("P-Value:", p)

if p < 0.05:
    print("Decision: Reject H0")
    print("Conclusion: Bigger TVs cost more")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: Screen size does not affect price")


# ### 2.Do bigger TVs also have more speaker power?

# In[184]:


print("\nH0: Screen size and speaker watt have NO relationship")
print("H1: Bigger TVs have higher watt sound\n")

corr, p = pearsonr(df['Length(CM)'], df['Sound(Watts)'])
print("Correlation:", corr)
print("P-Value:", p)

if p < 0.05:
    print("Decision: Reject H0")
    print("Conclusion: Bigger TVs usually have higher watt speakers")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: Size and sound are not linked")


# In[164]:


for c in ['Price','Length(CM)','Sound(Watts)','Rating']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['Price','Length(CM)','Sound(Watts)','Rating'])


# In[185]:


def critical_value_test(stat, tail, pop_mean, samp_mean, n, cl, samp_std=None, pop_std=None):
    stat = stat.lower()
    tail = tail.lower()

    # Print hypotheses
    if tail == 'two':
        print(f"H0: Population mean = {pop_mean}")
        print(f"H1: Population mean ≠ {pop_mean}\n")
    elif tail == 'right':
        print(f"H0: Population mean ≤ {pop_mean}")
        print(f"H1: Population mean  > {pop_mean}\n")
    elif tail == 'left':
        print(f"H0: Population mean ≥ {pop_mean}")
        print(f"H1: Population mean  < {pop_mean}\n")

    # Z test
    if stat == 'z':
        if pop_std is None:
            raise ValueError("Population std required for Z test")

        crit = st.norm.ppf(1 - (1-cl)/2) if tail=='two' else st.norm.ppf(cl)
        z = (samp_mean - pop_mean) / (pop_std/(n ** 0.5))

        print("Z-stat:", z, "Critical:", crit)
        reject = (abs(z) > crit) if tail=='two' else \
                 (z < -crit if tail=='left' else z > crit)

    # T test
    elif stat == 't':
        if samp_std is None:
            raise ValueError("Sample std required for T test")

        dfree = n - 1
        crit = st.t.ppf(1 - (1-cl)/2, dfree) if tail=='two' else st.t.ppf(cl, dfree)
        t = (samp_mean - pop_mean) / (samp_std/(n ** 0.5))

        print("T-stat:", t, "Critical:", crit)
        reject = (abs(t) > crit) if tail=='two' else \
                 (t < -crit if tail=='left' else t > crit)

    print("Decision:", "Reject H0" if reject else "Fail to Reject H0")


# ### Are TVs priced higher than ₹30,000 on average?

# In[186]:


import scipy.stats as st
sample = df['Price']
critical_value_test(
    stat='t',
    tail='right',
    pop_mean=30000,
    samp_mean=sample.mean(),
    n=len(sample),
    cl=0.95,
    samp_std=sample.std())


# ### Is the average screen size 140 cm?

# In[188]:


sample = df['Length(CM)']
critical_value_test(
    stat='t',
    tail='two',
    pop_mean=140,
    samp_mean=sample.mean(),
    n=len(sample),
    cl=0.95,
    samp_std=sample.std())


# ### Do TVs output more than 30 watts on average?

# In[189]:


sample = df['Sound(Watts)']
critical_value_test(
    stat='t',
    tail='right',
    pop_mean=30,
    samp_mean=sample.mean(),
    n=len(sample),
    cl=0.95,
    samp_std=sample.std())


# In[ ]:




