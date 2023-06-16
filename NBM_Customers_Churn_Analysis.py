#!/usr/bin/env python
# coding: utf-8

# # National Bank of Malawi (NBM) customers churn Analysis
# 
# Customer churn also known as customer attrition, customer turnover, or customer defection, is the loss of clients or customers. Customer churn rate is a key business metric in any industry and company providing services to end customers, because the cost of retaining an existing customer is far less than acquiring a new one.
# 
# In a highly competitive business landscape, the ever increasing rate of customer churn or attrition is a big challenge in front of Banks today. It takes subtantial effort and investment to lure in new customers, while the danger of losing existing customers to competitors is always lurking. This is why it is very important to analyze and understand the reasons of customer's churn in Banks. The more a Bank is able to retain its existing customer base, the better it will perform in the long run.
# 
# In this notebook, we will try to analyze the factors which contribute to the possibility of customer's churn in National Bank of Malawi (NBM). This analysis is performed based on NBM customers dataset which was extracted from the NBM database on 30th September 2021.

# # Solution for exploratory data analysis
# Our solution approach for exploratory data analysis will be as follow:
# 1. Read data
# 2. Prepare data for Analysis
# - Explore data
# - Pre-process and clean-up data
# - Analyze data (through visualization

# # Section 1: Read input data

# In[1]:


#importing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os  
from textwrap import wrap
import gc 
import glob
from pathlib import Path

# Set default fontsize and colors for graphs
SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 10, 12, 20
plt.rc('font', size=MEDIUM_SIZE)       
plt.rc('axes', titlesize=BIG_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE) 
plt.rc('ytick', labelsize=MEDIUM_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIG_SIZE)
my_colors = 'rgbkymc'

# Disable scrolling for long output
from IPython.display import display, Javascript
disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))


# In[2]:


#Our data is too big to be in one csv file, we have split it into three csv files and load it as below:
# get data file names
path =r'C:\Users\htembani\NBM-customers-churn-analysis'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
df = pd.concat(dfs, ignore_index=True)


# In[3]:


df.head() #Let us see how the top 5 rows of our dataframe look like


# # Section 2: Prepare data for Analysis
# This section will perform the following key tasks:
# 1. Basic statistical analysis of the data
# 2. Explore individual features (by looking at data distribution in data set) and cleanup data where necessary
# 3. Analyse relationship of the target (independent variable) and dependent variables visualisation

# In[4]:


#We have 675655 rows and 40 columns 
df.shape


# In[5]:


#Let us see if we have columns with null or missing values. YES we have null values, let us now see which columns have null 
df.isnull().values.any() 


# In[6]:


#Let us view the columns with null values
df.isnull().sum()


# District has 41 missing values and Nationality has 2, we also have a column 'customerID' which is unique to the customer and cannot be used in our study. Let us remove these three columns from the data set

# In[7]:


#We are going to drop all columns with null values and all columns that are not useful in our study
df = df.drop(['CustomerID','District', 'Nationality' ],axis=1)
              


# In[8]:


#creating a new column called churn value taking the value of churn as YES = 1 AND NO =0, we will need this during visualisation
churn_mapping = {"NO": 0, "YES": 1}
df['Churn_value'] = df['Churned'].map(churn_mapping)


# In[9]:


#Checking unique values in the newly created column
df.Churn_value.unique()


# In[10]:


#Let us see if we have achieved what we intended to do with the newly created column
selection = df.loc[:2,['Churned', 'Churn_value']]
print(selection)


# In[11]:


#Let us confirm that the columns have successfully been droped and that our dataset has no null values
df.isnull().sum()/df.shape[0]


# In[12]:


# Let us see the total number of records in the dataset after removing the null values
print("Total number of records in NBM Dataset:", df.shape)


# In[13]:


#Let us view the structure and details of our data frame having done some changes.
df.describe()


# In[14]:


#Let ue go into details to see the features available and their data types
df.info() 


# # what are our observations this far
# 1. There are 675,655 records in the dataset. Meaning that NBM had 675,655 customers as at 30th September 2021
# 2. There are 36 features in the dataset - "Churned" is the target/independent variable (this is what we will try to predict), and rest 35 are dependent variables which we need to explore further.
# 3. There are 19 features that have numeric data type and 19 have object data type .
# 4. All missing values have been removed and we have data without missing values
# 
# # LET US START EXPLORING THE DATASET

# In[15]:


#Let us see percentage of churn in NBM and in each gender category 
fig, axes2 = plt.subplots(figsize=(12,8))

# Pie chart of churn percentage
width = 0.5

# Percentage of Churned vs Retained
data = df.Churned.value_counts().sort_index()
axes2.pie(
    data,
    labels=['Retained', 'Churned'],
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=90,
    textprops={'color':'black', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1,
)

# Percentage of Gender based on Churn
data = df.groupby(["Churned", "Gender"]).size().reset_index()
axes2.pie(
    data.iloc[:,2], 
    labels=list(data.Gender),
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1-width,
    rotatelabels=True,
)

axes2.set_title('What is the ratio of the customer\'s gender and churn?')
#axes2.legend(loc='best', bbox_to_anchor=(1,1))
axes2.axis('equal')

plt.show()


# We see that 55.2% customers were retained out of which 19.1% are female and 29.6% Male while 6.4% Other (non individual). On the other side we see that 44.8% customers churned, 11.9% female, 23.6% male and 9.3% Other (Non individual). Looking at distribution and margin of churn, we can establish that gender does not determin churn. 

# In[16]:


#We want to establish if being married or not determin churn in NBM
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(18,6))
# Plot distribution of marital status data
data = df["MaritalStatus"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Marital Status data %')
axes1.set_xlabel('Living with partner?')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on marital status
sns.barplot(x="MaritalStatus", y=df.Churn_value*100, data=df, ci=None, ax=axes2)
axes2.set_xlabel('Living with partner?')
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on presence of a partner?', 30)))
axes2.set_ylim(0,100)
plt.show()


# The percentage of married and unmarried is almost the same in our dataset. Churn among the married and unmarried is also almost the same. This means that being married or single does not determin churn as these two groups have same rate of churn. 
# 
# Churn is very higher in a group whose marital status is not known as well as widowed and those that have a partner.

# In[17]:


#Here we want to see the relationship between employment status and churn.
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of employment status data
data = df["EmploymentStatus"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of customer Employment status data %')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on employment status
sns.barplot(x="EmploymentStatus", y=df.Churn_value*100, data=df, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on employment status?', 30)))
axes2.set_ylim(0,100)

plt.show()


# The largest number of customers are the ones whose employment status is not known and churn is very high in this group. The lowest number of customers is named 'OTHER' which is largely of non individual and churn is also high in this group.

# In[18]:


#Let us see Age distribution in our data set
plt.figure(figsize=(15, 5))
sns.distplot(df.Age, kde=True, label='Age of customer or Company registration date', hist_kws={"histtype": "step", "linewidth": 3,
                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})

plt.title('Age Distribution plot', fontsize=20)      
plt.show()


# This is not a normal Age distribution, There is a large number of customers Aged 26 and 36.

# In[19]:


#Let us see churn in each Age group
fig, ax=plt.subplots(figsize=(20,5))
sns.countplot(data = df, x='Age', order=df['Age'].value_counts().index, palette='viridis', hue='Churned')
plt.xticks(rotation=90)
plt.xlabel('Age', fontsize=10, fontweight='bold')
plt.ylabel('Customers', fontsize=10, fontweight='bold')
plt.title('Age distribution and churn by Age', fontsize=12, fontweight='bold')
plt.show()


# As seen already, most customers are between 20 and 50 years. There is a large number of customers aged 22 and 36 and these are the only age groups with churn surpasing customers retained.
# 
# Churn is high in the youthful Ages as compared to the other Age group.

# In[20]:


#Let us see if tenure determin churn
fig, ax=plt.subplots(figsize=(20,5))
sns.countplot(data = df, x='Tenure', order=df['Tenure'].value_counts().index, palette='viridis', hue='Churned')
plt.xticks(rotation=90)
plt.xlabel('Tenure', fontsize=10, fontweight='bold')
plt.ylabel('Customers', fontsize=10, fontweight='bold')
plt.title('Tenure distribution and churn by Tenure', fontsize=12, fontweight='bold')
plt.show()


# The largest number of customers are short lived (less than 3 Years) and customers who stay with the Bank for over 5 years are less likely to quit. 
# 
# It is surprising to see high churn in customers who are less than one year cosidering that the requlatory requirement is that a customer is diactivated after two years of inactivity. NBM rarely diactivate customers that are less than one year. We need to explore what kind of customers could these be. 

# We now confirm that new customers churn more and customers retained for over 5 years are royal to the brand
# 
# On the other side we see that customers with no access to e services are more likely to churn than those with access to e services.

# In[21]:


#Let us look at churn distribution by geographical region
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of Region data
data = df["Region"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of customer Region data %')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on geographical region
sns.barplot(x="Region", y=df.Churn_value*100, data=df, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on Geographical Region?', 30)))
axes2.set_ylim(0,100)

plt.show()


# There is high churn rate in the southern region, the number of customers that churned in this region is almost the same as the number of customers retained. Central region has the lowest churn rate, we need to explore further.

# In[22]:


#Show churn per service centre.
fig, ax=plt.subplots(figsize=(20,5))
sns.countplot(data = df, x='ServiceCentre', order=df['ServiceCentre'].value_counts().index, palette='viridis', hue='Churned')
plt.xticks(rotation=90)
plt.xlabel('Service Centre', fontsize=10, fontweight='bold')
plt.ylabel('Customers', fontsize=10, fontweight='bold')
plt.title('Distribution of customers and churn per Service centre', fontsize=12, fontweight='bold')
plt.show()


# We mentioned of high churn in the southern region and also we mentioned of churn in customers who are less than one year with the Bank.
# 
# Now we see that Top Mandala Service Centre has the highest churn. NBM acquired INDE Bank in 2016 and all customers acquired from INDE Bank were domiciled in Top Mandala Service Centre. The high churn in Top Mandala tells that the largest number of customers acquired from INDE Bank churned and most of them churned within a year.

# In[23]:


#We want to see which customer target churn more
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of Target data
data = df["Target"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of customer target data %')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on target
sns.barplot(x="Target", y=df.Churn_value*100, data=df, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on target?', 30)))
axes2.set_ylim(0,100)

plt.show()


# We see that the largest number of customers are retail ordinary and churn is 43% in this group of customers. Now look at corporate, NGOs, Parastatals and Government, we see that churn is 80% or above. These customers bring more revenue to the Bank as such it is very important that we come to the bottom of why many of them churn.

# In[24]:


#Let us see if Bank charges determin churn
diag = px.pie(df, values='CumlativeCharges', names='Churned', hole=0.5)
diag.show()


# We see that among customers that paid fees and other commissions, only 3.37% churned as such we can conclude that fees and commissions charged to customers do not determin churn in National Bank of Malawi. 

# In[25]:


#Let us see if access to eServices determine churn in the Bank

#We want to establish if Access to e services determin churn in NBM
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(18,6))
# Plot distribution of marital status data
data = df["Access_to_eServices"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Access to e services data %')
axes1.set_xlabel('Has access to e Services?')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on marital status
sns.barplot(x="Access_to_eServices", y=df.Churn_value*100, data=df, ci=None, ax=axes2)
axes2.set_xlabel('Did the customer churn?')
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on Access to e Services?', 30)))
axes2.set_ylim(0,100)
plt.show()


# This is very interesting, see that 58% of customers had access to eServices and out of them 10% churned. 42% did not have access to e services and out of them 85% churned. 
# 
# We conclude that access to eServices (Mo626, ATM/POS and Banknet services) is a key factors determining churne in NBM. This observation agree with our finding that churn is high in non individual customers as non individual customers have limited access to eServices hence high churn.

# # Different Service Types
# 
# We are going to analyse the following services offered by NBM: Receive_SMS_Alerts, DebitCardServices, InternetBanking, MobileBanking, CounterCashDeposits, InternetBankingUsage, CounterCashWithdrawal, Counter.ChequeWithdrawal, NBM.ATM_Usage
# 
# # Here we are plotting 2 diagrams for each service type:
# 
# on left - we show the distribution of data in the dataset
# on right - we look at the percentage of churn in each category of service types

# In[26]:


Services = ['Receive_SMS_Alerts', 'DebitCardServices', 'InternetBankingRegistration', 'MobileBankingRegistration', 'CounterCashDeposits_Services', 
            'CounterCashWithdrawal_Services', 'Counter.ChequeWithdrawal.Services', 'NBM.ATM_Usage', 'MobileBankingUsage','InternetBankingUsage']
n_cols = 2
n_rows = len(Services)
fig = plt.figure(figsize=(15,40))
#fig.suptitle('Distribution of Service Types and relation with Churn')
idx = 0

for serviceType in enumerate(Services):
    # Fetch data of Service Type
    data = df[serviceType[1]].value_counts(normalize=True).sort_index()

    # Now, plot the data
    i = 0
    for i in range(n_cols):
        idx+=1
        axes = fig.add_subplot(n_rows, n_cols, idx)

        # On column 1 - Plot the data distribution on bar plot
        if idx%2 != 0:
            axes.bar(data.index, data*100, color=my_colors)
        # On column 2 - Plot the percentage of churns on each service type
        else:
            sns.barplot(x=serviceType[1], y=df.Churn_value*100, data=df, ci=None, ax=axes)

        if idx == 1 : axes.set_title('Distribution of service category data')
        if idx == 2 : axes.set_title('% of churn in each service category')
            
        axes.set_xlabel(serviceType[1])
        axes.set_ylabel('')
        axes.set_ylim(0,100)

fig.tight_layout()
plt.show()


# # Observations
# (1).SMS Alerts: 78% receive SMS alerts and 36% of those that receive alerts churned, this 36% churn is still higher. On the other side 22% do not receive Alerts and 80% of them churned, that is too much as such giving access of SMS alerts to as many customers as possible can help to ease churn.
# 
# (2).Debit Card: 76% had a debit card and 38% of them churnes. 24% did not have debit cards and out of them 70% churned, that is customers without debit cards are two times likely to churn than those with a debit card.
# 
# (3).Banknet Services: 98% have no access to Banknet Services and 44% of them churned. 2% are the ones with access to Banknet and over half of them churned. We have to establish which customers are these.
# 
# (4).Mobile Banking registration: 59% are registered on Mobile Banking and 18% of them churned. 41% are not registered on Mobile Banking and 80% of those not registered on Mobile Banking churned. Mobile Banking is a key factor determining churn in individual customers.
# 
# (5).Over the counter cash deposits: 60% deposited cash on the counter and 18% of them churned, 40% did not deposit on the counter and 87% of them churned. Availability of points of cash deposits close to customer's location can not be overlooked.  
# 
# (6).Over the counter cash withdrawal: 50% did counter withdrawals and 8% of them churned.The other 50% did not withdrawal on the counter and 78% of them churned. Availability of points of easy access to cash withdrawals is also key.  
# 
# Observation 5 and 6 tells us that over the counter services are very crucial, where customers find it hard to access counter services, churn is likely to be higher.
# 
# 7. NBM ATM Usage:59% did not use NBM ATMs and close to 70% of them churned. 49% used NBM ATMs and 4% of them churned, ATM usage is another key factor determining churn.
# 
# 

# In[27]:


#Let us see how our data is correlated using correlation matrix
correlation_mat = df.corr()
greater_than= correlation_mat[correlation_mat > 0.5]
plt.show()
plt.figure(figsize=(12,12))
sns.heatmap(greater_than,annot=True )
plt.show()


#  Both the value and volume of ATM usage has a high correlation with number of transactiona on ATMs and amount withdrawn on ATMs. This means that customers who frequently use ATMs churn less than those who do not use ATMs frequently.  

# # Section 3: Data Preprocessing 

# In[28]:


#Importhin algorithims to be used
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[29]:


df = df.drop('Churn_value',axis=1) #This column is nolonger necessarly


# In[30]:


#We want to see all features and their categorical variables
for i in df.columns:
    if df[i].dtypes=="object":
        print(f'{i} : {df[i].unique()}')
        print("****************************************************")


# There are some features in which categorical variables are more than two and they are not 'YES' or 'NO' type. We need to identify these and deal with them accordingly.

# In[31]:



for i in df.columns:
    if (len(df[i].unique()) >2) & (df[i].dtypes != "int64") &(df[i].dtypes!= "float64"):
        print(i)


# We see that Target, ServiceCentre,Region,Gender Marital Status and Employement status have more than two categorical values

# In[32]:


# Let us start by dealing with Gender by replace Male with 0, Female by 1, and Other by 2

df['Gender'].replace({'FEMALE':1,'MALE':0,'Other':2},inplace=True)
print(df['Gender'].value_counts(ascending=True))


# Next we are going to view the contents of the remaining 5 categorical variables, this will help us to decide how to deal with each of them 

# In[33]:


#Let us view contents of categorical variable Target
print(df['Target'].value_counts(ascending=True))


# In[34]:


#Let us view contents of categorical variable Service Centre
print(df['ServiceCentre'].value_counts(ascending=True))


# In[35]:


#Let us view contents of categorical variable Region
print(df['Region'].value_counts(ascending=True))


# In[36]:


#Let us view contents of categorical variable Marital Status
print(df['MaritalStatus'].value_counts(ascending=True))


# In[37]:


#Let us view contents of categorical variable Employment Status
print(df['EmploymentStatus'].value_counts(ascending=True))


# In[38]:


#Having view the contents, we come to a conclusion that the best is to deal with these feature columns using one-hot encoding
more_than_2 = ['Target' ,'ServiceCentre' ,'Region','MaritalStatus','EmploymentStatus']
df = pd.get_dummies(data=df, columns= more_than_2)
df.dtypes


# In[39]:


#Let us see our new dataframe after applying one-hot ecoding
df.shape


# In[40]:


#Let us see the columns of our data set after one-hot ecoding
df.columns


# In[41]:


#It is now time to deal with columns with numeric data
for i in df.columns:
    if (df[i].dtypes == "int64")  | (df[i].dtypes== "float64"):
        print(i)


# The above feature column have numerical data so we will need to bring it into a particular range if they varies a lot. We are going to use MinMaxScaler of Feature Scaling

# In[42]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[43]:


#Applying MinMaxScaler
large_cols = ["CounterWithdrawalValue", "CounterWithdrawalVolume", "ChequeWithdrawalValue","ChequeWithdrawalVolume",' CounterCashDepositValue ',"CounterCashDepositVolume","Mo626_Values", "Mo626_Volume","NBM_ATM_Values","NBM_ATM_volumes","Other_Banks_ATM_volume","Other_Banks_ATM_fee","Banknet_Values","Banknet_Volumes","CumlativeCharges","TimesFeePaid","Age","Tenure"]
df[large_cols] = scaler.fit_transform(df[large_cols])
df[large_cols].head()


# In[44]:


#Having finished feature scaling we have following dataset
df.head()


# In[45]:


#We have not worked on columns with object data type particularly those with "YES" and "NO", let us identify them
for i in df.columns:
    if (df[i].dtypes == "object"):
        print(i)


# In[46]:


# Churned is our independent variable, let us handle it separately 

df['Churned'].replace({'YES':1,'NO':0},inplace=True)
print(df['Churned'].value_counts(ascending=True))


# In[47]:


#Let us deal with the other remaining variables
two_cate = ['Receive_SMS_Alerts', 'DebitCardServices', 'Access_to_eServices', 'InternetBankingUsage', 'InternetBankingRegistration', 'MobileBankingRegistration', 'MobileBankingUsage', 'NBM.ATM_Usage', 'Other.Banks.ATM_Usage', 'CounterCashDeposits_Services', 'CounterCashWithdrawal_Services','Counter.ChequeWithdrawal.Services']
for i in two_cate:
    df[i].replace({"NO":0, "YES":1}, inplace=True)
df.head() #let us see the output after processing the data


# In[48]:


# It time to Split our Dataset into dependent and independent variables

X = df.drop('Churned', axis=1) #independent variable
y = df['Churned'] # dependent variable


# In[49]:


X.shape, y.shape


# In[50]:


# Performing Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[51]:


y_train.value_counts(normalize=True),y_test.value_counts(normalize=True)


# # Section 4. Model Building

# ## 1. Using Logistics Regression

# In[52]:


# Importing Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[53]:


# creating object for model
model_lg = LogisticRegression(max_iter=120,random_state=0, n_jobs=20)


# In[54]:


# Model Training

model_lg.fit(X_train, y_train)


# In[55]:


# Making Predictions
pred_lg = model_lg.predict(X_test)


# In[56]:


# Calculating Accuracy of the model

lg = round(accuracy_score(y_test, pred_lg)*100,2)
print(lg)


# In[57]:


# Classification Report

print(classification_report(y_test, pred_lg))


# In[58]:


# confusion Matrix

cm1 = confusion_matrix(y_test, pred_lg)
sns.heatmap(cm1/np.sum(cm1), annot=True, fmt='0.2%', cmap="Reds")
plt.title("Logistic Regression Confusion Matrix",fontsize=12)
plt.show()


# ## 2. Using decision Tree Classifer

# In[59]:


from sklearn.tree import DecisionTreeClassifier


# In[60]:


# Creating object of the model
model_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
model_dt.fit(X_train, y_train)


# In[61]:


pred_dt = model_dt.predict(X_test)


# In[62]:


dt  = round(accuracy_score(y_test, pred_dt)*100, 2)
print(dt)


# In[63]:


print(classification_report(y_test, pred_dt))


# In[64]:


# confusion Maxtrix
cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot = True, fmt=  '0.2%', cmap = 'Reds')
plt.title("Decision Tree Classifier Confusion Matrix",fontsize=12)
plt.show()


# ## 3.Using Random Forest

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


# Creating model object
model_rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.16, random_state=42)


# In[67]:


# Training Model
model_rf.fit(X_train, y_train)


# In[68]:


# Making Prediction
pred_rf = model_rf.predict(X_test)


# In[69]:


# Calculating Accuracy Score
rf = round(accuracy_score(y_test, pred_rf)*100, 2)
print(rf)


# In[70]:


print(classification_report(y_test,pred_rf))


# In[71]:


# confusion Maxtrix
cm3 = confusion_matrix(y_test, pred_rf)
sns.heatmap(cm3/np.sum(cm3), annot = True, fmt=  '0.2%', cmap = 'Reds')
plt.title("RandomForest Classifier Confusion Matrix",fontsize=12)
plt.show()


# ## 4. Using XGBoost Classifier

# In[72]:


from xgboost import XGBClassifier


# In[73]:


# Creating model object
model_xgb = XGBClassifier(max_depth= 8, n_estimators= 125, random_state= 0,  learning_rate= 0.03, n_jobs=5)


# In[74]:


# Training Model
model_xgb.fit(X_train, y_train)


# In[75]:


# Making Prediction
pred_xgb = model_xgb.predict(X_test)


# In[76]:


# Calculating Accuracy Score
xgb = round(accuracy_score(y_test, pred_xgb)*100, 2)
print(xgb)


# In[77]:


print(classification_report(y_test,pred_xgb))


# In[78]:


# confusion Maxtrix
cm4 = confusion_matrix(y_test, pred_xgb)
sns.heatmap(cm4/np.sum(cm4), annot = True, fmt=  '0.2%', cmap = 'Reds')
plt.title("XGBoost Classifier Confusion Matrix",fontsize=12)
plt.show()


# # 5. Using AdaBoost Classifier

# In[79]:


from sklearn.ensemble import AdaBoostClassifier


# In[80]:


model_ada = AdaBoostClassifier(learning_rate= 0.002,n_estimators= 205,random_state=42)


# In[81]:


model_ada.fit(X_train, y_train)


# In[82]:


# Making Prediction
pred_ada = model_ada.predict(X_test)


# In[83]:


# Calculating Accuracy Score
ada = round(accuracy_score(y_test, pred_ada)*100, 2)
print(ada)


# In[84]:


print(classification_report(y_test,pred_ada))


# In[85]:


# confusion Maxtrix
cm7 = confusion_matrix(y_test, pred_ada)
sns.heatmap(cm7/np.sum(cm7), annot = True, fmt=  '0.2%', cmap = 'Reds')
plt.title("Adaboost Classifier Confusion Matrix",fontsize=12)
plt.show()


# In[86]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'AdaBoost'],
    'Accuracy_score' :[lg, dt, rf, xgb, ada]
})
models
sns.barplot(x='Accuracy_score', y='Model', data=models)

models.sort_values(by='Accuracy_score', ascending=False)

