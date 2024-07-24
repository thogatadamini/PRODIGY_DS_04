import matplotlib
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Additional necessary imports
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
(nltk.download('wordnet'))
data = pd.read_csv('twitter_training.csv')
v_data = pd.read_csv('twitter_validation.csv')
data
v_data
data.columns = ['id','game', 'sentiment', 'text']
v_data.columns = ['id','game', 'sentiment', 'text']
data
v_data
data.shape
data.columns
data.describe(include='all')
id_types= data['id'].value_counts()
id_types
plt.figure(figsize=(10, 7))
sns.barplot(y=id_types.index, x=id_types.values)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Type vs Count')
plt.show()
game_types= data['game'].value_counts()
game_types
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'game_types' is a Series with index and values
# Replace 'game_types' with your actual data

# Example data for demonstration
game_types = pd.Series([50, 30, 20], index=['Action', 'Adventure', 'Puzzle'])

plt.figure(figsize=(14, 10))
sns.barplot(y=game_types.index, x=game_types.values, palette='viridis')
plt.title('Types of Games and their Count')
plt.ylabel('Type')
plt.xlabel('Count')
plt.show()
import seaborn as sns

# Assuming 'data' is your DataFrame with columns 'game' and 'sentiment'
# Replace 'data' with your actual DataFrame name

# Example data for demonstration
# Replace with your actual data
data = pd.DataFrame({
    'game': ['Game1', 'Game2', 'Game1', 'Game3', 'Game2'],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'positive']
})

# Create a count plot using sns.catplot()
sns.catplot(x='game', hue='sentiment', kind='count', data=data)
plt.title('Sentiment Counts by Game')
plt.xlabel('Game')
plt.ylabel('Count')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'game': ['Game1', 'Game2', None, 'Game3', 'Game2'],
    'sentiment': ['positive', 'negative', 'neutral', None, 'positive']
})

sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Values in Data')
missing_data = data.isnull().sum().sort_values(ascending=False)
percent = (missing_data / data.shape[0]) * 100

print(f"Total records = {data.shape[0]}")
missing_data = pd.DataFrame({'Total Missing': missing_data, 'In Percent': percent})
print(missing_data.head(10))
train0 = data[data['sentiment'] == 'Negative']
train1 = data[data['sentiment'] == 'Positive']
train2 = data[data['sentiment'] == 'Neutral']
import matplotlib.pyplot as plt
import seaborn as sns
id_types = pd.Series([100, 150], index=['TV Shows', 'Movies'])

plt.figure(figsize=(14, 7))
sns.barplot(x=id_types.index, y=id_types.values, palette='muted')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Count of TV Shows vs Movies')
plt.show()
game_types = data['game'].value_counts()
game_types
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 7))
sns.barplot(x=game_types.values, y=game_types.index, palette='muted')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of TV Shows vs Movies')
plt.show()
sentiment_types = data['sentiment'].value_counts()
sentiment_types
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
plt.pie(x=sentiment_types.values, labels=sentiment_types.index, autopct='%1.1f%%', startangle=140)
plt.title('The Difference in the Type of Contents')
plt.show()
import seaborn as sns
sns.catplot(x='game', hue='sentiment', kind='count', height=7, aspect=2, data=data)
plt.title('Count of Sentiments by Game')
plt.show()
data.nunique()
v_data.nunique()