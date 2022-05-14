- 👋 Hi, I’m @gopiprasad008
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
gopiprasad008/gopiprasad008 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
#import library  for EDA
# Most basic stuff for EDA.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core packages for text processing.
import string
import re

# Libraries for text preprocessing.
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Loading some sklearn packaces for modelling.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Some packages for word clouds and NER.

from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image
import spacy
import en_core_web_sm# Most basic stuff for EDA.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core packages for text processing.
import string
import re

# Libraries for text preprocessing.
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Loading some sklearn packaces for modelling.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Some packages for word clouds and NER.

from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image
import spacy
import en_core_web_sm

#import dataset 
df1 = pd.read_csv('/content/ecom - data.csv')
df1.info()
df1.isnull()
df1.shape

# dropping duplicated
df1.duplicated('Description').value_counts()
df1.drop_duplicates(["Description"], keep=False, inplace=True)

#importing nlp for extracting  the words 
df1['Descr']=df1['Description'].str.replace("[^a-zA-Z0-9]", " ")
df1['Descr'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))
df1['Descr']=[review.lower() for review in df1['Descr']]

#importing the stopword and  tokenizing the word
stop = stopwords.words('english')
df1['Descr'].apply(word_tokenize).apply(
    lambda x: [word.lower() for word in x]).apply(nltk.tag.pos_tag).apply(
    lambda x: [word for word in x if word not in stop])
    
# lemmatizer and wording grammering 
# function to convert nltk tag to wordnet tag
lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


df1['Descr'].apply(lambda x: lemmatize_sentence(x))

#frequrncy distance occerunce in the dataset
all_words = ' '.join([text for text in df1['Descr']])
all_words = all_words.split()
words_df = FreqDist(all_words)

words_df = pd.DataFrame({'word':list(words_df.keys()), 'count':list(words_df.values())})
words_df

# Subsets top 30 words by frequency
words_df = words_df.nlargest(columns="count", n = 30) 

words_df.sort_values('count', inplace = True)

# Plotting 30 frequent words
plt.figure(figsize=(7,10))
ax = plt.barh(words_df['word'], width = words_df['count'])
plt.show()
# counting vector in the dataset 
count_vect = CountVectorizer(max_features=4000)
bow_da = count_vect.fit_transform(df1['Descr'])
bow_da[1]
# using ngram 
count_vect = CountVectorizer(ngram_range=(1,2))
Bigram_data = count_vect.fit_transform(df1['Descr'])
print(Bigram_data[1])
# doucumentation the word in the dataset 
tf_idf = TfidfVectorizer(max_features=5000)
tf_data = tf_idf.fit_transform(df1['Descr']).toarray()
tf_data


# scaling and kmeans cluster formulation 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import silhouette_samples, silhouette_score

ss=StandardScaler()
x= ss.fit_transform(tf_data)

for n_clusters in range(3,12):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(x)
    clusters = kmeans.predict(x)
    sil_avg = silhouette_score(x, clusters)
    print( n_clusters,sil_avg)

n_clusters = 6
silhouette_avg = -1
while silhouette_avg > 0.140:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=20)
    kmeans.fit(x)
    clusters = kmeans.predict(x)
    silhouette_avg = silhouette_score(x, clusters)

pd.Series(clusters).value_counts()

#Elbow curve segamentation 
inertias = []
list_k = list(range(1, 30))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(x)
    inertias.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, inertias, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Inertia'); inertias = []
list_k = list(range(1, 30))

#Plot the clustered data

km = KMeans(n_clusters=10) # applying k = 2
km.fit(x)
centroids = kmeans.cluster_centers_
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(tf_data[kmeans.labels_ == 0, 0],tf_data[kmeans.labels_ == 0, 1],
            c='green', label='cluster 4')
plt.scatter(tf_data[kmeans.labels_ == 1, 0], tf_data[kmeans.labels_ == 1, 1],
            c='blue', label='cluster2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='red', label='centroid')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data', fontweight='bold')


import matplotlib as mpl
import matplotlib.cm as cm
def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
    #____________________________
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        #___________________________________________________________________________________
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        #____________________________________________________________________
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        #______________________________________
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  

# worlds clouding 
from wordcloud import WordCloud
word_cloud_df = df1['Descr']
all_words = ' '.join([text for text in word_cloud_df])
 

wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='black', 
                      min_font_size = 10).generate(all_words)

#plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()






