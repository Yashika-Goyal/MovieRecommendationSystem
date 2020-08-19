#Movie Recommendation System 


# DATASET link - https://www.kaggle.com/rounakbanik/the-movies-dataset

import numpy as np
import pandas as pd 


# Read ratings data Set
ratings_data_frame = pd.read_csv("ratings_small.csv")
ratings_data_frame.shape

ratings_data_frame.head()

# Check the dataset for Duplicate values
ratings_data_frame.userId.nunique()

ratings_data_frame.movieId.nunique()


# Read Movies data Set

movies_data_frame = pd.read_csv('movies_metadata.csv', low_memory=False)

movies_data_frame.head()

movies_data_frame.shape

movies_data_frame.shape

type(movies_data_frame.id[0])

# Convert the datatype of the field 'id' from string to integer

movies_data_frame.id = movies_data_frame.id.astype(np.int64)

type(movies_data_frame.id[0])

#Merge the movie and ratings data files to display the ratings given by each user to different movies
ratings_data_frame = pd.merge(ratings_data_frame,movies_data_frame[['title','id']],left_on='movieId',right_on='id')
ratings_data_frame.head()

# Dropping the 'timestamp' and the 'id' column
ratings_data_frame.drop(['timestamp','id'],axis=1,inplace=True)

ratings_data_frame.shape

ratings_data_frame.sample(5)

ratings_data_frame.to_csv("MergedRatings_Movie.csv")


# Calculating the average rating for each movie


rating_scores_mean_for_each_movie = ratings_data_frame.groupby(['title'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})[['title','rating_mean']]


rating_scores_mean_for_each_movie.head()

rating_scores_mean_for_each_movie.sort_values(by=['rating_mean'])

# Selecting the movies which have average rating of more than 1
Ratings_more_than_1 = rating_scores_mean_for_each_movie.query('rating_mean > 1 ')

Ratings_more_than_1.sort_values(by=['rating_mean'])


# Merging rating_mean to Ratings data frame after removing the movies with avg rating <1
ratings_data_frame = pd.merge(ratings_data_frame,Ratings_more_than_1,on='title')
ratings_data_frame.head()

ratings_data_frame.to_csv("2ndTransformRatingsDF.csv")

# Checking for null values in the ratings_df dataset
ratings_data_frame.isnull().sum()

# The number of ratings for each movie
ratings_count = ratings_data_frame.groupby(by="title")['rating'].count().reset_index().rename(columns={'rating':'totalRatings'})[['title','totalRatings']]

#See the count of unique movies in the ratings_count dataframe
ratings_count.shape[0]
ratings_count.sample(5)

ratings_count.to_csv("TotalRating.csv")

ratings_count.head()

ratings_data_frame.head()

#Merging total number of ratings given by the user to each movie to ratings_df 

ratings_total = pd.merge(ratings_data_frame,ratings_count,on='title',how='left')


ratings_total.to_csv("Ratings_total.csv")

ratings_total.shape

ratings_total.head()

ratings_count['totalRatings'].describe()

ratings_count['totalRatings'].quantile(np.arange(.6,1,0.01))

#About top 21% of the movies received more than 30 votes. Let's remove all the other movies so that we are only left with significant movies (in terms of total votes count)
votes_count_threshold = 30

# Got the final clean dataset ratings_top to be use in the Model

#Merging Ratings df and removing the movies that were rated by less than 30 users
ratings_top = ratings_total.query('totalRatings > @votes_count_threshold')

ratings_top.shape

#This dataset contains only those movies , which have recieved ratings from at least 20 ppl

ratings_top.head()

ratings_top.to_csv("3rdTransformRatingsTop.csv")

#Making the data consistent by ensuring there are unique entries for [title,userId] pairs
if not ratings_top[ratings_top.duplicated(['userId','title'])].empty:
    ratings_top = ratings_top.drop_duplicates(['userId','title'])

ratings_top.shape

# Reshaping the data using pivot function

df_for_knn = ratings_top.pivot(index='title',columns='userId',values='rating').fillna(0)
# building dataframe for applying KNN algorithm - creating a matrix for which userid rated which movie and what rating he gave

df_for_knn.head()

df_for_knn.shape

from scipy.sparse import csr_matrix

# Creating a sparse matrix
df_for_knn_sparse = csr_matrix(df_for_knn.values)

from sklearn.neighbors import NearestNeighbors

# Recommendation using KNN algo with Cosine distance measure

model_knn = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=10)

#Fitting the model here
model_knn.fit(df_for_knn_sparse)

#Using random movie index 
#query_index = np.random.choice(df_for_knn.shape[0])

#Using movie index 100
query_index = 100

# considering 5 nearest neighbors for a randomly selected movie
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=5)

for i in range(0,len(distances.flatten())):
    if i==0:
        print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))


# trying with different Neighbors
#model_knn = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=2)
# model_knn.fit(df_for_knn_sparse)

# query_index = 100
# distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=10)
# for i in range(0,len(distances.flatten())):
#     if i==0:
#         print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
#     else:
#         print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))



#Trying difference distance metrics - Euclidean

model_knn = NearestNeighbors(metric='euclidean',algorithm='brute',n_neighbors=5)
model_knn.fit(df_for_knn_sparse)

query_index = 100
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=10)
for i in range(0,len(distances.flatten())):
     if i==0:
         print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
     else:
         print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))



#Trying difference distance metrics - Manhattan
model_knn = NearestNeighbors(metric='manhattan',algorithm='brute',n_neighbors=5)
model_knn.fit(df_for_knn_sparse)

query_index = 100
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=10)
for i in range(0,len(distances.flatten())):
     if i==0:
         print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
     else:
         print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))



#Matrix factorization


df_for_MF = ratings_top.pivot(index='title',columns='userId',values='rating').fillna(0)

df_for_MF.head()

df_for_MF.shape


import sklearn
from sklearn.decomposition import TruncatedSVD

# Using SVD - Simgle Value Decomposition
SVD = TruncatedSVD(n_components=12,random_state=17)
matrix = SVD.fit_transform(df_for_MF)
matrix.shape


import warnings
warnings.filterwarnings("ignore",category = RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape

import numpy 
T_df_for_MF = numpy.transpose(df_for_MF)

T_df_for_MF.head()

movie_title = T_df_for_MF.columns
movie_list = list(movie_title)
movie_recom = movie_list.index("Batman Returns")
print(movie_recom)

corr_movie_recom = corr[movie_recom]

list(movie_title[(corr_movie_recom < 1.0) & (corr_movie_recom > 0.9)])


