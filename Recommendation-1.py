import pandas as pd

# import Dataset 
game = pd.read_csv(r"C:\Users\Koushik\Desktop\game.csv", encoding = 'utf8')
game.shape # shape
game.columns
game.game # game columns

from sklearn.feature_extraction.text import TfidfVectorizer ##counting frequency of words
# term frequency inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer ##stop words are words that doesnt give any meaning i.e =in,on...

#Checking for null values
game.isnull().sum()


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000,3068

# From the above matrix we need to find the similarity score.
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 
# cosine(x,y)= (x.y⊺)/(||x||.||y||)

# calculating the dot product using sklearn's linear_kernel()
from sklearn.metrics.pairwise import linear_kernel ##linear kernel used to calculate similarity

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix) ##(bw same rows and columns i.e=tfid amtrix and tfid matrix)

# creating a mapping of game game to index number 
game_index = pd.Series(game.index, index = game['game']).drop_duplicates() ##strong game with index

game_id = game_index["SoulCalibur"]
game_id

def get_recommendations(game, topN):    
    # topN = 10
    # Getting the movie index using its title 
    game_id = game_index[game]
    
    # Getting the pair wise similarity score for all the game's with that 
    # game
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1] ##+1 as index starts ith 0
    
    # Getting the movie index ##separating index and cosine
    game_idx  =  [i[0] for i in cosine_scores_N] ##index
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    print (game_similar_show)
    # The End

    
# Enter your game and number of game's to be recommended
game_index["Super Mario Galaxy"]
get_recommendations("Super Mario Galaxy", topN = 10)

##It gives the top 10 recommendations




