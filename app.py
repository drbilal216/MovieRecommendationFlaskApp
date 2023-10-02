from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

knn_user_model = pickle.load(open('knn_user_model', 'rb'))
merged_dataset2 = pd.read_csv('merged_dataset2.csv')
user_to_movie_df = pd.read_csv('user_to_movie_df.csv')
movies_list = user_to_movie_df.columns
#####################
# load the saved Knn model
knn_movie_model = pickle.load(open('knn_movie_model', 'rb'))
# loading csv matrix
movie_to_user_df = pd.read_csv('movie_to_user_df.csv')
# Creating a dictionary with movie name as key and its index from the list as value:
movie_dict = {movie : index for index, movie in enumerate(movies_list)}
# movie name in lower character
case_insensitive_movies_list = [i.lower() for i in movies_list]
#####################

# Spell Checker
def get_possible_movies(movie):

    temp = ''
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie :
      out = []
      temp += i
      for j in possible_movies:
        if temp in j:
          out.append(j)
      if len(out) == 0:
          return possible_movies
      out.sort()
      possible_movies = out.copy()

    return possible_movies
#

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

######################################################
@app.route('/process_form2', methods=['POST'])
def process_form2():
    genre = request.form.get("genre")
    movie_name = str(request.form['user_input2'])
    
    ####
    try:
       movie_name_lower = movie_name.lower()
       if movie_name_lower not in case_insensitive_movies_list :
           raise ValueError()
    except ValueError:
       possible_movies = get_possible_movies(movie_name_lower)
       indices = [case_insensitive_movies_list.index(i) for i in possible_movies]
       PossibleMovies = [movies_list[i] for i in indices[:10]]
       return render_template('index.html', error_message=PossibleMovies), 400
    else:
       movie_name = str(request.form['user_input2'])
    ####
    
    index = movie_dict[movie_name]
    knn_input_m = np.asarray([movie_to_user_df.values[index]])
    n = 800 #10
    n = min(len(movies_list)-1,n)
    distances, indices = knn_movie_model.kneighbors(knn_input_m, n_neighbors=n+1)

    L = []
    for i in range(1,len(distances[0])):
        L.append(movies_list[indices[0][i]])

    # also saving Movies Genre in seperate column
    TList = []
    GList = []
    MovieRecommended = L
    for i in MovieRecommended:
        TList.append(i)
        index = merged_dataset2.title.tolist().index(i)
        Mgenre = merged_dataset2[["genres"]].iloc[index]
        Mgenre = Mgenre.to_string(index=False)
        GList.append(Mgenre)

    RecommendationGenre = pd.DataFrame([TList, GList], index=['title', 'genres']).T.explode('genres')
    R = ((RecommendationGenre).values)

    action_movies = RecommendationGenre["genres"].str.contains(genre)
    MG = (RecommendationGenre.loc[action_movies, ["title", "genres"]][:10].reset_index(drop=True).values)

    return render_template('index.html', result2=MG)
######################################################


if __name__ == "__main__":
    app.run(debug=True)