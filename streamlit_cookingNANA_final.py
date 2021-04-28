import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import heapq
import base64
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import euclidean, hamming, cosine, cdist
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

st.set_page_config(page_title='CookingNaNa',page_icon='NaNaLogo.png')
st.balloons()

st.title(" Cooking NaNa Recipe Recommendation ")


with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# -----------------------------------  LOAD CSV -------------------------------------------------

# GET HOT RECIPE
@st.cache(persist=True,allow_output_mutation=True) #method to get data once and store in cache.
def get_data():
    url = "rating_count.csv"
    return pd.read_csv(url)

# getting data
df_rating_count = get_data()

# GET RECIPE NAME
@st.cache(persist=True,allow_output_mutation=True) #method to get data once and store in cache.
def get_data_recipe_name():
    url = 'recipe_name.csv'
    return pd.read_csv(url,index_col='recipe_id')

# getting data
df_recipe_name = get_data_recipe_name() 

# GET RAW RECIPE
def to_join(x):
    if type(x)==list:
        return ', '.join(x)
    return x
@st.cache(persist=True,allow_output_mutation=True)
def get_recipe_data():
    # url_recipe = "data_recipe_clean.csv"    #data_recipe_clean
    url_recipe = "data_recipe_deploy.csv"
    df = pd.read_csv(url_recipe,index_col='recipe_id')
    df['ingredients'] = df['ingredients'].str.split('^')
    df['ingredients'] = df['ingredients'].apply(lambda x: to_join(x))
    return df
@st.cache(persist=True,allow_output_mutation=True)
def get_rating_data():
    url_rating = "data_rating_clean.csv"
    return pd.read_csv(url_rating)

#   Getting data
df_raw_recipe = get_recipe_data()
df_raw_rating = get_rating_data()

# GET COOKING TIME  
@st.cache(persist=True,allow_output_mutation=True) #method to get data once and store in cache.
def get_data_cooking_time():
    url = "cooking_time_deploy.csv"
    return pd.read_csv(url)    

# getting data
df_cooking_time = get_data_cooking_time()

# GET INGRIDENT & COOKING DIRECTION DETAIL
@st.cache(suppress_st_warning=True)
def get_recipe():
    return pd.read_csv("raw_recipe_clean.csv")

recipe_df = get_recipe()


# -----------------------------------  SIDE BAR -------------------------------------------------

from PIL import Image
image = Image.open('NaNaLogo.PNG')

st.sidebar.image(image,width=200)

st.sidebar.subheader("Find a recipe...")

# source_txt = st.markdown("<input type='text' placeholder='Find a recipe...'><br>",unsafe_allow_html=True)
search_word = st.sidebar.text_input("",key='1002')
# submitted = st.sidebar.button("Search")

#   Preference search
trigger_nutrition = st.sidebar.checkbox("Search by Nutrition")
trigger_cooking_method = st.sidebar.checkbox("Search by Cooking Method")
trigger_cooking_time = st.sidebar.checkbox('Serch by Cooking Time')


# -------------------------------------- GET HOT RECIPE ----------------------------------------------

# randomly generate 3 most popular recipes
from random import sample
pop_recipe_lst = df_rating_count['recipe_id'].tolist()
ran_pop_recipe_lst = sample(pop_recipe_lst,3)

# get the image url from the df
img_url = []
for recipe in ran_pop_recipe_lst:
    img = df_rating_count[df_rating_count['recipe_id']==recipe]['image_url'].values[0]
    img_url.append(img)

# get the recipe name from the df
img_name = []
for recipe in ran_pop_recipe_lst:
    name = df_rating_count[df_rating_count['recipe_id']==recipe]['recipe_name'].values[0]
    img_name.append(name)
#  ------------------------------------- DOC2VEC FUNCTION ---------------------------------------

def parseF(x):
    x = x.strip("[")
    x = x.strip("]")
    wordList = x.split(",")
    return wordList

@st.cache(persist=True,allow_output_mutation=True)
def get_data_df_all():
    url_all = "df_all_deploy2.csv"
    df_all=pd.read_csv(url_all, index_col="recipe_id")
    df_all["ingredients"] = df_all["ingredients"].apply(parseF)
    df_all["cooking_directions"] = df_all["cooking_directions"].apply(parseF)
    df_all["Vect_recipe_name"] = df_all["Vect_recipe_name"].apply(parseF)
    df_all["recipe_tag"] = df_all["recipe_tag"].apply(parseF)
    return df_all

df_all = get_data_df_all()

model_directions = Doc2Vec.load("doc2vec_model_final")

def docToVecAlgo(dataframe, recipe_id, n):
    test_doc = dataframe.loc[recipe_id]["cooking_directions"]
    originalData = dataframe[dataframe.index == recipe_id]
    test_doc_vector = model_directions.infer_vector(test_doc)
    predicted, proba = zip(*model_directions.docvecs.most_similar(positive=[test_doc_vector], topn=3*n))
    new_dict = dataframe.iloc[list(predicted)]
    new_dict = pd.concat([new_dict, originalData], axis=0)
    return new_dict

def recipe_name_recommender_123(dataframe, recipe_id,N=5):
    name_lst = dataframe.loc[recipe_id]['Vect_recipe_name']
    # create dataframe used to store the number of matching word
    allName = pd.DataFrame(df_all.index).set_index("recipe_id")
    allName['recipe_name'] = df_all[['recipe_name']]
    allName['Vect_recipe_name'] = df_all[['Vect_recipe_name']]

    # check the word in name_lst, how many word is matching with other recipe
    allName['matching'] = allName['Vect_recipe_name'].apply(lambda x: sum([1 if a in name_lst else 0 for a in x]) )
    # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
    TopNRecommendation =  allName.sort_values(['matching'],ascending=False).head(N+1)
    TopNRecommendation =  TopNRecommendation[TopNRecommendation.index != recipe_id]

    return df_all.loc[TopNRecommendation.index]

def finalized(recipe_id, n=5):
    processing = docToVecAlgo(df_all,recipe_id,n*15)
    df_new = recipe_name_recommender_123(processing, recipe_id,n)
    return list(df_new.index)[:n]
#   ------------------------------ Deploy collabarative recommentdation ----------------------------------
def recipeRecommender_by_recipeID(recipeID , N = 3):
    userRecommended = list(df_raw_rating[df_raw_rating['recipe_id']==recipeID]['user_id'])
    recipeRelated = list(df_raw_rating[df_raw_rating['user_id'].isin(userRecommended)]['recipe_id'])
    Good_Recipe_lst = list(df_raw_recipe[df_raw_recipe['aver_rate']>=4].index)
    df_Recommended = df_raw_rating[df_raw_rating['recipe_id'].isin(Good_Recipe_lst)]
    df_Recommended = df_raw_rating[df_raw_rating['recipe_id'].isin(recipeRelated)]
    df_Recommended = df_raw_rating[df_raw_rating['user_id'].isin(userRecommended)]

    #User-item-rating matrix
    userRecommendedMatrix = pd.DataFrame.pivot_table(
        df_Recommended,
        values='rating',
        index='user_id',
        columns='recipe_id',).fillna(0)

    recipeRecommended = list(userRecommendedMatrix.sum().sort_values(ascending=False).index)
    if recipeID in recipeRecommended:
        recipeRecommended.remove(recipeID)
    return recipeRecommended[:N]
#  ---------------------------------- SHOW IMG & NAME FUNCTION ------------------------------------------
def show_img(lst):

    col1, col2, col3 = st.beta_columns(3)
    # get the image url from the df
    img_url = []
    for recipe in lst:
        img = df_recipe_name.loc[recipe]['image_url']
        img_url.append(img)

        
    # get the recipe name from the df
    img_name = []
    for recipe in lst:
        name = df_recipe_name.loc[recipe]['recipe_name']
        img_name.append(name)

    
    # visualize the img and name of the recipe 
    
    col1.image(img_url[0],width=128, use_column_width=True)
    col1.text(img_name[0])
    
    col2.image(img_url[1],width=128, use_column_width=True)
    col2.text(img_name[1])
    
    col3.image(img_url[2],width=128, use_column_width=True)
    col3.text(img_name[2])
    
        
#  ----------------------------------- BUTTON FUNCTION(LOAD RECIPE) -------------------------------------
def load_recipe(recipe_id):

    st.subheader(df_recipe_name.loc[recipe_id]['recipe_name'])

    recipe_name = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['recipe_name'].squeeze()

    # Part 1: Select info from dataframe
    # extract dataframe from raw_recipe_clean
    ingredients = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['ingredients'].squeeze()
    ingredients = ingredients.split("\n")

    cook_dir = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['cooking_directions'].squeeze()
    # ensure all unwanted characters are deleted
    if cook_dir[0].isalpha() == False:
        cook_dir = cook_dir.strip(cook_dir[0])
    if cook_dir[len(cook_dir)-1] != ".":
        cook_dir = cook_dir.strip(cook_dir[len(cook_dir)-1])

    cook_dir = cook_dir.split("\\n")

    # Part 2: Expander
    with st.beta_expander('Time Info'):
        try:
            st.write(f"{cook_dir[0]} : {cook_dir[1]}")
            # st.write('Preparation Time:', df_cooking_time.loc[recipe_id]['Prep_Time'],'mins')
        except:
            st.write('Preparation Time: N/A')
            # st.write('Cooking Time:', df_cooking_time.loc[recipe_id]['Cook_Time'],'mins')
        try:
            st.write(f"{cook_dir[2]} : {cook_dir[3]}")
        except:
            st.write('Cooking Time: N/A')


    with st.beta_expander('Ingredients'):
        for i in ingredients:
            st.write(i)

    with st.beta_expander('Direction'):
        for i in range(6, len(cook_dir)):
            st.write(cook_dir[i])

    with st.beta_expander('Related Recipe'):
        closest_recipe_list = finalized(recipe_id,3)
        show_img(closest_recipe_list)
    

    st.markdown('<h1>  </h1>', unsafe_allow_html=True)
    st.markdown('<h1>  </h1>', unsafe_allow_html=True)

    return
    
#  ----------------------------------- BUTTON FUNCTION(LOAD RECIPE_collab_) -------------------------------------
def load_recipe_collab(recipe_id):

    st.subheader(df_recipe_name.loc[recipe_id]['recipe_name'])

   
    recipe_name = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['recipe_name'].squeeze()


    # Part 1: Select info from dataframe
    # extract dataframe from raw_recipe_clean
    ingredients = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['ingredients'].squeeze()
    ingredients = ingredients.split("\n")

    cook_dir = recipe_df.loc[recipe_df['recipe_id'] == recipe_id]['cooking_directions'].squeeze()
    # ensure all unwanted characters are deleted
    if cook_dir[0].isalpha() == False:
        cook_dir = cook_dir.strip(cook_dir[0])
    if cook_dir[len(cook_dir)-1] != ".":
        cook_dir = cook_dir.strip(cook_dir[len(cook_dir)-1])

    cook_dir = cook_dir.split("\\n")

    # Part 2: Expander
    with st.beta_expander('Time Info'):
        # st.write('Preparation Time:', df_cooking_time.loc[recipe_id]['Prep_Time'],'mins')
        # st.write('Cooking Time:', df_cooking_time.loc[recipe_id]['Cook_Time'],'mins')
        st.write(f"{cook_dir[0]} : {cook_dir[1]}")
        st.write(f"{cook_dir[2]} : {cook_dir[3]}")
        # st.write(f"{cook_dir[4]} : {cook_dir[5]}")

    with st.beta_expander('Ingredients'):
        for i in ingredients:
            st.write(i)

    with st.beta_expander('Direction'):
        for i in range(6, len(cook_dir)):
            st.write(cook_dir[i])

    with st.beta_expander('Related Recipe'):
        closest_recipe_list = finalized(recipe_id,3)
        show_img(closest_recipe_list)

    with st.beta_expander('Recommended by Others'):
        collab_recipe_list = recipeRecommender_by_recipeID(recipe_id,3)
        show_img(collab_recipe_list)
    

    st.markdown('<h1>  </h1>', unsafe_allow_html=True)
    st.markdown('<h1>  </h1>', unsafe_allow_html=True)

    return
    

# ----------------------------------- SHOW IMG + EXPANDER FUNCTION ----------------------------------------------
def show_result(lst):

    # get the image url from the df
    img_url = []
    for recipe in lst:
        img = df_recipe_name.loc[recipe]['image_url']
        img_url.append(img)

        
    # get the recipe name from the df
    img_name = []
    for recipe in lst:
        name = df_recipe_name.loc[recipe]['recipe_name']
        img_name.append(name)


    # visualize the img and name of the recipe 
    for i in range(len(img_url)):
        st.image(img_url[i],width=512)
        load_recipe(lst[i])


#   --------------------------------- WORD PROCESSING FUNCTION -------------------------------------

# text processing
def token(words):
    words = re.sub("[^a-zA-Z]"," ",words)
    text = words.lower().split()
    return " ".join(text)

stop_words = stopwords.words('english')+['m','h','u','directions','f']

def stopwords(review):
    text = [ word.lower() for word in review.split() if word.lower() not in stop_words]
    return " ".join(text)

lem = WordNetLemmatizer()
def lemma(text):
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return lem_text

#   ---------------------------------  UPDATE MATCHING LIST   --------------------------------------

def update_matching_list(search_word):
    name_lst = lemma(stopwords(token(search_word)))
    print('search_word: ',search_word)
    print('name_lst:    ', name_lst)
    df_raw_recipe['matching'] = df_raw_recipe['recipe_name_clean'].apply(lambda x: sum(
     [1 if lem.lemmatize(a) in name_lst else 0 for a in x.split(" ")])) 
    return 
#  ----------------------------------- RECIPE NAME RECOMMENDER --------------------------------------
def recipe_name_recommender(search_word):
    name_lst = lemma(stopwords(token(search_word)))
    # create dataframe used to store the number of matching word
    allName = pd.DataFrame(df_recipe_name.index).set_index('recipe_id')
    allName['recipe_name'] = df_recipe_name[['recipe_name']]
    allName['aver_rate'] = df_recipe_name[['aver_rate']]
    allName['clear_recipe_name'] = df_recipe_name[['clear_recipe_name']]

    
    # check the word in name_lst, how many word is matching with other recipe
    allName['matching'] = allName['clear_recipe_name'].apply(lambda x: sum([1 if a in name_lst else 0 for a in x.replace(" ","").replace("[","").replace("]", "").replace("'","").split(",")]) )


    # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
    TopNRecommendation =  allName.sort_values(['matching','aver_rate'],ascending=False).head(5)

    TopNRecommendation_lst = TopNRecommendation.index.tolist()


    # return the list of recommend recipe
    return TopNRecommendation_lst

#   ---------------------------------  Nutrition Recommender   --------------------------------------
def nutrition_recommender_ByNutrition(nutrition_input, search_word, N = 5):
    update_matching_list(search_word)
    if search_word == "":
        nutrition_input['matching']=0
        df_raw_recipe['matching']=0

    else:
        nutrition_input['matching']=len(lemma(stopwords(token(search_word))))

    df_score = pd.DataFrame(list(df_raw_recipe.index),columns=['recipe_id'])
    hamming_col = ['niacin_Scaled', 'sugars_Scaled', 'sodium_Scaled','carbohydrates_Scaled', 'vitaminB6_Scaled', 'calories_Scaled',\
            'thiamin_Scaled', 'fat_Scaled', 'folate_Scaled','caloriesFromFat_Scaled', 'calcium_Scaled','fiber_Scaled','magnesium_Scaled',\
            'iron_Scaled', 'cholesterol_Scaled','protein_Scaled', 'vitaminA_Scaled', 'potassium_Scaled','saturatedFat_Scaled', 'Total_Time_Normalized','matching']
    df_score['Distance'] = cdist( np.reshape(list(nutrition_input.values()) , (1,21)) , df_raw_recipe.loc[:,hamming_col],'euclidean').T
    df_score.set_index('recipe_id',inplace=True)
    df_score.sort_values(["Distance"],ascending=True,inplace=True)
    recommend_lst = list(df_score.index[0:N])
    return recommend_lst
#   -----------------------------------  Cooking Recommender   ---------------------------------------
def by_cooking_method_recommender(cooking_method_input, search_word, N = 5):
    update_matching_list(search_word)
    cm_arr = [0,0,0,0,0,0,0,0,0,0]
    if search_word == "":
        cm_arr.append(0)
        df_raw_recipe['matching']=0
    
    else:
        cm_arr.append(len(lemma(stopwords(token(search_word)))))

    for x in range(10):
        if cooking_options[x]==cooking_method_input:
            cm_arr[x] = 1

    allRecipes = pd.DataFrame(list(df_raw_recipe.index),columns=['recipe_id'])
    hamming_col = cooking_options
    hamming_col.append('matching')
    allRecipes['Distance'] = cdist( np.reshape(cm_arr , (1,len(cm_arr))) , df_raw_recipe.loc[:,hamming_col],'euclidean').T
    allRecipes.set_index('recipe_id',inplace=True)
    allRecipes.sort_values(["Distance"],ascending=True,inplace=True)
    recommend_lst = list(allRecipes.index[0:N])
    return recommend_lst
# ----------------------------------- COOKING TIME FUNCTION -----------------------------------------
def by_cook_time(search_word, cooking_time_limit):
    name_lst = lemma(stopwords(token(search_word)))
    # create dataframe used to store the number of matching word
    allName = pd.DataFrame(df_raw_recipe.index).set_index('recipe_id')
    allName['recipe_name'] = df_raw_recipe[['recipe_name']]
    allName['aver_rate'] = df_raw_recipe[['aver_rate']]
    allName['recipe_name_clean'] = df_raw_recipe[['recipe_name_clean']]
    allName['Total_Time'] = df_raw_recipe[['Total_Time']]
    if cooking_time_limit == 'Less than 5mins':
        allName = allName[(allName['Total_Time']<=5)&(allName['Total_Time']>0)]
    elif cooking_time_limit == '5-15mins':
        allName = allName[(allName['Total_Time']>5)&(allName['Total_Time']<=15)]
    elif cooking_time_limit == '15-30mins':
        allName = allName[(allName['Total_Time']>15)&(allName['Total_Time']<=30)]
    elif cooking_time_limit == '30-60mins':
        allName = allName[(allName['Total_Time']>30)&(allName['Total_Time']<=60)]
    elif cooking_time_limit =='1-2hrs':
        allName = allName[(allName['Total_Time']>60)&(allName['Total_Time']<=120)]
    else:
        allName = allName[(allName['Total_Time']>120)]
    # check the word in name_lst, how many word is matching with other recipe
    allName['matching'] = allName['recipe_name_clean'].apply(lambda x: sum(
        [1 if lem.lemmatize(a) in name_lst else 0 for a in x.split(" ")])) 
    TopNRecommendation =  allName.sort_values(['matching','aver_rate'],ascending=False).head(5)
    TopNRecommendation_lst = TopNRecommendation.index.tolist()
    return TopNRecommendation_lst

# ----------------------------------- Main Display ----------------------------------------------
if trigger_cooking_method ==False and trigger_nutrition==False and trigger_cooking_time==False and search_word=="":

    # Hot Recipe display
    st.header(':blush: Hot Recipe!!! :blush:')

    st.image(img_url[0],width=512, use_column_width=True)
    load_recipe_collab(ran_pop_recipe_lst[0])
    
    st.image(img_url[1],width=512, use_column_width=True)
    load_recipe_collab(ran_pop_recipe_lst[1])
    
    st.image(img_url[2],width=512, use_column_width=True)
    load_recipe_collab(ran_pop_recipe_lst[2])
    

elif trigger_cooking_method == False and trigger_nutrition==False and trigger_cooking_time==False and search_word!="":    

    recipe_list = recipe_name_recommender(search_word)
    show_result(recipe_list)
    

elif trigger_nutrition==True:
    # Nutrition Input
    st.sidebar.title("Nutrition Ratio")
    calories_input = st.sidebar.slider('Calories:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="9")
    carbohydrates_input = st.sidebar.slider('Carbohydrates:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="1")
    fiber_input = st.sidebar.slider('Fiber:', min_value=0.0, max_value=1.0, step=0.1,value=0.8, key="2")
    fat_input = st.sidebar.slider('Fat:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="10")
    sugars_input = st.sidebar.slider('Sugars:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="14")
    cholesterol_input = st.sidebar.slider('Cholesterol:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="3")
    protein_input = st.sidebar.slider('Protein:', min_value=0.0, max_value=1.0, step=0.1,value=0.8, key="4")
    saturatedFat_input = st.sidebar.slider('SaturatedFat:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="11")
    caloriesFromFat_input = st.sidebar.slider('CaloriesFromFat:', min_value=0.0, max_value=1.0, step=0.1,value=0.3, key="12")
    vitaminA_input = st.sidebar.slider('VitaminA:', min_value=0.0, max_value=1.0, step=0.1, key="5")
    vitaminB6_input = st.sidebar.slider('VitaminB6:', min_value=0.0, max_value=1.0, step=0.1, key="6")
    vitaminC_input = st.sidebar.slider('VitaminC:', min_value=0.0, max_value=1.0, step=0.1, key="7")
    calcium_input = st.sidebar.slider('Calcium:', min_value=0.0, max_value=1.0, step=0.1, key="8")
    folate_input = st.sidebar.slider('Folate:', min_value=0.0, max_value=1.0, step=0.1, key="13")
    sodium_input = st.sidebar.slider('Sodium:', min_value=0.0, max_value=1.0, step=0.1, key="15")
    thiamin_input = st.sidebar.slider('Thiamin:', min_value=0.0, max_value=1.0, step=0.1, key="16")
    niacin_input = st.sidebar.slider('Niacin:', min_value=0.0, max_value=1.0, step=0.1, key="17")
    magnesium_input = st.sidebar.slider('Magnesium:', min_value=0.0, max_value=1.0, step=0.1, key="18")
    iron_input = st.sidebar.slider('Iron:', min_value=0.0, max_value=1.0, step=0.1, key="19")
    potassium_input = st.sidebar.slider('Potassium:', min_value=0.0, max_value=1.0, step=0.1, key="20")
    
    nutritions_lst = ['carbohydrates','fiber','cholesterol','protein','vitaminA','vitaminB6','vitaminC','calcium',\
        'calories','fat','saturatedFat','caloriesFromFat','folate','sugars','sodium','thiamin','niacin','magnesium','iron','potassium',]
    
    nutrition_input = {'niacin_input':niacin_input, 'sugars_input':sugars_input, 'sodium_input':sodium_input, 'carbohydrates_input':carbohydrates_input,\
        'vitaminB6_input':vitaminB6_input, 'calories_input':calories_input, 'thiamin_input':thiamin_input, 'fat_input':fat_input, 'folate_input':folate_input,
        'caloriesFromFat_input':caloriesFromFat_input, 'calcium_input':calcium_input, 'fiber_input':fiber_input, 'magnesium_input':magnesium_input,\
        'iron_input':iron_input, 'cholesterol_input':cholesterol_input, 'protein_input':protein_input, 'vitaminA_input':vitaminA_input, \
        'potassium_input':potassium_input, 'saturatedFat_input':saturatedFat_input, 'vitaminC_input':vitaminC_input }
    
    recipe_list = nutrition_recommender_ByNutrition(nutrition_input,search_word)
    show_result(recipe_list)

elif trigger_cooking_method==True:
    st.sidebar.title("Pick a Cooking Method: ")
    cooking_options = ['baking','frying','roasting','grilling','steaming','poaching','simmering','broiling','stewing','braising']
    cooking_method_input = st.sidebar.selectbox(" ", cooking_options)
    recipe_list = by_cooking_method_recommender(cooking_method_input,search_word)
    show_result(recipe_list)    

elif trigger_cooking_time==True:
    st.sidebar.title("Choose the Cooking Time ")
    cooking_time_input = st.sidebar.selectbox(" ", ['Less than 5mins','5-15mins','15-30mins','30-60mins','1-2hrs','More than 2hrs'])
    cooking_time_lst = by_cook_time(search_word,cooking_time_input)
    show_result(cooking_time_lst)

