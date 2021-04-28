# CookingNaNa_RecommendationSystem

CookingNaNa is recommendation system base on the [dataset](https://www.kaggle.com/elisaxxygao/foodrecsysv1) from Kaggle. Developed using Python. Visulization using Streamlit.
The recommendation model recommends restaurants to user based on their preferences.

## Recommendation Model

The recommendation model is based on content-based filtering which uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback and collaborative filtering which is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating).

For the content-based filtering, the model recommend similar recipes to the users based on recommended products. We mainly use “cooking_directions” and “recipe_name” features to recommend recipe to the user. First, we use Doc2Vec instead of Word2Vec.Doc2Vec means convert a document to a vector. It can match sentences with different lengths, calculate the probability and then infer other similar vectors. Second, we use name matching. We use One Hot Encoding for inputs to calculate the distances between input and the DataFrame, and then return the best matching.


For the collaborative filtering,the model based on user score from over 380M historical rating records.

## Structure

Here is how we recommend the recipe to the user in streamlit.
1. Before user enter anything in our model, we recommend the Top 5 recipes with high rating to the user proactively.
2. User can search by keywords, and then the model will recommend the most matching recipe to the user sorted by the rating and number of reviews.
3. And the user also can get the recommendation based on the  nutrition, cooking time and different cooking method.
4. Under every recommendation, we also provide futher recommendation based on content-based filtering and using the doc2vec method.





