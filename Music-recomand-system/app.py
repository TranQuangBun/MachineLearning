import streamlit as st
import pickle
import pandas as pd
import requests


def fetch_poster(music_title):
    response = requests.get("https://saavn.me/search/songs?query={}&page=1&limit=2".format(music_title))
    try:
        data = response.json()
        return data['data']['results'][0]['image'][2]['link']
    except ValueError:
        print("Invalid JSON")
        return None

def recommend(musics):
    music_index = music[music['title'] == musics].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_music = []
    recommended_music_poster = []
    for i in music_list:
        music_title = music.iloc[i[0]].title
        recommended_music.append(music.iloc[i[0]].title)
        # recommended_music_poster.append(fetch_poster(music_title))
    return recommended_music


music_dict = pickle.load(open(r'C:\Users\Admin\musicrec.pkl', 'rb'))
music = pd.DataFrame(music_dict)

similarity = pickle.load(open(r'C:\Users\Admin\similarities.pkl', 'rb'))
st.title('Music Recommendation System')

selected_music_name = st.selectbox('Select a music you like', music['title'].values)

if st.button('Recommend'):
    names  = recommend(selected_music_name)
    st.text(names[0])
    st.text(names[1])
    st.text(names[2])
    st.text(names[3])
    st.text(names[4])