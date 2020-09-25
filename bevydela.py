import streamlit as st
import sys
import re
import pandas   as pd
import numpy    as np
import helper
import general
import tf_idf
from wordcloud import WordCloud 


import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections     import Counter
import matplotlib.pyplot as plt


###############################
#######    Funcions     #######
###############################
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def read_files():
   df = helper.import_data('./chatt.txt')
   df = helper.preprocess_data(df)
   df['Date'] = df['Date'].apply(pd.to_datetime)
   df.sort_values('Date', inplace=True)
   return df


@st.cache(allow_output_mutation=True)
def filter(df,any, mesos):
   df = df[(df['Year']>=any[0])& (df['Year']<=any[1])]
   df = df[(df['Month']>=mesos[0])& (df['Month']<=mesos[1])]
   return df

###############################
#######       Viz       #######
###############################
# Titol i subtitol

st.markdown("<h1 style='text-align: center; color: #ed6d9b;font-family:verdana;font-size:300%;'>Bevydela</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;font-family:verdana; color: #D0D3DA;'>Conversa de Whatsapp\n", unsafe_allow_html=True)
st.write("# ")

df = read_files()

st.write("# ")


df1 = df[ df['User'] == 'Ge']
df2 = df[ df['User']=='Àlex'] 


st.write('El període de temps que ens empara és del:',df.Date[0],'al ', df.Date[308624],'. En aquest cas es tracta de la conversa de Whatsapp entre: ',df.User.unique()[0], ' i ',df.User.unique()[1],'amb un total de ',len(df1),'i',len(df2) ,'missatges respectivament' )
st.write("# ")

# Visualitzacions temporal
any = st.slider('Anys', 2018,2020, (2018,2020) )
mes = st.slider('Mesos', 1,12, (1,12) )
da = df

# Global
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Evolució de la conversa</h1>", unsafe_allow_html=True)
st.write("# ")
st.pyplot(general.plot_messages(filter(da,any,mes), colors=None, trendline=False, savefig=False, dpi=100))
# Dies
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Comunicació per dies</h1>", unsafe_allow_html=True)
st.write("# ")
st.pyplot(general.plot_day_spider(filter(da,any,mes), colors=None, savefig=False, dpi=100))
# Quantitat
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Nombre de missatges per dia</h1>", unsafe_allow_html=True)
st.write("# ")
fig5 = px.scatter(filter(da,any,mes), x='Date' , y='N_words', color = 'User',color_discrete_map={'Ge': '#D0D3DA', 'Àlex':'#e73575' }, size = 'N_words',width=770,height=400)
fig5.update_layout(showlegend=True)
fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig5.update_layout({"yaxis"+str(i+1): dict(range = [10, 200]) for i in range(4)})
st.plotly_chart(fig5)
# Hores
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Hores de activitat</h1>", unsafe_allow_html=True)
st.write("# ")
st.pyplot(general.plot_active_hours(filter(da,any,mes), color='#ed6d9b', savefig=False, dpi=100, user='Àlex'))
st.pyplot(general.plot_active_hours(filter(da,any,mes), color='#D0D3DA', savefig=False, dpi=100, user='Ge'))

st.write("# ")


import emoji2
temp = df[['index', 'Message_Raw', 'User', 'Message_Clean', 'Message_Only_Text']].copy()
temp = emoji2.prepare_data(temp)



# Count all emojis
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Emoji World</h1>", unsafe_allow_html=True)
st.write("# ")
st.write('Els emojis que mes ha enviar la Gemma són: ' )
st.write("# ")

counts = emoji2.count_emojis(temp, non_unicode=True)
emojig = pd.DataFrame()
emojig['Emoji'] = counts['Ge'].keys()
emojig['Ge'] = counts['Ge'].values()

emojia = pd.DataFrame()
emojia['Emoji'] = counts['Àlex'].keys()
emojia['Àlex'] = counts['Àlex'].values()

st.table(emojig.sort_values('Ge',ascending=False).head(5))

st.write('Els emojis que mes utilitza el Àlex són: ' )
st.write("# ")
st.table(emojia.sort_values('Àlex',ascending=False).head(5))

# Dies mes populars
df1 = df[ df['User'] == 'Ge']
df2 = df[ df['User']=='Àlex'] 

ff = pd.DataFrame(df1['Date'].value_counts().rename_axis('dia').reset_index(name='counts')).head(5)
ff2 = pd.DataFrame(df2['Date'].value_counts().rename_axis('dia').reset_index(name='counts')).head(5)
dia = ff['dia']
counts =ff['counts']
dia2 = ff2['dia']
counts2 =ff2['counts']

st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Dies més populars</h1>", unsafe_allow_html=True)
st.write("# ")

import plotly.graph_objects as go
fig7 = go.Figure()
fig7.add_trace(go.Scatter(
    x=dia, y=counts,
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#D0D3DA'),
    stackgroup='one' # define stack group

))

fig7.add_trace(go.Scatter(
    x=dia2, y=counts2,
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#e73575'),
    stackgroup='one'

))

fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=770,height=400)
fig7.update_layout(showlegend=False)
st.plotly_chart(fig7)




st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Paraules més utilitzades</h1>", unsafe_allow_html=True)
st.write("# ")


stop_words = open('catala.txt','r',encoding = 'utf-8').read().split()
comment_words = '' 
for val in df.Message_Only_Text: 
    # typecaste each val to string 
    val = str(val) 
    # split the value 
    tokens = val.split()  
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    
    comment_words += " ".join(tokens)+" "
   
wordcloud = WordCloud(width = 1000, height = 400, background_color ='white', stopwords = stop_words, colormap='PuRd',min_font_size = 10).generate(comment_words)   
plt.figure(figsize = (20, 7), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

st.pyplot(plt.show())






