import streamlit as st
import sys
import re
import pandas   as pd
import numpy    as np

import emoji                as emoji_package

import operator

import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from tqdm import tqdm 

import matplotlib.dates     as mdates
from matplotlib.colors      import ColorConverter, ListedColormap
from matplotlib.lines       import Line2D



###############################
#######    Funcions     #######
###############################
st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_active_hours(df, color='#ffdfba', savefig=False, dpi=100, user='All'):
    """ Plot active hours of a single user or all 
    users in the group. A bar is shown if in that hour
    user(s) are active by the following standard:
    
    If there are at least 20% of the maximum hour of messages
    
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    color : str, default '#ffdfba'
        Hex color of bars
    savefig : boolean, deafult False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
    user : str, default 'All'
        Variable to choose if you want to see the active hours
        of a single user or all of them together. 'All' shows 
        all users and the name of the user shows that user. 
    
    """
    # Prepare data for plotting
    if user != 'All':
        df = df.loc[df.User == user]
        title = 'Hores Active de {}'.format(user)
    else:
        title = 'Active hours of all users'
        
    hours = df.Hour.value_counts().sort_index().index
    count = df.Hour.value_counts().sort_index().values
    font = {'fontname':'Comic Sans MS'}

    # Only get active hours
    #count = [1 if x > (.2 * max(count)) else 0 for x in count]

    # Plot figure
    fig, ax = plt.subplots()
    
    # Then plot the right part which covers up the right part of the picture
    ax.bar(hours, count, color=color,align='center', width=1,
            alpha=1, lw=4, edgecolor='w', zorder=2)

    # Set ticks and labels
    ax.yaxis.set_ticks_position('none') 
    ax.set_yticks([])
    ax.set_ylabel('', labelpad=50, rotation='horizontal',
                   color="#6CA870",**font)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xticklabels(["Mitjanit", "3 AM", "6 AM", "9 AM", "Migdia", "3 PM", "6 PM", "9 PM", 
                       "Mitjanit"], **font)
    plt.title(title, y=0.8)
    
    # Create horizontal line instead of x axis
    plt.axhline(0, color='black', xmax=1, lw=2, zorder=3, clip_on=False)

    # Make axes white to remove any image line that may be left
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Remove the left and bottom axis
    ax.spines['left'].set_visible(False)

    # Set sizes
    fig.set_size_inches((13.5, 1))
    fig.tight_layout(rect=[0, 0, .8, 1])

    # Save or show figure    
    if savefig:
        plt.savefig(f'results/{savefig}active_hours.png', dpi = dpi)
    else:
        plt.show()


def plot_day_spider(df, colors=None, savefig=False, dpi=100):
    """ Plot active days in a spider plot with all users
    shown seperately. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing all messages
    colors : list, default None
        List of colors to be used for the plot. 
        Random colors are chosen if nothing is chosen
    savefig : boolean, deafult False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
        
    """
    
    # Initialize colors
    if not colors:
        colors = ['#ed6d9b','#D0D3DA']

    # Get count per day of the week
    categories = ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous', 'Divendres', 'Disabte', 'Diumenge']
    N = len(categories)
    count = list(df.Day_of_Week.value_counts().sort_index().values)
    count += count[:1]

    # Create angles of the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [], color='grey', size=12)
    ax.set_yticklabels([])

    # Plot data
    max_val = 0
    legend_elements = []    
    
    for index, user in enumerate(df.User.unique()):
        values = list(df[df.User == user].Day_of_Week.value_counts().sort_index().values)
        values += values[:1]
        
        if len(values) < 8:
            continue
        
        # Set values between 0 and 1
        values = [(x - min(values)) / (max(values) - min(values)) + 1 for x in values]
        

        ax.plot(angles, values, linewidth=2, linestyle='solid', zorder=index, color=colors[index], alpha=0.8)
        ax.fill(angles, values, colors[index], alpha=0.1, zorder=0)

        if max(values) > max_val: max_val = max(values) # To align ytick labels
            
        legend_elements.append(Line2D([0], [0], color=colors[index], lw=4, label=user))

    # Draw ytick labels to make sure they fit properly
    for i in range(len(categories)):
        angle_rad = i/float(len(categories))*2*np.pi
        angle_deg = i/float(len(categories))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, max_val*1.15, categories[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")
    
    # Legend and title
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    plt.title('', y=1.2)
    
    # Save or show figure    
    if savefig:
        plt.savefig(f'results/spider_plot.png', dpi = dpi)
    else:
        plt.show()
         
def plot_messages(df, colors=None, trendline=False, savefig=False, dpi=100):
    
        
    # Prepare data
    if not colors:
        colors = ['#ed6d9b','#D0D3DA']

    df = df.set_index('Date')   
    users = {user: df[df.User == user] for user in df.User.unique()}
    
    # Resample to a week by summing
    for user in users:
        users[user] = users[user].resample('7D').count().reset_index()
    
    # Create figure and plot lines
    fig, ax = plt.subplots()
    legend_elements = []
    
    for i, user in enumerate(users):
        ax.plot(users[user].Date, users[user].Message_Raw, linewidth=3, color=colors[i])
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=4, label=user))

    # calc the trendline
    if trendline:
        x = [x for x in users[user].Date.index]
        y = users[user].Message_Raw.values
        z = np.polyfit(x, y, 5)
        p = np.poly1d(z)
        ax.plot(users[user].Date, p(x), linewidth=2, color = 'g')

    # Remove axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    font = {'fontname':'Comic Sans MS', 'fontsize':40}
    ax.set_ylabel('Nr of Messages', {'fontname':'Comic Sans MS', 'fontsize':14})
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.)

    # Set size of graph
    fig.set_size_inches(20, 10)
    
    # Creating custom legend
    custom_lines = [Line2D([], [], color=colors[i], lw=4, 
                          markersize=6) for i in range(len(colors))]

    # Create horizontal grid
    ax.grid(True, axis='y')
    
    # Legend and title
    ax.legend(custom_lines, [user for user in users.keys()], bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    plt.title("", fontsize=20)
    
    if savefig:
        plt.savefig(f'results/moments.png', format="PNG", dpi=dpi)
    else:
        plt.show()
           
 
def import_data(file, path = ''):
    with open('./chatt.txt', encoding = 'utf-8') as outfile:
        raw_text = outfile.readlines()
        messages = {}

        # Getting all the messages for each user 
        messages_per_user = {}

        for message in raw_text: 

            # Some messages are not sent by the user, 
            # but are simply comments and therefore need to be removed
            try:
                name = message.split(' ')[2].split(':')[0]
            except:
                continue

            # Add name to dictionary if it exists
            if name in messages:
                messages[name].append(message)
            else:
                messages[name] = [message]

    df = pd.DataFrame(columns=['Message_Raw', 'User'])
    for name in messages.keys():
        df = df.append(pd.DataFrame({'Message_Raw': messages[name], 'User': name}))

    df.reset_index(inplace=True)
    df1 = df[ df['User'] == 'Ge']
    df2 = df[ df['User']=='Àlex'] 
    df= df1.append(df2)
    return df

def clean_message(row):
    name = row.User + ':'
    
    try:
        return row.Message_Raw.split(':')[3][:-1]
    except:
        return row.Message_Raw
    
def remove_inactive_users(df, min_messages=10):
    # Remove users that have not posted more than min_messages
    to_keep = df.groupby('User').count().reset_index()
    to_keep = to_keep.loc[to_keep['Message_Raw'] >= min_messages, 'User'].values
    df = df[df.User.isin(to_keep)]
    return df


def preprocess_data(df, min_messages=10):
    # Create column with only message, not date/name etc.
    df['Message_Clean'] = df.apply(lambda row:clean_message(row), axis = 1)
  

    # Create column with only text message, no smileys etc.
    df['Message_Only_Text'] = df.apply(lambda row: re.sub(r'[^a-zA-Z ]+', '', row.Message_Clean.lower()),axis = 1)
    
    # Create column with th number of words for message
    df['N_words'] = df.apply(lambda row: len(row.Message_Clean.split()),axis = 1)
    
    # Remove inactive users
    df = remove_inactive_users(df, 10)

   # Remove indices of images
    indices_to_remove = list(df.loc[df.Message_Clean.str.contains('|'.join(['<', '>'])),'Message_Clean'].index)
    df = df.drop(indices_to_remove)
    
    # Extract Time
    # Date
    df['Date'] = df.apply(lambda row: row['Message_Raw'].split(']')[0], axis = 1)
    sample_list = []

    for i in df['Date']:
        l1 = i.split(' ')[0]
        if len(l1) < 9:
            p1 = i.split('/')[0]
            p2 = i.split('/')[1:2]
            if len(p1)==2: 
                par = i[:1]+'0'+i[1:]
            elif len(p2)!=2: 
                par = i[:4]+'0'+i[4:]
            sample_list.append(par[1:])
        else:
            sample_list.append(i[1:])
    
    
    defin = []
    for data in sample_list:
        if data[0] =='[':
            p1 = data.split('/')[0]
            if len(p1) == 2:
                defin.append('0'+ data[1:])
            else:
                defin.append(data[1:])
            
        else:
            defin.append(data)
            
    df['Date'] = pd.DataFrame(defin)
    df = df.reset_index(drop=True)
    df = df[df['Date'].notna()]       
            
    from tqdm import tqdm 
    for i in tqdm(range(len(df))): 
#     if df.Date[i] != '0' or df.Date[i].find('/') != 1 :
        if (df.Date[i].find('/') != -1) :
            df.Date[i] = datetime.strptime(df.Date[i],'%d/%m/%y %H:%M:%S')
        else:
            df.drop(i, inplace=True)

  
    df['Date'] = df['Date'].apply(pd.to_datetime)
    
    # Extact Year
    df['Year'] = df.apply(lambda row: row.Date.year, axis = 1)
    # Extact Mont    
    df['Month'] = df.apply(lambda row: row.Date.month, axis = 1)
    
    # Extact Hour 
    df['Hour'] = df.apply(lambda row: row.Date.hour, axis = 1)
    # Extact Day of the Week
    df['Day_of_Week'] = df.apply(lambda row: row.Date.weekday(), axis = 1)
    
    # Sort values by date to keep order
    df.sort_values('Date', inplace=True)
    
    return df
    
    
def count_emojis(df, non_unicode = False):
    """ Calculates how often emojis are used between users
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users
    non_unicode : boolean, default False
        Whether to count the non-unicode emojis or not
        
    Returns:
    --------
    emoji_all : dictionary of Counters
        Indicating which emojis are often used by which user
    """
    
    emoji_all = {}

    # Count all "actual" emojis and not yet text smileys
    for user in df.User.unique():
        # Count all sets of emojis
        temp_user = df.Emoji[(df.User == user) & (df.Emoji_Count < 20)].value_counts().to_dict()
        emoji_all[user] = {}

        # Go over all set of emojis
        for emojis, count in temp_user.items():
            
            # Create a list of emojis
            emojis = regex.findall(r'\p{So}\p{Sk}*', emojis)

            # Loop over individual emojis
            for emoji_value in emojis:

                # Skip empty values
                if emoji_value != '':     
                    try:
                        emoji_all[user][emoji_value] += count
                    except:
                        emoji_all[user][emoji_value] = count

                        
    # Count non-unicode smileys
    if non_unicode:
        for user in df.User.unique():
            # Loop over
            for _, row in df[(df.User == user) & (df.Different_Emojis.str.len() > 0)].iterrows():
                for some_emoji in row.Different_Emojis:
                    if len(some_emoji) > 1:
                        try:
                            emoji_all[user][some_emoji] += 1
                        except:
                            emoji_all[user][some_emoji] = 1        

    return emoji_all

def extract_emojis(str):
    """ Used to extract emojis from a string using the emoji package
    """
    return ''.join(c for c in str if c in emoji_package.UNICODE_EMOJI)

def prepare_data(df):
    """ Prepares the data by extracting and 
    counting emojis per row (per message).
    
    New columns:
    * Emoji - List of emojis that are in the message
    * Emoji_count - Number of emojis used in a message
    * Different_Emojis - Number of unique emojis used in a message
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users
    Returns:
    --------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users with
        the added columns containing information about emoji use
    
    """
    
    # Extract unicode emojis per message and count them
    df['Emoji'] = df.apply(lambda row: extract_emojis(str(row.Message_Clean)), 
                                       axis = 1)
    df['Emoji_Count'] = df.apply(lambda row: len(regex.findall(r'\p{So}\p{Sk}*', 
                                                                           row.Emoji)), axis = 1)
    
    # Find non-unicode smileys
    eyes, noses, mouths = r":;8BX=", r"-~'^", r")(/\|DPp"
    pattern = "[%s][%s]?[%s]" % tuple(map(re.escape, [eyes, noses, mouths]))
    df['Different_Emojis'] = df.apply(lambda row: re.findall(pattern, str(row.Message_Clean)), 
                                                  axis=1)
    
    return df


@st.cache(allow_output_mutation=True)
def read_files():
   df = import_data('./chatt.txt')
   df = preprocess_data(df)
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
st.pyplot(plot_messages(filter(da,any,mes), colors=None, trendline=False, savefig=False, dpi=100))
# Dies
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Comunicació per dies</h1>", unsafe_allow_html=True)
st.write("# ")
st.pyplot(plot_day_spider(filter(da,any,mes), colors=None, savefig=False, dpi=100))
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
st.pyplot(plot_active_hours(filter(da,any,mes), color='#ed6d9b', savefig=False, dpi=100, user='Àlex'))
st.pyplot(plot_active_hours(filter(da,any,mes), color='#D0D3DA', savefig=False, dpi=100, user='Ge'))

st.write("# ")


temp = df[['index', 'Message_Raw', 'User', 'Message_Clean', 'Message_Only_Text']].copy()
temp = prepare_data(temp)



# Count all emojis
st.write("# ")
st.markdown("<h2 style='text-align: center; color: #D0D3DA;font-family:verdana;font-size:150%;'>Emoji World</h1>", unsafe_allow_html=True)
st.write("# ")
st.write('Els emojis que mes ha enviar la Gemma són: ' )
st.write("# ")

counts = count_emojis(temp, non_unicode=True)
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






