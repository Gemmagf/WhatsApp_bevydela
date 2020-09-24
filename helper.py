import re
import pandas   as pd
import numpy    as np
import streamlit as st
from datetime import datetime

from collections     import Counter


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
    
    