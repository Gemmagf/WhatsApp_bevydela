import random
import numpy       as np
from PIL                                   import Image
from wordcloud                             import WordCloud





def create_wordcloud(data, cmap=None, savefig=False, name=None, **kwargs):
    """ Creates a wordcloud based on a text document
    
    Parameters:
    -----------
    data : list, pandas series or dictionary
        List of words, pandas series of messages
        or dictionary that includes words with 
        frequency. If a dictionary is supplied
    cmap : str, default None
        A string that relates that a matplotlib
        color map. See:
        https://matplotlib.org/examples/color/colormaps_reference.html
    savefig : boolean, default False
        Whether or not to save the file, if True
        then it will be saved in the current
        working directory        
        
    Returns:
    --------
    PIL.Image.Image
        An image as presented in PIL (Pillow)
    """
    
    stop_words = open('catala.txt','r',encoding = 'utf-8').read().split()
    
    # Create Word Cloud
    wc = WordCloud(background_color="white", mode='RGBA',stopwords = stop_words, **kwargs)
    
    if type(data) != dict:
        text = ' '.join(data)
        wc.generate_from_text(text)
    else:
        wc.generate_from_frequencies(data)
    
    if not cmap:
        wc.recolor(color_func=color_func, random_state=kwargs['random_state'])
    else:
        wc.recolor(colormap=cmap, random_state=kwargs['random_state'])
        
    if savefig:
        wc.to_file(f"results/wordcloud_{name}.png")
    else:
        return wc.to_image()

def extract_sentiment_count(counts, user):
    """ Extract and return counts of negative and positive words
    Positivity is based on the Pattern package which gives
    sentiment values between -1 (negative) and 1 (positive). 
    Words with a sentiment > 0 are positive and words
    with a sentiment < 0 are negative. All others are classified
    as being neutral. 
    
    Parameters:
    -----------
    counts : pandas dataframe
        Dataframe that contains a count of how often
        a user has used a word. 
    user : str
        The user for which the sentiment count is extracted
        
    Returns:
    --------
    positive, negative : dictionary
        Contains counts of positive words used and
        counts of negative words used
    """
    
    counts_dict = counts[['Word', user]].set_index('Word').to_dict()[user]
    counts_dict = {key: value for key, value in counts_dict.items() if value > 0}

    positive = {}
    negative = {}

    for word in counts_dict.keys():
        if sentiment_nl(word)[0] < 0:
            negative[word] = counts_dict[word]
        if sentiment_nl(word)[0] > 0:
            positive[word] = counts_dict[word]
            
    return positive, negative