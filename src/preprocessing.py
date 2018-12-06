#from ekphrasis.classes.preprocessor import TextPreProcessor
#from ekphrasis.classes.tokenizer import SocialTokenizer
#from ekphrasis.dicts.noslang.slangdict import slangdict
from emoticons import str2emoji
from nltk.corpus import stopwords
from string import punctuation
import nltk
from tokenizer import tokenizer
import re


stopwords = set(stopwords.words('english')) - set(('not', 'no'))

"""
tags = ['<url>', '<email>', '<user>', '<hashtag>', '</hashtag>',
        '<elongated>', '</elongated>', '<repeated>', '</repeated>']


text_processor = TextPreProcessor(
    normalize=['url', 'email', 'user'],
    annotate={'hashtag', 'elongated', 'repeated'},
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[slangdict]
)


def preprocess(text):
    txt = text_processor.pre_process_doc(text)
    preprocessed_txt = list(filter(lambda x: x not in tags and
                                 x not in stopwords and
                                 x not in punctuation, txt))
    preprocessed_txt = list(map(lambda x: str2emoji(x), preprocessed_txt))
    return ' '.join(preprocessed_txt)
"""

def contraction_removal(tweet):
    #replace uniquote to ascii quote
    tweet = re.sub(r"\u2019", "'", tweet)
    tweet = re.sub(r"\u002c", "'", tweet)

    #contractions
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r" can\'t", " cannot", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'s", "", tweet)
    tweet = re.sub(r"\'n", "", tweet)
    tweet = re.sub(r"\'m", " am", tweet)

    return tweet


def tokenize(tweet):
    #remove email
    tweet = re.sub('\S*@\S*\s?', '', tweet)
    #remove url
    tweet = re.sub(r'http\S+', '', tweet)

    tweet = tokenizer.TweetTokenizer(
        preserve_case=False, preserve_handles=False, preserve_hashes=False,
        regularize=True, preserve_emoji=True
    ).tokenize(tweet)

    #emoji processing
    tweet = list(map(lambda x: str2emoji(x), tweet))
    tweet = ' '.join(tweet)

    #remove contraction
    tweet = contraction_removal(tweet)

    #remove puntuation
    tweet = re.sub('[' + punctuation + ']', '', tweet).split(' ')
    tweet = list(filter(lambda x: x != u'', tweet))

    return tweet


def lemmatize(tokenizedTweet):
    L = nltk.stem.WordNetLemmatizer()
    return list(map(L.lemmatize, tokenizedTweet))


def normalize(lemmatizedTweet):
    return list(filter(lambda x: x not in stopwords, lemmatizedTweet))


def preprocess(tweet):
    return normalize(lemmatize(tokenize(tweet)))
