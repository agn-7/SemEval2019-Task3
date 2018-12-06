from emoticons import str2emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from tokenizer import tokenizer
import re


stopwords = set(stopwords.words('english')) - set(('not', 'no'))


def contraction_removal(tweet):
    #replace uniquote to ascii quote
    tweet = re.sub(r"\u2019", "'", tweet)
    tweet = re.sub(r"\u002c", "'", tweet)

    #contractions
    tweet = re.sub(r"u r "," you are ",tweet)
    tweet = re.sub(r"U r "," you are ",tweet)
    tweet = re.sub(r" u(\s|$)"," you ",tweet)
    tweet = re.sub(r"didnt","did not",tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r" can\'t", " cannot", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'s", "", tweet)
    tweet = re.sub(r"\'n", "", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub(r" plz[\s|$]", " please ",tweet)

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
    L = WordNetLemmatizer()
    return list(map(L.lemmatize, tokenizedTweet))


def normalize(lemmatizedTweet):
    return list(filter(lambda x: x not in stopwords, lemmatizedTweet))


def preprocess(tweet):
    return normalize(lemmatize(tokenize(tweet)))