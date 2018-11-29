from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.noslang.slangdict import slangdict
from emoticons import str2emoji
from nltk.corpus import stopwords
from string import punctuation


stopwords = set(stopwords.words('english')) - set(('not', 'no'))

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
    return list(filter(lambda x: x not in tags and
                                 x not in stopwords and
                                 x not in punctuation, txt))

"""
sentences = [
    "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
    "@SentimentSymp: plz we'll can't wait for the Nov 9 25,000  25.000 #Sentiment talks!  YAAAAAAY !!! :-D :@ <3 http://sentimentsymposium.com/."
]

for s in sentences:
    preprocessed_txt = preprocess(s)
    preprocessed_txt = list(map(lambda x: str2emoji(x), preprocessed_txt))
    preprocessed_txt = ' '.join(preprocessed_txt)
    print(preprocessed_txt)
"""
