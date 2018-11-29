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
    preprocessed_txt = list(filter(lambda x: x not in tags and
                                 x not in stopwords and
                                 x not in punctuation, txt))
    preprocessed_txt = list(map(lambda x: str2emoji(x), preprocessed_txt))
    return ' '.join(preprocessed_txt)
    