import pickle
import pandas as pd
from helpers import Embedder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import SnowballStemmer, word_tokenize, download
from nltk.corpus import stopwords
import re
from annoy import AnnoyIndex


class StemTokenizer:
    ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`"]

    def __init__(self):
        download("punkt")
        download("stopwords")
        self.stemmer = SnowballStemmer("english")

    def __call__(self, doc):
        doc = doc.lower()
        return [
            self.stemmer.stem(t)
            for t in word_tokenize(re.sub("[^a-z' ]", "", doc))
            if t not in self.ignore_tokens
        ]


class BagOfWordsEmbedder(Embedder):
    def __init__(self, df) -> None:
        super().__init__()
        self.dim = 3000
        self.preprocessor = StemTokenizer()
        stop_words = self.preprocessor(" ".join(stopwords.words("english")))
        self.model = TfidfVectorizer(
            stop_words=stop_words,
            tokenizer=self.preprocessor,
            max_features=self.dim,
        )
        self.df = df
        self.dataset = None

    def prepare(self, data):
        self.model.fit(data)

    def __call__(self, inputs):
        return self.model.transform(inputs).toarray()

    @classmethod
    def load_pretrained(cls, path, df=None):
        embedder = BagOfWordsEmbedder(df)
        with open(path, "rb") as f:
            embedder.model = pickle.load(f)
        return embedder

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def create_annoy_db(self, path):
        embeddings = self(self.df["overview"])
        embeddings = [e for e in embeddings]
        df = pd.DataFrame(
            {
                "title": self.df["original_title"],
                "overview": self.df["overview"],
            }
        )
        df.to_pickle("dataframes/feature-title-BoW.pickle")

        dim = self.dim
        annoy_index = AnnoyIndex(dim, "angular")
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)

        annoy_index.build(10)
        annoy_index.save(path)


if __name__ == "__main__":
    data = pd.read_csv("data/movies_metadata.csv")
    data = data[
        data["overview"].notna() & data["original_title"].notna()
    ].reset_index()
    # embedder = BagOfWordsEmbedder(data)
    # embedder.prepare(data["overview"])
    # embedder.save("tfidf.pickle")

    embedder = BagOfWordsEmbedder.load_pretrained("models/tfidf.pickle", data)
    embedder.create_annoy_db("annoy_indices/BoW_index.ann")
