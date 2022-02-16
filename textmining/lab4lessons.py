import nltk
import regex
from collections import defaultdict


def tutorial():
    with open("mytext.txt", mode="r") as f:
        text = f.read()
        print("***** Text in the Document *****")
        print(text)

        text = regex.sub(r"[.,?!:;()]|\d+", "", text)
        print("***** Text after punctuations removed *****")
        print(text)

        tokens = nltk.word_tokenize(text)
        print("***** Tokens from the Document *****")
        print(type(tokens))
        print(tokens)

        i = 0
        while i < len(tokens):
            tokens[i] = tokens[i].lower()
            i += 1

        print("***** Tokens in Lowercase *****")
        print(tokens)

    vocabs = []
    with open("wordnetNoun.txt", "r") as nouns, open(
            "wordnetVerb.txt", "r") as verbs, open(
        "wordnetAdj.txt", "r") as adjs, open(
        "wordnetAdv.txt", "r") as adv:

        for line in nouns:
            vocabs.append(line.strip().split()[0])

        for line in verbs:
            vocabs.append(line.strip().split()[0])

        for line in adjs:
            vocabs.append(line.strip().split()[0])

        for line in adv:
            vocabs.append(line.strip().split()[0])

    print(f"Total vocabs length: {len(vocabs)}")

    validated_bag = []
    for t in tokens:
        if t in vocabs:
            validated_bag.append(t)

    print(f"Number of valid words: {len(validated_bag)}")
    print("***** Tokens after validation *****")
    print(validated_bag)

    porter_stemmer = nltk.PorterStemmer()
    stemmed_bag = [porter_stemmer.stem(v) for v in validated_bag]
    print("***** Tokens after Stemming *****")
    print(stemmed_bag)

    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_bag = [lemmatizer.lemmatize(v) for v in validated_bag]
    print("***** Tokens after Lemmatization *****")
    print(lemmatized_bag)

    pos_tagged_bag = nltk.pos_tag(validated_bag)
    print("***** Tokens with parts-of-speech tagging *****")
    print(pos_tagged_bag)

    tag_frequencies = defaultdict(int)
    for v in validated_bag:
        tag_frequencies[v] += 1

    print("***** Tokens with tag frequencies *****")
    print(tag_frequencies.items())

    stop_words = set(nltk.corpus.stopwords.words("english"))
    validated_bag_no_stop = []
    for v in validated_bag:
        if v not in stop_words:
            validated_bag_no_stop.append(v)
        else:
            print(f"Stop word: {v}")

    print("***** Tokens without stop words *****")
    print(validated_bag_no_stop)

    max_freq = 0
    freq_term = ''
    for key in tag_frequencies.keys():
        if tag_frequencies[key] > max_freq:
            max_freq = tag_frequencies[key]
            freq_term = key

    print(f"Most frequent term(before removing stop words): {freq_term} - with frequency: {max_freq}")

    tag_frequencies = defaultdict(int)
    for v in validated_bag_no_stop:
        tag_frequencies[v] += 1

    max_freq = 0
    freq_term = ''
    for key in tag_frequencies.keys():
        if tag_frequencies[key] > max_freq:
            max_freq = tag_frequencies[key]
            freq_term = key

    print(f"Most frequent term(after removing stop words): {freq_term} - with frequency: {max_freq}")

    print(f"Finally remaining number of words: {len(validated_bag_no_stop)}")


if __name__ == "__main__":
    tutorial()
