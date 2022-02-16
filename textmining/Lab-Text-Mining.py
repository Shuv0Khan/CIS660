# common libraries
import nltk
# nltk.download() select the entire set of book resources
from nltk import word_tokenize
from collections import defaultdict  # for term-freq calculations

# open an input and output file, with py built in file io
rawinputfile = open("rawtext.txt")
# fout = open("analysisResults.txt","w")
rawtext = rawinputfile.read()
rawinputfile.close()
print(rawtext)

# build a list containing tokens out of the whole file
tokens = word_tokenize(rawtext)

# check the type of the output returned by word_tokenize
print("word_tokenize returns a: ", end=" "),
print(type(tokens))

# how many indices in the list?
print("containing: " + str(len(tokens)) + " indices")

# return data from the start of the list until index 4
print("the list of data begins with: ", end=" "),
print(tokens[:4])

# convert tokens to lowercase terms by applying function to each item in a list (i.e., list comprehensions)
terms = [t.lower() for t in tokens]
print("Lowercase terms: ", end=" "),
print(terms)

# build a vocabulary of acceptable NOUNS found in WordNet
vocab = []
with open("wordnetNoun.txt") as WordNetinputfile:
    for line in WordNetinputfile:
        newTerm = line.split()
        vocab.append(newTerm[0])
print("The WordNet vocab starts with : ", end=" "),
print(vocab[0:10])
WordNetinputfile.close()
print("WordNet size: " + str(len(vocab)))

# build a vocabulary of acceptable VERBS found in WordNet
with open("wordnetVerb.txt") as WordNetinputfile:
    for line in WordNetinputfile:
        newTerm = line.split()
        vocab.append(newTerm[0])
WordNetinputfile.close()
print("After adding verbs: " + str(len(vocab)))

# build a vocabulary of acceptable ADJECTIVES found in WordNet
with open("wordnetAdj.txt") as WordNetinputfile:
    for line in WordNetinputfile:
        newTerm = line.split()
        vocab.append(newTerm[0])
WordNetinputfile.close()
print("After adding adjectives: " + str(len(vocab)))

# build a vocabulary of acceptable ADVERBS found in WordNet
with open("wordnetAdv.txt") as WordNetinputfile:
    for line in WordNetinputfile:
        newTerm = line.split()
        vocab.append(newTerm[0])
WordNetinputfile.close()
print("After adding adverbs: " + str(len(vocab)))

# limit our dataset to just valid words (defined by WordNet)
validatedBag = []
for t in terms:
    if t in vocab:
        validatedBag.append(t)
print("Our remaining validated terms are: ", end=" "),
print(validatedBag)

# apply the Porter stemmer - to each index of our list
porter = nltk.PorterStemmer()
stemmedBag = [porter.stem(v) for v in validatedBag]
print("The stems of our terms are: ", end=" "),
print(stemmedBag)

# alternatively, apply the WordNet lemmatizer to each index of our list
lemmatizer = nltk.WordNetLemmatizer()
lemmatizedBag = [lemmatizer.lemmatize(v) for v in validatedBag]
print("The lemmas of our terms are: ", end=" "),
print(lemmatizedBag)

# POS tagging based on Penn Treebank tags (Univ. Penn 1992)
POSTaggedBag = nltk.pos_tag(validatedBag)
print("The parts of speech of our terms are: ", end=" "),
print(POSTaggedBag)

# calculate the term frequencies using an inverted index/dictionary
tf = defaultdict(int)
for v in validatedBag:
    tf[v] += 1
# print(tf.keys())
# print(tf.values())
print("The term-frequencies of our terms are: ", end=" "),
print(tf.items())

# remove common stopwords
stopWords = set(nltk.corpus.stopwords.words('english'))
validatedBagNoStop = []
for t in validatedBag:
    if t not in stopWords:
        validatedBagNoStop.append(t)
    if t in stopWords:
        print("removed term: " + t)
print("Our validated terms without stopwords are: ", end=" "),
print(validatedBagNoStop)
# validatedBag = validatedBagNoStop


# fout.close()
# fout.write(str(float(content[i][0])*9/5+32)+'\n')
