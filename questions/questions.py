import nltk
import sys
import os
import string
import math

FILE_MATCHES = 2
SENTENCE_MATCHES = 3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    # Path to corpus directory
    corpus = os.path.join(os.getcwd(), directory)

    for document in os.listdir(corpus):
        # Open each document in the corpus
        with open(os.path.join(corpus, document)) as f:
            # Read first two lines which are just hyperlink and blank in all docs
            for _ in range(2):
                f.readline()
            text = f.read()
            # Map string of all text in document to document name in files dict
            files[document] = text

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized = nltk.tokenize.word_tokenize(document)

    tokens = []
    for word in tokenized:
        word = word.lower()
        # If word is punctuation or is a stopword, do not add to final list
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            tokens.append(word)

    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    totalDocs = len(documents)

    for document in documents:
        # Loop through all words in each document to find idfs for any and all words
        for word in documents[document]:
            # If idf value for word has already been calculated, skip
            if word in idfs:
                continue

            appearances = 1
            for doc in documents:
                # Count amt of documents word appears in
                if doc != document and word in documents[doc]:
                    appearances += 1
            # Idf for word is natural log of total amt of documents divided by amt of documents word appears in
            idfs[word] = math.log(totalDocs / appearances)
    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create a dictionary where each possible filename is mapped to a tfidf value (0 to start)
    tfidfs = {filename: 0 for filename in files}

    for keyword in query:
        for file in files:
            tf = 0
            # Loop through each word of the file, count how many times the keyword appears in the file
            for word in files[file]:
                if word == keyword:
                    tf += 1
            # If the keyword appears in the file, compute tfidf for the keyword in the file and add it to file's tfidf dict value
            if tf:
                tfidfs[file] += tf * idfs[keyword]
    
    # Sort dictionary into list of tuples where each tuple has a filename and that file's tfidf value
    # Tuples will be in descending order based on file tfidf values
    sortedTfidfs = sorted(tfidfs.items(), key=lambda x: x[1], reverse=True)

    topFiles = []
    # Return list of files with top n tfidf values (descending order)
    for i in range(n):
        topFiles.append(sortedTfidfs[i][0])

    return topFiles


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create dictionaries where each sentence is mapped to idf values and qtd values respectively (0 to start)
    idfRanks = {sentence: 0 for sentence in sentences}
    densityRanks = {sentence: 0 for sentence in sentences}

    for keyword in query:
        for sentence in sentences:
            idfCounted = False
            for word in sentences[sentence]:
                if word == keyword:
                    # Keep track of amt of keywords in each sentence
                    densityRanks[sentence] += 1
                    if not idfCounted:
                        # If keyword appears in sentence, add keyword idf value to sentence idf value
                        idfRanks[sentence] += idfs[word]
                        idfCounted = True

    # Create qtd values for each sentence by dividing the amt of keywords in the sentence by the length of the sentence
    for sentence in densityRanks:
        densityRanks[sentence] /= len(sentence.split(" "))

    # Sort idf dictionary into list of tuples where each tuple has a sentence and that sentence's idf value
    # Tuples will be in descending order based on sentence idf values
    sortedIdfRanks = sorted(idfRanks.items(), key=lambda x: x[1], reverse=True)

    bestSentences = []
    index = 0
    # Create a list of n best sentences according to idf values
    # List may be longer than n if multiple sentences have same idf values
    # Ex: If n is 1 but 2 sentences both have highest idf values of 11, both sentences will be added so that they can be sorted by qtd in order_sentences
    # List will be of tuples where each tuple consists of sentence, sentence idf value and sentence qtd value
    while len(bestSentences) < n:
        bestIdf = sortedIdfRanks[index][1]
        bestSentences.append((sortedIdfRanks[index][0], sortedIdfRanks[index][1], densityRanks[sortedIdfRanks[index][0]]))
        index += 1
        while sortedIdfRanks[index][1] == bestIdf:
            bestSentences.append((sortedIdfRanks[index][0], sortedIdfRanks[index][1], densityRanks[sortedIdfRanks[index][0]]))
            index += 1

    return order_sentences(bestSentences, n)


def order_sentences(sentences, n):
    """
    Accepts a list of tuples where each tuple contains a sentence, its idf value, and its qtd value.
    Returns list of sentences of length n where sentences are ordered primarily by idf value and secondarily by qtd value.
    """
    orderedSentences = []
    while sentences:
        idf = sentences[0][1]
        sameIdfs = []
        # Create lists of sentences with same idfs
        while sentences and sentences[0][1] == idf:
            sameIdfs.append(sentences.pop(0))
        # Sort sentences with same idf by qtd, then add them into ordered sentences list
        sortedByqtd = sorted(sameIdfs, key=lambda x: x[2], reverse=True)
        for sentence in sortedByqtd:
            orderedSentences.append(sentence[0])
    
    # Resulting list is sorted by idf primarily and qtd amongst sentences with same idf, so the top n sentences
    # are the top matches for the query
    return orderedSentences[:n]


if __name__ == "__main__":
    main()
