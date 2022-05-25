import nltk
import sys
import string

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | DP VP | NP VP VP
NP -> N | AP NP | N PP | N Conj NP | N Adv
PP -> P NP | P DP
DP -> Det NP
AP -> Adj | Adj AP
VP -> V | V DP | V NP | V PP | Conj VP | V Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokenized = nltk.tokenize.word_tokenize(sentence)

    tokens = []
    for word in tokenized:
        word = word.lower()
        valid = False
        # If any character of the word is an alphabetical letter, the word is valid
        for char in word:
            if char in string.ascii_lowercase:
                valid = True
                break
        # Add lowercase versions of valid words to tokens list
        if valid:
            tokens.append(word)

    return tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # Start with height 3 because it is the lowest possible height where a tree can be a noun phrase
    # Height 2 trees are nouns, verbs, adjs, etc
    height = 3
    chunks = []
    while height < tree.height():
        # Go through all subtrees of tree that are a certain height
        for subtree in tree.subtrees(lambda t: t.height() == height):
            npChunk = False
            if subtree.label() == 'NP':
                npChunk = True
                # If the subtree in question is a noun phrase, go through all strees (subtrees) of that subtree
                for stree in subtree.subtrees(lambda t: t.height() < subtree.height()):
                    # If any strees are a noun phrase, the subtree is not a chunk, so move on to next subtree
                    if stree.label() == 'NP':
                        npChunk = False
                        break
            # If no strees of the subtree in question are noun phrases and the subtree is a noun phrase itself, then it is a chunk
            if npChunk:
                chunks.append(subtree)
        # Increase in height until the sentence tree
        height += 1

    return chunks


if __name__ == "__main__":
    main()
