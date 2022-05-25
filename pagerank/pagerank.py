import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create probability value for each page in corpus
    probabilities = {}
    for key in corpus:
        probabilities[key] = 0

    # If the page links to others
    if len(corpus[page]) > 0:
        # Probability of selecting any page in corpus (.15 / amount of pages)
        anyProbability = (1 - damping_factor) / len(corpus)
        # Probability of selecting a link from the current page (.85 / amount of links)
        linkedProbability = damping_factor / len(corpus[page])
        for pg in corpus:
            # If page links to another page, add linkedProbability to the probability of choosing linked page
            if pg in corpus[page]:
                probabilities[pg] += linkedProbability
            # Every page should have chance of being selected randomly
            probabilities[pg] += anyProbability

    # If page has no links
    else:
        # Probabilities for choosing next page are just 1 / N for each page
        anyProbability = 1 / len(corpus)
        for pg in corpus:
            probabilities[pg] += anyProbability

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = []
    visitCounters = {}
    for key in corpus:
        pages.append(key)
        # Keep track of amount of times each page is visited/sampled
        visitCounters[key] = 0
    # Choose first page randomly
    sample = random.choice(pages)

    for _ in range(n):
        # Get transition model probabilities
        model = transition_model(corpus, sample, damping_factor)
        ranges = {}
        total = 0
        # Assign each page a range of decimals between 0 and 1 corresponding to the probability
        # that page will be chosen
        # Ex: model = {'1.html': 0.25, '2.html': 0.75} --> ranges['1.html'] = (0, 0.25) ranges['2.html'] = (0.25, 1)
        for page in model:
            ranges[page] = (total, total + model[page])
            total += model[page]
        # Get a random decimal between 0 and 1, the sample becomes the page thats range includes the decimal
        # Continue ex from above: number = 0.5 --> sample = '2.html', number = 0.1 --> sample = '1.html'
        number = random.random()
        for page in ranges:
            if number >= ranges[page][0] and number < ranges[page][1]:
                sample = page
                visitCounters[page] += 1
                break
    
    # Page rank for each page is number of visits it got divided by number of visits to all pages
    pageRanks = {}
    for page in corpus:
        pageRanks[page] = visitCounters[page] / n

    return pageRanks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Start with all pageranks as 1 / N
    pageRanks = {}
    for page in corpus:
        pageRanks[page] = 1 / len(corpus)
    significantChange = True

    # While at least one rank changes by more than .001
    while significantChange:
        newRanks = []
        # Create new ranks for each page
        for page in pageRanks:
            linkProbability = 0
            for pg in corpus:
                # Sum up the quotient of the pagerank divided by the amount of links on the page for all
                # pages that link to the page being evaluated
                if page in corpus[pg]:
                    linkProbability += pageRanks[pg] / len(corpus[pg])
            # New rank according to formula
            newRank = ((1 - damping_factor) / len(corpus)) + (damping_factor * linkProbability)
            newRanks.append(newRank)

        # Go through list of new ranks and determine if any changed by more than .001
        counter = 0
        for page in pageRanks:
            if newRanks[counter] <= pageRanks[page] + .001 and newRanks[counter] >= pageRanks[page] - .001:
                # Break while loop if no significant change for any page
                significantChange = False
            else:
                significantChange = True
                break
            counter += 1
        
        # Set pageranks equal to new pageranks
        counter = 0
        for page in pageRanks:
            pageRanks[page] = newRanks[counter]
            counter += 1
    
    return pageRanks


if __name__ == "__main__":
    main()
