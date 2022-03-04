import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialize joint probability
    joint = 1

    # Create joint probability
    for person in people:
        if person in one_gene:
            joint = joint_helper(1, one_gene, have_trait, two_genes, people[person], joint)
        elif person in two_genes:
            joint = joint_helper(2, one_gene, have_trait, two_genes, people[person], joint)
        else:
            joint = joint_helper(0, one_gene, have_trait, two_genes, people[person], joint)
    
    return joint


def joint_helper(gene, one_gene, have_trait, two_genes, person, joint):
    """
    Accepts int number of genes, one_gene set, have_trait set, two_genes set, person dict and joint int.
    Uses that info to calculate the probability that person will have gene number of genes.
    Multiplies that probability with joint and returns joint.
    """
    geneProb = None
    hasTrait = False

    # No parents in database
    if not person["mother"]:
        # Probability random person will have gene amount of genes
        geneProb = PROBS["gene"][gene]
        if person["name"] in have_trait:
            hasTrait = True
    # Parents in database
    else:
        # Get probability parents will pass gene number of genes on to child
        geneProb = inheritance_probability(one_gene, two_genes, gene, person["mother"], person["father"])
        if person["name"] in have_trait:
             hasTrait = True

    # Multiply the probability of the person having gene number of genes with the probability of them displaying the trait or not
    # with that amount of genes (depending on whether they are in have_traits or not)
    joint *= geneProb * PROBS["trait"][gene][hasTrait]

    return joint


def inheritance_probability(one_gene, two_genes, child_gene, mother, father):
    """
    Returns the probability of parents passing on a certain number of genes to their child based on how many genes the
    parents supposedly possess.
    """
    # If mother has two genes, probability of passing on without mutation is .99
    if mother in two_genes:
        mInheritance = 0.99
    # I left probability at .5 for one gene since it could get passed on and mutate out of the gene, but also not get passed on and mutate into the gene
    elif mother in one_gene:
        mInheritance = 0.50
    # If mother has no genes, probability of passing on is .01 (mutation)
    else:
        mInheritance = 0.01
    if father in two_genes:
        fInheritance = 0.99
    elif father in one_gene:
        fInheritance = 0.50
    else:
        fInheritance = 0.01

    # If child gene is 0, get probability that both parents will not pass the gene on (1 - the probability they will pass it on)
    if child_gene == 0:
        inheritance = (1 - fInheritance) * (1 - mInheritance)
    # If child gene is 1, get probability that either parent will pass the gene on
    elif child_gene == 1:
        inheritance = ((1 - fInheritance) * mInheritance) + (fInheritance * (1 - mInheritance))
    # If child gene is 2, get probability that both parents will pass the gene on
    else:
        inheritance = fInheritance * mInheritance
    
    return inheritance


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][1] += p
        else:
            probabilities[person]["trait"][0] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Sum probabilities together and divide each probability by the sum to normalize
    for person in probabilities:
        traitProbs = [probabilities[person]["trait"][0], probabilities[person]["trait"][1]]
        for i in range(len(traitProbs)):
            probabilities[person]["trait"][i] = traitProbs[i] / sum(traitProbs)

        geneProbs = [probabilities[person]["gene"][0], probabilities[person]["gene"][1], probabilities[person]["gene"][2]]
        for i in range(len(geneProbs)):
            probabilities[person]["gene"][i] = geneProbs[i] / sum(geneProbs)


if __name__ == "__main__":
    main()
