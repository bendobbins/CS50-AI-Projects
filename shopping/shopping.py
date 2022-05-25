import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

MONTHNUMS = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "June": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11,
}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename) as f:
        # Read each row of the csv file into a list of lists, skip first line
        userData = csv.reader(f)
        next(userData)
        # Indexes in each list of user data that need to be converted to ints/floats
        ints = [0, 2, 4, 11, 12, 13, 14]
        floats = [1, 3, 5, 6, 7, 8, 9]

        for user in userData:
            # Get the label for the user
            labelTF = user.pop()
            label = 1 if labelTF == "TRUE" else 0

            try:
                # Convert int and float values, raise exception if conversion fails
                for column in ints:
                    user[column] = int(user[column])
                for column in floats:
                    user[column] = float(user[column])
            except ValueError:
                raise Exception("Invalid shopping data")
            
            # Convert months, visitortype and weekend to ints
            user[10] = MONTHNUMS[user[10]]
            user[15] = 0 if user[15] == "New_Visitor" else 1
            user[16] = 0 if user[16] == "FALSE" else 1

            # Add user evidence to evidence list and label to labels list
            evidence.append(user)
            labels.append(label)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return KNeighborsClassifier(n_neighbors=1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    truePositivesAmt = 0
    trueNegativesAmt = 0
    predictPositivesAmt = 0
    predictNegativesAmt = 0
    for i in range(len(labels)):
        # Always add to truePositives if labels value is true
        if labels[i] == 1:
            truePositivesAmt += 1
            # Only add to predictPositives if predictions value is true
            if predictions[i] == 1:
                predictPositivesAmt += 1
        # Same as above for negatives
        else:
            trueNegativesAmt += 1
            if predictions[i] == 0:
                predictNegativesAmt += 1

    # Divide amounts of predicted positives/negatives by amounts of true positives/negatives
    sensitivity = float(predictPositivesAmt) / truePositivesAmt
    specificity = float(predictNegativesAmt) / trueNegativesAmt

    return sensitivity, specificity


if __name__ == "__main__":
    main()
