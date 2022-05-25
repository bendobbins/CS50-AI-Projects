import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            removes = []
            for word in self.domains[var]:
                # If a word in the domain does not have the same length as the variable, add it to removes
                if var.length != len(word):
                    removes.append(word)
            # Remove all words in removes from the domain of the variable (Do this here to not mess up for loop above)
            for word in removes:
                self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        if not self.crossword.overlaps[x, y]:
            return revised
        
        removes = set()
        for word in self.domains[x]:
            hasMatch = False
            for aword in self.domains[y]:
                # If the word in the x domain has a match in the y domain, break the loop and move to the next x domain word
                if word[self.crossword.overlaps[x, y][0]] == aword[self.crossword.overlaps[x, y][1]]:
                    hasMatch = True
                    break
            # If a word in the x domain has no match in the y domain, add the word to removes and set revised to true (will remove that word)
            if not hasMatch:
                removes.add(word)
                revised = True
        
        # Remove words from x domain that don't match any in y domain
        for word in removes:
            self.domains[x].remove(word)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Create list of initial arcs, avoiding duplicates
        if not arcs:
            arcs = []
            for var in self.crossword.variables:
                for var2 in self.crossword.variables:
                    if var != var2:
                        if (var2, var) not in arcs:
                            arcs.append((var, var2))

        # Revise each arc in the list, adding new arcs between any variables with revised domains and their neighbors
        while arcs:
            arc = arcs.pop()
            if self.revise(arc[0], arc[1]):
                if len(self.domains[arc[0]]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(arc[0]):
                    if neighbor != arc[1]:
                        arcs.append((neighbor, arc[0]))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        allVariables = self.crossword.variables.copy()
        for variable in assignment:
            allVariables.remove(variable)
        
        # If all variable spots are filled and there are no mistakes, puzzle solved
        if not len(allVariables) and self.consistent(assignment):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # For all neighbors of each variable in assignment, return false if there is a bad overlap at a meeting point
        distinct = set()
        for variable in assignment:
            # If value is the wrong length, not consistent
            if len(assignment[variable]) != variable.length:
                return False
            distinct.add(assignment[variable])
            neighbors = self.crossword.neighbors(variable)
            for neighbor in neighbors:
                if neighbor in assignment:
                    if assignment[variable][self.crossword.overlaps[variable, neighbor][0]]\
                    != assignment[neighbor][self.crossword.overlaps[variable, neighbor][1]]:
                        return False

        # If every value in the crossword is not distinct, it is not consistent
        if len(distinct) != len(assignment):
            return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbors = self.crossword.neighbors(var)
        sortingList = []
        for word in self.domains[var]:
            numRuledOut = 0
            for neighbor in neighbors:
                if neighbor not in assignment:
                    for neighborWord in self.domains[neighbor]:
                        # If word rules out neighborWord, add to numRuledOut
                        if word[self.crossword.overlaps[var, neighbor][0]]\
                        != neighborWord[self.crossword.overlaps[var, neighbor][1]]:
                            numRuledOut += 1
            # Make a list of the words in the domain of var and how many neighbor words they rule out
            sortingList.append((word, numRuledOut)) 

        # Sort list according to numRuledOut and create list of only words
        sortingList.sort(key=lambda wordAndNum: wordAndNum[1])
        orderedDomains = []
        for word in sortingList:
            orderedDomains.append(word[0])
        
        return orderedDomains

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        variables = self.crossword.variables.copy()
        shortest = None
        unassigned = []
        for variable in variables:
            if variable not in assignment:
                unassigned.append(variable)
                # Find the shortest domain length for any variable
                if not shortest:
                    shortest = len(self.domains[variable])
                else:
                    if len(self.domains[variable]) < shortest:
                        shortest = len(self.domains[variable])

        bestVars = []
        # Make a list of variables with the shortest domain length and their number of neighbors
        for variable in unassigned:
            if len(self.domains[variable]) == shortest:
                bestVars.append((variable, len(self.crossword.neighbors(variable))))
        
        # Sort list by number of neighbors and return first value
        bestVars.sort(key=lambda varAndNbrAmt: varAndNbrAmt[1])
        bestVars.reverse()

        return bestVars[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # If assignment is complete, return it
        if self.assignment_complete(assignment):
            return assignment
        variable = self.select_unassigned_variable(assignment)
        for word in self.domains[variable]:
            # Try putting a word in puzzle and checking consistency
            assignment[variable] = word
            if self.consistent(assignment):
                # Keep adding words to puzzle and checking consistency until puzzle is complete or there is a mistake
                result = self.backtrack(assignment)
                if self.assignment_complete(result):
                    return result
            # If puzzle is not consistent, remove word
            assignment.pop(variable)
        return assignment


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
