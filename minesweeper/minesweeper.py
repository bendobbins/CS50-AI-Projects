import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # Know cells are mines if the number of mines in the cells equals the number of cells
        if self.count == len(self.cells):
            return self.cells
        return None

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # Know cells are safe if the number of mines in them is 0
        if self.count == 0:
            return self.cells
        return None

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # Remove cell from sentence and reduce count
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # Only remove cell from sentence
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        def check_safes_and_mines():
            """
            Marks safe spaces and mine spaces if possible based on current knowledge base.
            Returns all sentences in knowledge base with non-empty cell sets after marking spaces.
            """
            aliveSentences = []
            for sentence in self.knowledge:
                # Check if sentence proves safe spaces
                safes = sentence.known_safes()
                if safes:
                    # Create a new list of same cells from sentence to avoid editing sentence while iterating over it
                    iterSafes = []
                    for space in safes:
                        iterSafes.append(space)
                    # Mark safe spaces
                    for space in iterSafes:
                        self.mark_safe(space)

                # Check if sentence proves mines
                mines = sentence.known_mines()
                if mines:
                    # Create a new list of same cells from sentence to avoid editing sentence while iterating over it
                    iterMines = []
                    for mine in mines:
                        iterMines.append(mine)
                    # Mark mines
                    for mine in iterMines:
                        self.mark_mine(mine)
                # Get rid of any sentences with no cells after marking spaces
                if len(sentence.cells) != 0:
                    aliveSentences.append(sentence)
            # Return all sentences that still have cells
            return aliveSentences

        # All neighbors of cell that was chosen
        neighbors = [(cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1), 
        (cell[0] + 1, cell[1] + 1), (cell[0] - 1, cell[1] + 1), (cell[0] + 1, cell[1] - 1), (cell[0] - 1, cell[1] - 1)]

        # Add chosen cell to moves made, mark it as safe
        self.moves_made.add(cell)
        self.mark_safe(cell)
        
        # Make a list of all neighbors that are not mines, safe spaces, or moves made
        undeterminedCells = []
        for neighbor in neighbors:
            if neighbor[0] < 8 and neighbor[1] < 8 and neighbor[0] >= 0 and neighbor[1] >= 0:
                if neighbor not in self.mines and neighbor not in self.safes and neighbor not in self.moves_made:
                    undeterminedCells.append(neighbor)
                # If one neighbor is a mine, other undetermined neighbors have one less mine
                elif neighbor in self.mines:
                    count -= 1

        # Add new sentence to knowledge with information about undetermined neighbors and their number of mines
        self.knowledge.append(Sentence(undeterminedCells, count))

        # Check if any sentences prove where mines/safe spaces are
        self.knowledge = check_safes_and_mines()

        newSentences = []
        overwritten = []
        # If one sentence is a subset of another, create a new sentence based on the rules in the Background section
        for sentence in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence != sentence2:
                    if sentence.cells.issubset(sentence2.cells):
                        newCount = sentence2.count - sentence.count
                        newCells = []
                        for newCell in sentence2.cells:
                            if newCell not in sentence.cells:
                                newCells.append(newCell)
                        overwritten.append(sentence2)
                        newSentences.append((newCells, newCount))
        
        # Remove sentences with repeating/old knowledge from KB
        for sentence in overwritten:
            if sentence in self.knowledge:
                self.knowledge.remove(sentence)
        
        # Add sentences with new knowledge deduced from subsets to KB
        for newSentence in newSentences:
            self.knowledge.append(Sentence(newSentence[0], newSentence[1]))
        
        # After adding new sentences, check again to see if any prove where mines/safe spaces are
        self.knowledge = check_safes_and_mines()

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # Space is safe move if in safe spaces and not a previously made move
        for space in self.safes:
            if space not in self.moves_made:
                return space
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        possibleMoves = []
        counter = 0
        for i in range(8):
            for j in range(8):
                # Every space that is not a mine and is not a previously made move is a possible move
                if (i, j) not in self.mines and (i, j) not in self.moves_made:
                    possibleMoves.append((i, j))
                    counter += 1
        # If there are any possible moves, return one at random, else return None
        if possibleMoves:
            return possibleMoves.pop(random.randint(0, counter - 1))
        return None
