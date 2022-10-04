class State:
    """Representa um estado do problema.
    Neste caso, é um par ordenado que representa a linha e a coluna onde se encontra o agente no labirinto."""

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def setRowCol(self, row, col):
        # Esse método não é necessário em Python, pois os atributos podem ser acessados diretamente
        self.row = row
        self.col = col

    def __eq__(self, other):
        try:
            return self.row == other.row and self.col == other.col
        except AttributeError:
            return False
        except:
            raise

    def __ne__(self, other):
        try:
            return self.row != other.row or self.col != other.col
        except AttributeError:
            return False
        except:
            raise

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        # Permite fazer um print(state) diretamente
        return "({0:d}, {1:d})".format(self.row, self.col)
