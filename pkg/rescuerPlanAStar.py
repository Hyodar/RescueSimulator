from state import State
from node import NodeType

possibilities = ["N", "S", "L", "O", "NE", "NO", "SE", "SO"]
movePos = {
    "N": (-1, 0),
    "S": (1, 0),
    "L": (0, 1),
    "O": (0, -1),
    "NE": (-1, 1),
    "NO": (-1, -1),
    "SE": (1, 1),
    "SO": (1, -1),
}
reverse = {
    "N": "S",
    "S": "N",
    "L": "O",
    "O": "L",
    "NE": "SO",
    "NO": "SE",
    "SE": "NO",
    "SO": "NE",
}


class RescuerPlanAStar:
    def __init__(
        self, maxRows, maxColumns, goal, initialState, discoveredMap: list, name="none", mesh="square"
    ):
        """
        Define as variaveis necessárias para a utilização do rescuer plan por um unico agente.
        """
        self.walls = [
            (row, col) for row,_ in enumerate(discoveredMap)
                for col,_ in enumerate(discoveredMap[row])
                    if discoveredMap[row][col].type == NodeType.OBSTACLE or discoveredMap[row][col].type == NodeType.UNKNOWN
        ]
        self.maxRows = maxRows
        self.maxColumns = maxColumns
        self.initialState = initialState
        self.currentState = initialState
        self.goalPos = goal
        self.actions = []
        self.previousState = initialState

        self.backtrack = []
        self.map = discoveredMap
        self.victims = [discoveredMap[row][col] for row,_ in enumerate(discoveredMap)
            for col,_ in enumerate(discoveredMap[row])
                if discoveredMap[row][col].type == NodeType.VICTIM
        ]

        self.victims.sort(key=lambda v: v.gravityLevel)

        self.distances = {
            victim.state: {
                victim2.state: pathCost(self.aStar(victim.state, victim2.state)) for victim2 in self.victims
            } for victim in self.victims
        }

        tmpVictims = []

        tmp = {}

        for victim in self.victims:
            if tmp.get(victim.state, False) == False:
                tmpVictims.append(victim)
                tmp[victim.state] = True

            for victim2 in self.victims:
                distance = self.distances[victim.state][victim2.state]

                if distance < 2 and tmp.get(victim2.state, False) == False:
                    tmpVictims.append(victim2)
                    tmp[victim2.state] = True

        self.victims = tmpVictims

        self.currentVictim = None
        self.finished = False

        self.counter = 0

    def updateCurrentState(self, state):
        self.currentState = state

    def goalTest(self):
        return self.finished

    def isPossibleToMove(self, fromState, toState):
        """Verifica se eh possivel ir da posicao atual para o estado (lin, col) considerando
        a posicao das paredes do labirinto e movimentos na diagonal
        @param toState: instancia da classe State - um par (lin, col) - que aqui indica a posicao futura
        @return: True quando é possivel ir do estado atual para o estado futuro"""

        if toState.col < 0 or toState.row < 0:
            return False

        if toState.col >= self.maxColumns or toState.row >= self.maxRows:
            return False

        if len(self.walls) == 0:
            return True

        if (toState.row, toState.col) in self.walls:
            return False

        delta_row = toState.row - fromState.row
        delta_col = toState.col - fromState.col

        if delta_row != 0 and delta_col != 0:
            if (fromState.row + delta_row, fromState.col,) in self.walls and (
                fromState.row,
                fromState.col + delta_col,
            ) in self.walls:
                return False

        return True

    def aStar(self, initial, target):
        def heuristic(state):
            return float("inf") if (state.row, state.col) in self.walls else 1

        def getNeighbors(state):
            resp = []

            for possibility in possibilities:
                nextState = State(
                    state.row + movePos[possibility][0],
                    state.col + movePos[possibility][1],
                )

                if self.isPossibleToMove(state, nextState):
                    weight = 1 if len(possibility) == 1 else 1.5
                    resp.append((nextState, weight))

            return resp

        openList = set([initial])
        closedList = set([])

        poo = [
            [float("inf") for j in range(self.maxColumns)] for i in range(self.maxRows)
        ]
        poo[initial.row][initial.col] = 0

        par = [[False for j in range(self.maxColumns)] for i in range(self.maxRows)]
        par[initial.row][initial.col] = initial

        while len(openList) > 0:
            n = None

            for v in openList:
                if n == None or poo[v.row][v.col] + heuristic(v) < poo[n.row][
                    n.col
                ] + heuristic(n):
                    n = v

            if n == None:
                return None

            if n == target:
                reconstructedPath = []
                while par[n.row][n.col] != n:
                    reconstructedPath.append(n)
                    n = par[n.row][n.col]

                reconstructedPath.append(initial)
                reconstructedPath.reverse()

                # print(
                #     f"Path found: {list(map(lambda a: (a.row, a.col), reconstructedPath))}"
                # )
                return reconstructedPath

            for (m, weight) in getNeighbors(n):
                if m not in openList and m not in closedList:
                    openList.add(m)
                    par[m.row][m.col] = n
                    poo[m.row][m.col] = poo[n.row][n.col] + weight
                else:
                    if poo[m.row][m.col] > poo[n.row][n.col] + weight:
                        poo[m.row][m.col] = poo[n.row][n.col] + weight
                        par[m.row][m.col] = n

                        if m in closedList:
                            closedList.remove(m)
                            openList.add(m)

            openList.remove(n)
            closedList.add(n)

        # print("Path does not exist")
        return None

    def chooseAction(self, tl):
        """Escolhe o proximo movimento de acordo com uma DFS online.
        Eh a acao que vai ser executada pelo agente.
        @return: tupla contendo a acao (direcao) e uma instância da classe State que representa a posição esperada após a execução
        """

        self.currentVictim = next((v for v in self.victims if v.type == NodeType.VICTIM), None)
        pathVictim = self.aStar(self.currentState, self.currentVictim.state if self.currentVictim else self.goalPos)
        pathGoal = self.aStar(self.currentState, self.goalPos)
        pathPostVictim = self.aStar(pathVictim[-1], self.goalPos)
        costVictim = 0

        for (idx, state) in enumerate(pathVictim[0:-1]):
            costVictim += 1
            if (
                state.row - pathVictim[idx + 1].row != 0
                and state.col - pathVictim[idx + 1].col != 0
            ):
                costVictim += 0.5

        costPostVictim = costVictim

        for (idx, state) in enumerate(pathPostVictim[0:-1]):
            costPostVictim += 1
            if (
                state.row - pathPostVictim[idx + 1].row != 0
                and state.col - pathPostVictim[idx + 1].col != 0
            ):
                costPostVictim += 0.5

        if costVictim >= tl - 0.5 or costPostVictim >= tl - 0.5:
            pathToGo = pathGoal
            self.finished = True
        else:
            pathToGo = pathVictim

        if len(pathToGo) == 1:
            return "nop", self.currentState

        if len(pathToGo) == 2 and self.currentVictim:
            self.currentVictim.type = NodeType.SAVED
            print(
                    "vitima salva em ",
                    self.currentState,
                    " id: ",
                    self.currentVictim.victimId,
                    " nível de gravidade: ",
                    self.currentVictim.gravityLevel,
                )

        pathToGo.reverse()
        pathToGo.pop()

        nextState = pathToGo.pop()
        delta = (
            nextState.row - self.currentState.row,
            nextState.col - self.currentState.col,
        )

        codeRow = {0: "", 1: "S", -1: "N"}
        codeCol = {0: "", 1: "L", -1: "O"}

        direction = f"{codeRow[delta[0]]}{codeCol[delta[1]]}"
        if len(direction) == 2:
            direction = direction.replace("L", "E")

        return direction, nextState

    def do(self):
        """
        Método utilizado para o polimorfismo dos planos

        Retorna o movimento e o estado do plano (False = nao concluido, True = Concluido)
        """

        nextMove = self.move()
        return (nextMove[1], self.goalPos == State(nextMove[0][0], nextMove[0][1]))

def pathCost(path):
    cost = 0
    for (idx, state) in enumerate(path[0:-1]):
        cost += 1
        if (
            state.row - path[idx + 1].row != 0
            and state.col - path[idx + 1].col != 0
        ):
            cost += 0.5
    return cost
