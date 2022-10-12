import numpy as np
import random
import time

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
GRAVITY_LEVEL = {1: [0.0, 25.0], 2: [25.0, 50.0], 3: [50.0, 75.0], 4: [75.0, 100.0]}


class Population:
    def __init__(self, plan, base, bag, vitalSignals, vsgDenominator):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.base = base
        self.vitalSignals = vitalSignals
        self.vsgDenominator = vsgDenominator
        self.plan = plan

    @staticmethod
    def fromNodes(plan, victimNodes, base, populationCount, vsgDenominator):
        victimStates = list(map(lambda el: el.state, victimNodes))
        vitalSignals = {}

        for node in victimNodes:
            vitalSignals[node.state] = node.gravityLevel

        return Population(
            plan,
            base,
            [np.random.permutation(victimStates) for _ in range(populationCount)],
            vitalSignals,
            vsgDenominator,
        )

    def getSaved(self, chromosome, tl):
        costToNext = pathCost(self.plan.aStar(self.base, chromosome[0]))
        saved = []

        if costToNext <= tl:
            saved.append(chromosome[0])
            tl -= costToNext
        else:
            return saved

        for (idx, _) in enumerate(chromosome[0:-1]):
            costToNext = pathCost(self.plan.aStar(chromosome[idx], chromosome[idx + 1]))
            costToBase = pathCost(self.plan.aStar(chromosome[idx + 1], self.base))

            if costToNext + costToBase <= tl:
                saved.append(chromosome[idx + 1])
                tl -= costToNext
            else:
                break

        return saved

    def fitness(self, chromosome, tl):
        return 1 - self.computeVsg(self.getSaved(chromosome, tl))

    def computeVsg(self, saved):
        gravities = [0, 0, 0, 0]
        for victim in saved:
            gravities[self.vitalSignals[victim] - 1] += 1

        return (
            sum((gravity * (4 - idx) for (idx, gravity) in enumerate(gravities)))
            / self.vsgDenominator
        )

    def evaluate(self, tl):
        scores = np.asarray([self.fitness(chromosome, tl) for chromosome in self.bag])
        self.score = np.min(scores)
        self.best = self.bag[scores.tolist().index(self.score)]
        self.parents.append(self.best)

        if False in (scores[0] == scores):
            scores = np.max(scores) - scores

        scoresSum = np.sum(scores)

        for i in range(scores.shape[0]):
            scores[i] = (
                scores[i] / scoresSum
                if not (scores[i] == 0 and scoresSum == 0)
                else float("inf")
            )

        return scores

    def select(self, tl, k=4):
        fit = self.evaluate(tl)
        while len(self.parents) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.parents.append(self.bag[idx])

    def swapMutation(self, chromosome):
        a, b = np.random.choice(len(chromosome), 2)

        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]

        return chromosome

    def crossoverMutation(self, pCross=0.1):
        children = []
        count = len(self.parents)
        size = len(self.parents[0])

        for _ in range(len(self.bag)):
            if np.random.rand() < pCross:
                children.append(list(self.parents[np.random.randint(count, size=1)[0]]))
            else:
                parent1, parent2 = random.choices(self.parents, k=2)

                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)

                child = [None for _ in range(size)]
                for i in range(start, end + 1):
                    child[i] = parent1[i]

                pointer = 0

                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]

                children.append(child)

        return children

    def mutate(self, pCross=0.1, pMut=0.1):
        nextBag = []
        children = self.crossoverMutation(pCross)

        for child in children:
            if np.random.rand() < pMut:
                nextBag.append(self.swapMutation(child))
            else:
                nextBag.append(child)

        return nextBag


def pathCost(path):
    cost = 0
    for (idx, state) in enumerate(path[0:-1]):
        cost += 1
        if state.row - path[idx + 1].row != 0 and state.col - path[idx + 1].col != 0:
            cost += 0.5
    return cost


def geneticAlgorithm(
    plan,
    victimNodes,
    base,
    maxTime,
    vsgDenominator,
    populationCount=20,
    stagnation=3,
    selectivity=0.2,
    pCross=0.4,
    pMut=0.7,
    printInterval=100,
    verbose=False,
):
    population = Population.fromNodes(
        plan, victimNodes, base, populationCount, vsgDenominator
    )
    best = population.best
    score = float("inf")
    stagnationCounter = 0

    vitalSignals = {}

    for node in victimNodes:
        vitalSignals[node.state] = node.gravityLevel

    generation = 0
    while True:
        population.select(maxTime, populationCount * selectivity)

        if verbose or generation % printInterval == 0:
            print(f"Generation {generation}: {population.score}")

        if population.score < score:
            best = population.best
            score = population.score

        if score == population.score:
            stagnationCounter += 1
        else:
            stagnationCounter = 0

        if stagnationCounter == stagnation or score == 0:
            break

        children = population.mutate(pCross, pMut)
        population = Population(plan, base, children, vitalSignals, vsgDenominator)

        generation += 1

    return list(best)


class RescuerPlan:
    def __init__(
        self,
        tl,
        maxRows,
        maxColumns,
        goal,
        initialState,
        discoveredMap: list,
        name="none",
        mesh="square",
    ):
        """
        Define as variaveis necessárias para a utilização do rescuer plan por um unico agente.
        """
        self.walls = [
            [discoveredMap[i][j].type == NodeType.OBSTACLE for j in range(maxColumns)]
            for i in range(maxRows)
        ]

        self.maxRows = maxRows
        self.maxColumns = maxColumns
        self.initialState = initialState
        self.currentState = initialState
        self.goalPos = goal
        self.actions = []
        self.previousState = initialState
        self.tl = tl

        self.backtrack = []
        self.map = discoveredMap
        self.victims = [
            discoveredMap[row][col]
            for row, _ in enumerate(discoveredMap)
            for col, _ in enumerate(discoveredMap[row])
            if discoveredMap[row][col].type == NodeType.VICTIM
        ]

        self.currentVictim = None
        self.finished = False

        self.victims.sort(key=lambda v: v.gravityLevel)

        totalGravities = [0, 0, 0, 0]
        for i in range(self.maxRows):
            for j in range(self.maxColumns):
                if self.map[i][j].type == NodeType.VICTIM:
                    totalGravities[self.map[i][j].gravityLevel - 1] += 1
        self.vsgDenominator = sum(
            (gravity * (4 - idx) for (idx, gravity) in enumerate(totalGravities))
        )

        path = geneticAlgorithm(
            self,
            self.victims,
            self.initialState,
            self.tl,
            self.vsgDenominator,
            stagnation=5,
            printInterval=1,
            verbose=True,
        )

        self.victimPath = [self.map[state.row][state.col] for state in path]

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

        if self.walls[toState.row][toState.col]:
            return False

        delta_row = toState.row - fromState.row
        delta_col = toState.col - fromState.col

        if delta_row != 0 and delta_col != 0:
            if (
                self.walls[fromState.row + delta_row][fromState.col]
                and self.walls[fromState.row][fromState.col + delta_col]
            ):
                return False

        return True

    def aStar(self, initial, target):
        def heuristic(state):
            return 1

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

        return None

    def chooseAction(self, tl):
        """Escolhe o proximo movimento de acordo com uma DFS online.
        Eh a acao que vai ser executada pelo agente.
        @return: tupla contendo a acao (direcao) e uma instância da classe State que representa a posição esperada após a execução
        """

        self.currentVictim = next(
            (v for v in self.victimPath if v.type == NodeType.VICTIM), None
        )

        pathGoal = self.aStar(self.currentState, self.goalPos)
        pathCostGoal = pathCost(pathGoal)
        pathToGo = pathGoal

        if not self.currentVictim is None:
            pathVictim = self.aStar(self.currentState, self.currentVictim.state)
            pathCostVictim = pathCost(pathVictim)
            pathCostVictimReturn = pathCost(
                self.aStar(self.currentVictim.state, self.goalPos)
            )

            if pathCostVictim + pathCostVictimReturn <= tl:
                pathToGo = pathVictim

        if len(pathToGo) == 1:
            self.finished = True
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
