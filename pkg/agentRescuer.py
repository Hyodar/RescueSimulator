## AGENTE RESCUER
### @Author: Franco Barp Gomer e Gustabvo Brunholi Chierici (UTFPR)
### (TODO: mudar isso) Agente que fixa um objetivo aleatório e anda aleatoriamente pelo labirinto até encontrá-lo.
### Executa raciocíni on-line: percebe --> [delibera] --> executa ação --> percebe --> ...
import sys

## Importa Classes necessarias para o funcionamento
from model import Model
from node import Node, NodeType
from problem import Problem
from state import State
from random import randint

## Importa o algoritmo para o plano
from rescuerPlan import RescuerPlan
from rescuerPlanAStar import RescuerPlanAStar

GRAVITY_LEVEL = {
    1: [0.0, 25.0],
    2: [25.0, 50.0],
    3: [50.0, 75.0],
    4: [75.0, 100.0]
}

## Classe que define o Agente
class AgentRescuer:
    def __init__(self, model, configDict):
        """
        Construtor do agente rescuer
        @param model referencia o ambiente onde o agente estah situado
        """

        self.model = model

        self.map = [
            [Node(NodeType.EMPTY, i, j) for j in range(self.model.columns)]
            for i in range(self.model.rows)
        ]

        for i in range(self.model.rows):
            for j in range(self.model.columns):
                node = self.map[i][j]

                if (i, j) in configDict["Vitimas"]:
                    victimId = configDict["Vitimas"].index((i, j)) + 1
                    gravity = self.model.maze.vitalSignals[victimId - 1][5]

                    node.type = NodeType.VICTIM
                    node.victimId = victimId
                    node.gravityLevel = [k for k, v in GRAVITY_LEVEL.items() if gravity > v[0] and gravity <= v[1]][0]
                elif (i, j) in configDict["Parede"]:
                    node.type = NodeType.OBSTACLE

        for mapLine in self.map:
            print(list(map(lambda node: int(node.type), mapLine)))

        ## Obtem o tempo que tem para executar
        self.tl = configDict["Ts"]
        print("Tempo disponivel: ", self.tl)

        ## Pega o tipo de mesh, que está no model (influência na movimentação)
        self.mesh = self.model.mesh

        ## Cria a instância do problema na mente do agente (sao suas crencas)
        self.prob = Problem()
        self.prob.createMaze(model.rows, model.columns, model.maze)

        # O agente le sua posica no ambiente por meio do sensor
        initial = self.positionSensor()
        self.prob.defInitialState(initial.row, initial.col)
        print("*** Estado inicial do agente: ", self.prob.initialState)

        # Define o estado atual do agente = estado inicial
        self.currentState = self.prob.initialState

        # definimos um estado objetivo que veio do arquivo ambiente.txt
        self.prob.defGoalState(model.maze.board.posGoal[0], model.maze.board.posGoal[1])
        print("*** Objetivo do agente: ", self.prob.goalState)
        print(
            "*** Total de vitimas existentes no ambiente: ",
            self.model.getNumberOfVictims(),
        )

        """
        DEFINE OS PLANOS DE EXECUÇÃO DO AGENTE
        """

        ## Custo da solução
        self.costAll = 0

        if sys.argv[1] == "rescuerGA":
            self.plan = RescuerPlan(
                self.tl, model.rows, model.columns, self.prob.goalState, initial, self.map, "goal", self.mesh
            )
        else:
            self.plan = RescuerPlanAStar(
                model.rows, model.columns, self.prob.goalState, initial, self.map, "goal", self.mesh
            )
        ## Adiciona o(s) planos a biblioteca de planos do agente
        self.libPlan = [self.plan]

        ## inicializa acao do ciclo anterior com o estado esperado
        self.previousAction = "nop"  ## nenhuma (no operation)
        self.expectedState = self.currentState

    ## Metodo que define a deliberacao do agente
    def deliberate(self):
        if len(self.libPlan) == 0:
            return -1

        self.plan = self.libPlan[0]

        print("\n*** Inicio do ciclo raciocinio ***")
        print("Pos agente no amb.: ", self.positionSensor())

        self.currentState = self.positionSensor()
        self.plan.updateCurrentState(self.currentState)
        print("Ag cre que esta em: ", self.currentState)

        if not (self.currentState == self.expectedState):
            print(
                "---> falha na execucao da acao ",
                self.previousAction,
                ": esperava estar em ",
                self.expectedState,
                ", mas estou em ",
                self.currentState,
            )

        self.costAll += self.prob.getActionCost(self.previousAction)
        print("Custo até o momento (com a ação escolhida):", self.costAll)

        self.tl -= self.prob.getActionCost(self.previousAction)
        print("Tempo disponivel: ", self.tl)

        if (
            self.prob.goalTest(self.currentState) and (self.tl <= 0.5 or self.plan.goalTest()
            or len([node for nodes in self.map for node in nodes if node.type == NodeType.VICTIM]) == 0)
        ):
            print("!!! Objetivo atingido !!!")
            del self.libPlan[0]

            victims = [v for k in self.map for v in k if v.type == NodeType.VICTIM or v.type == NodeType.SAVED]
            saved = [s for k in self.map for s in k if s.type == NodeType.SAVED]

            totalGravities = [0, 0, 0, 0]
            for v in victims:
                totalGravities[v.gravityLevel - 1] += 1

            savedGravities = [0, 0, 0, 0]
            for s in saved:
                savedGravities[s.gravityLevel - 1] += 1

            vsg = sum(
                (
                    gravity * (4 - idx)
                    for (idx, gravity) in enumerate(savedGravities)
                )
            ) / sum(
                (gravity * (4 - idx) for (idx, gravity) in enumerate(totalGravities))
            )

            print("Estatísticas:")
            print("------------------------------")
            print(
                f"pvs = {len(saved)}/{len(victims)} = {len(saved) / len(victims)}"
            )
            print(
                f"tvs = {self.costAll}/{len(saved)} = {self.costAll / len(saved)}"
            )
            print(f"vsg = {vsg}")

        if self.map[self.currentState.row][self.currentState.col].type == NodeType.SAVED:
            self.model.maze.board.listPlaces[self.currentState.row][self.currentState.col].victim = False
            self.model.maze.board.listPlaces[self.currentState.row][self.currentState.col].saved = True

        result = self.plan.chooseAction(self.tl)
        print(
            "Ag deliberou pela acao: ",
            result[0],
            " o estado resultado esperado é: ",
            result[1],
        )

        ## Executa esse acao, atraves do metodo executeGo
        self.executeGo(result[0])
        self.previousAction = result[0]
        self.expectedState = result[1]

        # for mapLine in self.map:
        #     print(list(map(int, mapLine)))

        return 1

    ## Metodo que executa as acoes
    def executeGo(self, action):
        """Atuador: solicita ao agente físico para executar a acao.
        @param direction: Direcao da acao do agente {"N", "S", ...}
        @return 1 caso movimentacao tenha sido executada corretamente"""

        ## Passa a acao para o modelo
        result = self.model.go(action)

        ## Se o resultado for True, significa que a acao foi completada com sucesso, e ja pode ser removida do plano
        ## if (result[1]): ## atingiu objetivo ## TACLA 20220311
        ##    del self.plan[0]
        ##    self.actionDo((2,1), True)

    ## Metodo que pega a posicao real do agente no ambiente
    def positionSensor(self):
        """Simula um sensor que realiza a leitura do posição atual no ambiente.
        @return instancia da classe Estado que representa a posição atual do agente no labirinto."""
        pos = self.model.agentPos
        return State(pos[0], pos[1])

    def victimPresenceSensor(self):
        """Simula um sensor que realiza a deteccao de presenca de vitima na posicao onde o agente se encontra no ambiente
        @return retorna o id da vítima"""
        return self.model.isThereVictim()

    def victimVitalSignalsSensor(self, victimId):
        """Simula um sensor que realiza a leitura dos sinais da vitima
        @param o id da vítima
        @return a lista de sinais vitais (ou uma lista vazia se não tem vítima com o id)"""
        return self.model.getVictimVitalSignals(victimId)

    ## Metodo que atualiza a biblioteca de planos, de acordo com o estado atual do agente
    def updateLibPlan(self):
        for i in self.libPlan:
            i.updateCurrentState(self.currentState)

    def actionDo(self, posAction, action=True):
        self.model.do(posAction, action)
