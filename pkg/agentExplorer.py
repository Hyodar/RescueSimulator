## AGENTE EXPLORER
### @Author: Franco Barp Gomer e Gustabvo Brunholi Chierici (UTFPR)
### (TODO: mudar isso) Agente que fixa um objetivo aleatório e anda aleatoriamente pelo labirinto até encontrá-lo.
### Executa raciocíni on-line: percebe --> [delibera] --> executa ação --> percebe --> ...
import sys
import os

## Importa Classes necessarias para o funcionamento
from model import Model
from problem import Problem
from state import State
from random import randint

## Importa o algoritmo para o plano
from explorerPlan import ExplorerPlan

##Importa o Planner
sys.path.append(os.path.join("pkg", "planner"))
from planner import Planner
from node import Node, NodeType

GRAVITY_LEVEL = {1: [0.0, 25.0], 2: [25.0, 50.0], 3: [50.0, 75.0], 4: [75.0, 100.0]}

## Classe que define o Agente
class AgentExplorer:
    def __init__(self, model, configDict):
        """
        Construtor do agente random
        @param model referencia o ambiente onde o agente estah situado
        """

        self.model = model

        self.map = [
            [Node(NodeType.UNKNOWN, i, j) for j in range(self.model.columns)]
            for i in range(self.model.rows)
        ]

        self.totalVictims = configDict["Vitimas"]

        ## Obtem o tempo que tem para executar
        self.tl = configDict["Te"]
        self.totalTime = configDict["Te"]
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

        ## Cria a instancia do plano para se movimentar aleatoriamente no labirinto (sem nenhuma acao)
        self.plan = ExplorerPlan(
            model.rows, model.columns, self.prob.goalState, initial, "goal", self.mesh
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

        if self.prob.goalTest(self.currentState) and self.plan.goalTest(
            self.currentState
        ):
            print("!!! Objetivo atingido !!!")
            del self.libPlan[0]

            victims = []

            with open(
                os.path.join("config_data", "ambiente_rescuer.txt"), "w"
            ) as ambientRescuer:
                with open(os.path.join("config_data", "ambiente.txt"), "r") as ambient:
                    for line in ambient:
                        if not line.startswith("Vitimas") and not line.startswith(
                            "Parede"
                        ):
                            ambientRescuer.write(line)

                walls = []

                for i in range(self.model.rows):
                    for j in range(self.model.columns):
                        if self.map[i][j].type == NodeType.VICTIM:
                            victims.append((i, j, self.map[i][j].gravityLevel))
                        elif (
                            self.map[i][j].type == NodeType.OBSTACLE
                            or self.map[i][j].type == NodeType.UNKNOWN
                        ):
                            walls.append((i, j))

                ambientRescuer.write(
                    f"Vitimas {' '.join(list(map(lambda el: f'{el[0]},{el[1]}', victims)))}\n"
                )
                ambientRescuer.write(
                    f"Parede {' '.join(list(map(lambda el: f'{el[0]},{el[1]}', walls)))}"
                )

                with open(
                    os.path.join("config_data", "sinaisvitais_rescuer.txt"), "w"
                ) as vitalSignalsRescuer:
                    lines = [
                        f"{idx + 1},"
                        + ",".join(map(str, self.map[pos[0]][pos[1]].vitalSignals))
                        for (idx, pos) in enumerate(victims)
                    ]
                    vitalSignalsRescuer.write("\n".join(lines))
            print(victims)

            discoveredGravities = [0, 0, 0, 0]
            for victim in victims:
                discoveredGravities[victim[2] - 1] += 1

            totalGravities = [0, 0, 0, 0]
            for i in range(len(self.totalVictims)):
                vitalSignals = self.model.maze.vitalSignals[i]
                gravity = vitalSignals[5]
                gravityLevel = [
                    k
                    for k, v in GRAVITY_LEVEL.items()
                    if gravity > v[0] and gravity <= v[1]
                ][0]
                totalGravities[gravityLevel - 1] += 1

            veg = sum(
                (
                    gravity * (4 - idx)
                    for (idx, gravity) in enumerate(discoveredGravities)
                )
            ) / sum(
                (gravity * (4 - idx) for (idx, gravity) in enumerate(totalGravities))
            )

            print("Estatísticas:")
            print("------------------------------")
            print(
                f"pve = {len(victims)}/{len(self.totalVictims)} = {len(victims) / len(self.totalVictims)}"
            )
            print(
                f"tve = {self.costAll}/{len(victims)} = {self.costAll / len(victims)}"
            )
            print(f"veg = {veg}")

        if (
            self.map[self.currentState.row][self.currentState.col].type
            == NodeType.UNKNOWN
        ):
            self.map[self.currentState.row][self.currentState.col].type = NodeType.EMPTY

            victimId = self.victimPresenceSensor()
            if victimId > 0:
                vitalSignals = self.victimVitalSignalsSensor(victimId)
                gravity = vitalSignals[5]

                node = self.map[self.currentState.row][self.currentState.col]

                node.type = NodeType.VICTIM
                node.vitalSignals = vitalSignals
                node.gravityLevel = [
                    k
                    for k, v in GRAVITY_LEVEL.items()
                    if gravity > v[0] and gravity <= v[1]
                ][0]
                print(
                    "vitima encontrada em ",
                    self.currentState,
                    " id: ",
                    victimId,
                    " sinais vitais: ",
                    self.victimVitalSignalsSensor(victimId),
                )

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

        if self.expectedState != self.positionSensor():
            self.map[self.expectedState.row][
                self.expectedState.col
            ].type = NodeType.OBSTACLE

        for mapLine in self.map:
            print(list(map(lambda node: int(node.type), mapLine)))

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
        self.costAll += 2
        self.tl -= 2
        return self.model.getVictimVitalSignals(victimId)

    ## Metodo que atualiza a biblioteca de planos, de acordo com o estado atual do agente
    def updateLibPlan(self):
        for i in self.libPlan:
            i.updateCurrentState(self.currentState)

    def actionDo(self, posAction, action=True):
        self.model.do(posAction, action)
