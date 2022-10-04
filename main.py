import sys
import os
import time

## Importa as classes que serao usadas
sys.path.append(os.path.join("pkg"))
from model import Model
from agentExplorer import AgentExplorer


## Metodo utilizado para permitir que o usuario construa o labirindo clicando em cima
def buildMaze(model):
    model.drawToBuild()
    step = model.getStep()
    while step == "build":
        model.drawToBuild()
        step = model.getStep()
    ## Atualiza o labirinto
    model.updateMaze()


def main():
    # Lê arquivo de configuração
    configDict = {}
    with open(os.path.join("config_data", "ambiente.txt"), "r") as f:
        for line in f:
            field, *values = line.replace("\n", "").split(" ")

            if field == "Vitimas" or field == "Parede":
                configDict[field] = tuple(
                    map(lambda value: map(int, value.split(",")), values)
                )
            elif field == "Base":
                configDict[field] = tuple(map(int, values[0].split(",")))
            elif field == "XMax" or field == "YMax" or field == "Te" or field == "Ts":
                configDict[field] = int(values[0])

    print("dicionario config: ", configDict)

    # Cria o ambiente (modelo) = Labirinto com suas paredes
    mesh = "square"

    ## nome do arquivo de configuracao do ambiente - deve estar na pasta <proj>/config_data
    loadMaze = "ambiente"

    model = Model(configDict["XMax"], configDict["YMax"], mesh, loadMaze)
    buildMaze(model)

    model.maze.board.posAgent = configDict["Base"]
    model.maze.board.posGoal = configDict["Base"]
    # Define a posição inicial do agente no ambiente - corresponde ao estado inicial
    model.setAgentPos(model.maze.board.posAgent[0], model.maze.board.posAgent[1])
    model.setGoalPos(model.maze.board.posGoal[0], model.maze.board.posGoal[1])
    model.draw()

    # Cria um agente
    agent = AgentExplorer(model, configDict)

    ## Ciclo de raciocínio do agente
    agent.deliberate()
    while agent.deliberate() != -1:
        model.draw()
        time.sleep(0.01)
    model.draw()


if __name__ == "__main__":
    main()
