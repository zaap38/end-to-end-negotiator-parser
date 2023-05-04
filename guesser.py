from data_parser import Data
import copy as cp
import math


class Guesser:

    def __init__(self) -> None:
        self.values = []
        self.kb = []


    def guess(self, d):
        """
        R1. ask(x) => v(x) > 0
        R2. deal => v(taken) + v(min(x)) >= v(-taken)
        R3. reject(y) => v(y) == 0
        R4. ask(x) -ask(y) => x > y
        """

        self.values = [[], []]
        self.kb.append({})
        self.kb.append({})

        amount = d["input"]["amount"]

        states = []
        for c in d["conv"]:
            states.append(c["state"])

        for _ in amount:
            for i in range(2):
                self.values[i].append([0, 30])  # 30 stands for infinite here

        print(self.values)

        for turn, state in enumerate(states):
            s = cp.deepcopy(state)

            player = turn % 2  # get current player
            v = self.values[player]

            if player == 1:  # negative state
                tmp = data.negative(s, amount)
                s = cp.deepcopy(tmp)

            for i in range(len(v)):  # detect valuables (R1)
                if s[i] != 0:
                    v[i][0] = max(1, v[i][0])

            if turn > 0:  # R4
                delta = []
                for _ in amount:
                    delta.append(0)
                previous = data.negative(states[turn - 1], amount)
                for i in range(len(delta)):
                    delta[i] = s[i] - previous[i]
                print(previous, s, delta)
                added = 0
                removedValue = 0
                addedIndex = 0
                for i, e in enumerate(delta):
                    if e > 0:
                        added += e
                        addedIndex = i
                    else:
                        removedValue += -e * v[i][0]
                if added > 0:
                    v[addedIndex][0] = math.ceil(removedValue / added) + 1
                

            print(self.values)



if __name__ == "__main__":
    g = Guesser()
    data = Data("./src/data/negotiate/data.txt")
    g.guess(data.data[10])
    data.display(data.data[10])
