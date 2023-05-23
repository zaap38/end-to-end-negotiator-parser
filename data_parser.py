import copy as cp
import json


ITEM_NAMES = ["book", "hat", "ball"]


class Data:

    def __init__(self, path=None) -> None:

        '''
        self.data[dialogue_id][data_choice]{"input"}{"amount}[item_id] -> amount
                                                    {0}[item_id] -> value for agent 0 (YOU)
                                                    {1}[item_id] -> value for agent 1 (THEM)
                                           {"conv"}[sentence_id]{"author"} -> agent name "YOU" or "THEM" corresponding to agent IDs 0 and 1
                                                                {"raw"} -> sentence text as string
                                                                {"state"}[agent_id][item_id] -> amount ; current state of the negotiation
                                           {"output"}[agent_id][item_id] -> amount obtained by current agent as int, -1 everywhere means disagree
                                           {"reward"}[agent_id] -> reward obtained by the current agent as int

        @dialogue_id : int [0;n-1] - the dialogue line in the dataset
        @agent_id : int [0;1] - the agent side (mirrored)
        @data_choice : dict_key {"input"; "conv"; "output"; "reward"}
        @item_id : int [0; 2] - 3 existing items, returns the tuple (as a list) of the quantity
            and the value for each for the selected agent
        @sentence_id : a certain sentence of the conversation
        '''

        self.data = []
        if path is not None:
            self.load(path)

    def load(self, filename):
        
        with open(filename, "r") as f:
            valid = 0
            total = 0
            for line in f:
                if "agree" in line.split(" ") and \
                        line.split(" ").index("YOU:") < line.split(" ").index("THEM:"):  # load only agreement lines
                    valid += self.load_line(line)
                    total += 1
            print(str(valid) + "/" + str(total), str(round(100 * valid / total, 1)) + "%")

    def load_line(self, line):

        tokens = line.split(" ")  # split all then rebuild parts

        amounts = []  # amount of each item
        vYou = []  # value for agent 0 YOU
        vThem = []  # value for agent 1 THEM
        raw_numbers = []
        for w in tokens:
            if w.isnumeric():
                raw_numbers.append(int(w))
            else:
                break
        for i in range(len(raw_numbers) // 2):
            index = i * 2
            amounts.append(int(raw_numbers[index]))
            vYou.append(int(raw_numbers[index  + 1]))
        
        sentences_str = ""  # load only the dialogue lines
        temp = ""
        for w in tokens:
            if w == "YOU:" or w == "THEM:":
                temp = ""
            temp += w + " "
            if w == "<eos>":
                sentences_str += temp
                temp = ""
        remaining = temp  # reward x agree i1 v1 i2 v2 i3 v3
        if len(remaining.split("agree")) > 2:  # fixing "no agreement agree" cases
            return 0
        _, values_str = remaining.split("agree")
        values_str = remove_whitespace_start_end(values_str)
        raw_numbers = values_str.split(' ')
        for i in range(len(raw_numbers) // 2):  # getting the values for agent 1 "THEM"
            index = i * 2
            vThem.append(int(raw_numbers[index  + 1]))

        parsed_sentence = sentences_str.split("<eos>")
        parsed_sentence.pop(-1)  # each sentence is in a different cel, starting with YOU: or THEM:, without <eos>
        for i, s in enumerate(parsed_sentence):  # cleaning
            parsed_sentence[i] = remove_whitespace_start_end(s)
        
        t = []  # [sentence_id] -> {author, raw, state}

        output_raw = parsed_sentence.pop(-1)  # extract the "<selection>" sentence
        outputYou = []
        outputThem = []
        output_raw = remove_whitespace_start_end(output_raw.split("<selection>")[-1])
        temp = output_raw.split(' ')
        cpt = 0
        # print("vvvvvvvv")
        for w in temp:  # set the amount of each item for each agent given the final output
            # print(w)
            if not w.split('=')[-1].isnumeric():
                return 0
            outputYou.append(int(w.split('=')[-1]))
            outputThem.append(amounts[cpt] - outputYou[-1])
            cpt += 1

        rewardYou = 0
        rewardThem = 0
        for i in range(len(amounts)):  # compute the reward for each agent
            rewardYou += outputYou[i] * vYou[i]
            rewardThem += outputThem[i] * vThem[i]

        for s in parsed_sentence:
            t.append(dict())  # init a new sentence tuple
            # print(s)
            splitted = s.split(':')  # separate the author and the rest of the sentence
            author = splitted.pop(0)
            sentence = " ".join(splitted)
            author = remove_whitespace_start_end(author)  # cleaning
            sentence = remove_whitespace_start_end(sentence)
            sentence = sentence.replace('.', '')
            sentence = sentence.replace(':', '')
            sentence = sentence.replace(')', '')
            sentence = sentence.replace('\'', '')
            sentence = sentence.replace('`', '')
            sentence = remove_whitespace_start_end(sentence)
            tokens = sentence.split(' ')
            tokens2 = []
            for w in tokens:
                if w != " " and len(w) > 0:
                    tokens2.append(w.lower())

            sentence = " ".join(tokens2)

            t[-1]["author"] = cp.deepcopy(author)
            t[-1]["raw"] = cp.deepcopy(sentence)
            t[-1]["state"] = []  # not used yet

        self.data.append(dict())
        self.data[-1]["input"] = dict()
        self.data[-1]["input"]["amount"] = cp.deepcopy(amounts)
        self.data[-1]["input"][0] = cp.deepcopy(vYou)
        self.data[-1]["input"][1] = cp.deepcopy(vThem)
        self.data[-1]["output"] = [cp.deepcopy(outputYou), cp.deepcopy(outputThem)]
        self.data[-1]["conv"] = cp.deepcopy(t)
        self.data[-1]["reward"] = [rewardYou, rewardThem]

        sentences = []
        for c in self.data[-1]["conv"]:
            sentences.append(c["raw"])

        states = get_state_list(sentences, amounts)
        print("----", self.data[-1]["input"]["amount"])
        for i in range(len(states)):
            self.data[-1]["conv"][i]["state"] = cp.deepcopy(states[i])
            print(self.data[-1]["conv"][i]["raw"], self.data[-1]["conv"][i]["state"])

        if check_validity_of_parsing(self.data[-1]["conv"][-1]["state"], self.data[-1]["output"][0]) and self.data[-1] != {}:
            # print("CORRECT")
            return 1
        else:
            # print("ERROR:", self.data[-1]["conv"][-1]["state"], "instead of", self.data[-1]["output"][0])
            self.data.pop(-1)
            return 0
        
    def display(self, d):
        try:
            print("Values YOU:", d["input"][0],"Values THEM", d["input"][1], "Amounts:", d["input"]["amount"], "Reward:", d["reward"])
            for c in d["conv"]:
                print(c["author"], ':', c["raw"], c["state"])

        except:
            """print("################### Invalid format ###################")
            print(d)"""
            pass

    @staticmethod
    def negative(state, amount):
        tmp = []
        for i in range(len(amount)):
            tmp.append(amount[i] - state[i])
        return tmp

    def export(self):
        json_object = json.dumps(self.data, indent=4)
        with open("./src/data/parsed.json", "w") as out:
            out.write(json_object)
            print("Exported to src/data/parsed.json")


def get_state_list(sentences, amounts):
    current = []
    for _ in amounts:
        current.append(0)
    states = []  # does not consider the starting state with all amounts set to zero
    negative = False
    # print("Amounts:", amounts)
    for s in sentences:
        states.append(build_state_variation(s, current, amounts, negative))
        current = states[-1]
        # print("THEM:" if negative else "YOU:", s, states[-1])
        negative = not negative
    
    return states
    

def build_state_variation(s, previous, amounts, negative):
    # see the state modification with an automata, built for the YOU agent
    state = []
    for _ in previous:
        state.append(0)

    selfWords = ["i", "me", "lemme", "gimme", "ill"]
    otherWords = ["you"]
    # overwrite = [""]
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    addingState = 0  # 0 -> adds to YOU, 1 -> adds to THEM
    addNextItem = False
    value = 0
    dealing = False
    noDeal = False
    cannot = False
    for w in s.split(' '):
        if w in selfWords:
            addingState = 0
            noDeal = False
        elif w in otherWords:
            addingState = 1
            noDeal = False
        elif addNextItem and to_singular(w) in ITEM_NAMES:
            addNextItem = False
            index = ITEM_NAMES.index(to_singular(w))
            wasNoneValue = False
            if value is None:  # handling the "the" word meaning
                wasNoneValue = True
                if isPlurial(w):
                    value = amounts[index]
                else:
                    value = 1
            
            if cannot:
                value = -value
                cannot = False
            if addingState == 0:
                state[index] = value
            else:
                state[index] = amounts[index] - value
            if wasNoneValue:
                value = None
            noDeal = False
        elif w in ["the", "a", "both"] or w.isnumeric() or w in units:
            if w in ["the", "a", "both"]:
                value = None  # if singular, 1, otherwise all
            elif w in units:
                value = units.index(w)
            else:
                value = int(w)
            addNextItem = True
            noDeal = False
        elif w == "deal":
            if noDeal == False:
                state = cp.deepcopy(previous)
                dealing = True
            noDeal = False
        elif w == "no":
            noDeal = True
        elif w == "and":
            addNextItem = True
        elif w == "cant" or w == "cannot":
            cannot = True
        # print(w, addingState, addNextItem, value, noDeal, state)

        # TODO: currently the state is not taking care of the talking agent

    allZero = True
    for x in state:
        if x != 0:
            allZero = False

    if allZero:
        state = cp.deepcopy(previous)
        return state

    if negative and not dealing:
        tmp = []
        for i in range(len(state)):
            tmp.append(amounts[i] - state[i])
        state = tmp

    return state


def to_singular(word):
    if word[-1] == 's':
        word = word[:-1]
    return word

def isPlurial(word):
    return word[-1] == 's'


def remove_whitespace_start_end(string):
    if string[0] == ' ':
        string = string[1:]
    if string[-1] == ' ':
        string = string[:-1]
    return string

def check_validity_of_parsing(computed_final, final):
    for i in range(len(computed_final)):
        if computed_final[i] != final[i]:
            return False
    return True

if __name__ == "__main__":
    # test_string = "1 0 4 2 1 2 YOU: i would like 4 hats and you can have the rest . <eos> THEM: deal <eos> YOU: <selection> item0=0 item1=4 item2=0 <eos> reward=8 agree 1 4 4 1 1 2"
    # test_string = "1 10 3 0 1 0 YOU: hi i would like the book and ball and you can have the hats <eos> THEM: i can give you either the book or the ball <eos> YOU: ill take the book <eos> THEM: ok i will take the hats and ball <eos> YOU: deal <eos> THEM: <selection> item0=1 item1=0 item2=0 <eos> reward=10 agree 1 2 3 2 1 2"
    # test_string = "4 1 1 6 2 0 YOU: i'll take the hat , you can have the rest . <eos> THEM: i would like the balls , hat and 2 books <eos> YOU: i can give you the hat and the balls if i keep the books . <eos> THEM: sorry that wont work . <eos> YOU: can you either give me the hat or 3 of the books ? <eos> THEM: sorry looks like we wont be making a deal <eos> YOU: yeah i have no idea how to do that . <eos> THEM: we keep saying no deal until the no deal button appears <eos> YOU: oh , how about i get 2 books ? <eos> THEM: you get 2 books and i get the rest ? <eos> YOU: yes . <eos> THEM: deal <eos> YOU: <selection> item0=2 item1=0 item2=0 <eos> reward=2 agree 4 1 1 2 2 2"
    data = Data("./src/data/negotiate/data.txt")
    # data.load_line(test_string)
    # print(data.data[:-50:-1])
    data.export()
    """for d in data.data:
        print()
        print("######################")
        data.display(d)"""
