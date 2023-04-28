import copy as cp

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
            for line in f:
                if "agree" in line.split(" "):  # load only agreement lines
                    self.load_line(line)

    def load_line(self, line):
        self.data.append(dict())

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
        for w in temp:  # set the amount of each item for each agent given the final output
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

            author, sentence = s.split(':')  # separate the author and the rest of the sentence
            author = remove_whitespace_start_end(author)
            sentence = remove_whitespace_start_end(sentence)
            sentence = sentence.replace('.', '')
            sentence = remove_whitespace_start_end(sentence)
            tokens = sentence.split(' ')
            tokens2 = []
            for w in tokens:
                if w != " " and len(w) > 0:
                    tokens2.append(w.lower())

            sentence = " ".join(tokens2)

            t[-1]["author"] = cp.deepcopy(author)
            t[-1]["raw"] = cp.deepcopy(sentence)
            t[-1]["state"] = [[], []]  # not used yet

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

        get_state_list(sentences, amounts)


def get_state_list(sentences, amounts):
    current = []
    for _ in amounts:
        current.append(0)
    states = []  # does not consider the starting state with all amounts set to zero
    for s in sentences:
        states.append(build_state_variation(s, current, amounts))
        current = states[-1]
        print(s, states[-1])
    

def build_state_variation(s, previous, amounts):
    # see the state modification with an automata, built for the YOU agent
    state = []
    for _ in previous:
        state.append(0)

    selfWords = ["i", "me", "lemme", "gimme"]
    otherWords = ["you"]

    addingState = 0  # 0 -> adds to YOU, 1 -> adds to THEM
    addNextItem = False
    value = 0
    noDeal = False
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
            if addingState == 0:
                state[index] = value
            else:
                state[index] = amounts[index] - value
            if wasNoneValue:
                value = None
            noDeal = False
        elif w == "the" or w.isnumeric():
            if w == "the":
                value = None  # if singular, 1, otherwise all
            else:
                value = int(w)
            addNextItem = True
            noDeal = False
        elif w == "deal":
            if noDeal == False:
                state = cp.deepcopy(previous)
            noDeal = False
        elif w == "no":
            noDeal = True
        elif w == "and":
            addNextItem = True
        # print(w, addingState, addNextItem, value, noDeal, state)

        # TODO: currently the state is not taking care of the talking agent

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
                    

if __name__ == "__main__":
    # test_string = "1 0 4 2 1 2 YOU: i would like 4 hats and you can have the rest . <eos> THEM: deal <eos> YOU: <selection> item0=0 item1=4 item2=0 <eos> reward=8 agree 1 4 4 1 1 2"
    test_string = "1 10 3 0 1 0 YOU: hi i would like the book and ball and you can have the hats <eos> THEM: i can give you either the book or the ball <eos> YOU: ill take the book <eos> THEM: ok i will take the hats and ball <eos> YOU: deal <eos> THEM: <selection> item0=1 item1=0 item2=0 <eos> reward=10 agree 1 2 3 2 1 2"
    data = Data()
    data.load_line(test_string)
