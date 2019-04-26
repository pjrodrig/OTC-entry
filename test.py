import numpy as np

class Test:

    neurons = []
    training = True
    records = []
    
    def __init__(self, num_sensors, num_actions):
        self.sensors = [[] for i in range(0, num_sensors)]
        self.actions = [{'value': i, 'from': []} for i in range(0, num_actions)]

    def eval(self, observations):
        if(training):
            return self.train_eval(observations)
        return self.real_eval(observations)

    def train_eval(self, observations): 
        user_inputs = self.get_user_inputs() # set of inputs that the human user deems to be the best action
        if(user_inputs != False):
            even_weights = np.array([(observation/len(user_inputs)) for observation in observations])
            record = {'observations': observations, 'actions': [(1/len(user_inputs) if {action['value']}.issubset(user_inputs) else 0) for action in actions]}
            neuron = {'from': [], 'to': [], 'weights': even_weights, 'record': record}
            self.neruons.append(neuron)
            for action in self.actions:
                if({action['value']}.issubset(user_inputs)):
                    neuron['to'].append(action)
                    action['from'].append(action)
            for sensor in self.sensors:
                neuron['from'].append(sensor)
                sensor.append(neuron)
            self.records.append(record)
            return user_inputs
        return self.real_eval(observations)

    def get_user_inputs(self):
        user_inputs = str(input("Action(s): "))
        if(user_inputs == 'stop'):
            training = False
            return False
        return set([int(user_input) - 1 for user_input in user_inputs.split(' ')])

    def build_network(self):
        for neuron in neurons:
            if(neuron['record'] != None):
                for record in records:
                    if(record != neuron['record']):
                        for 
            

    def real_eval(self, observations):
        predictions = []
        for action in actions:
            calcs = []
            for neuron in action['from']:
                if(neuron.calc == None):
                    neuron.calc = sum([observation * weight for observation, weight in zip(observations, neuron['weights'])])/len(observations)
                calcs.append(neuron.calc)
            predictions.append(sum(calcs)/)
                

    
        