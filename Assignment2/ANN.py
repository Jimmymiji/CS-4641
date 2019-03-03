import os
import csv
import time
import sys
sys.path.append('./ABAGAIL/ABAGAIL.jar')
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import LogisticSigmoid


# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 51
HIDDEN_LAYER1 = 16
HIDDEN_LAYER2 = 5
HIDDEN_LAYER3 = 5
OUTPUT_LAYER = 1


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    dat = open(infile, "r") 
    reader = csv.reader(dat)

    for row in reader:
        instance = Instance([float(value) for value in row[:-1]])
        #instance.setLabel(Instance(int(row[-1])))
        #0 if float(row[-1]) <= 0 else 1
        instance.setLabel(Instance( 0 if row[-1] == 'no' else 1))
        instances.append(instance)

    dat.close()
    
    return instances
	

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc
	
	
def train(oa, network, oaName, training_ints, testing_ints, measure,oFile,TRAINING_ITERATIONS):
    print "\nError results for %s\n---------------------------" % (oaName,)
    f = open(oFile,'w')
    f.write("%s,%s,%s,%s,%s,%s\n"%('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))
    f.close()
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
    	    MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = "%s,%s,%s,%s,%s,%s\n"%(iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,times[-1])
            print txt
            f = open(oFile,'a+')
            f.write(txt)
            f.close()

def main():
    training_ints = initialize_instances('data/bank_train.csv')
    testing_ints = initialize_instances('data/bank_test.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    acti = LogisticSigmoid()
    rule = RPROPUpdateRule()
    ######################### back prop #####################

    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER],acti)
    train(BatchBackPropagationTrainer(data_set,classification_network,measure,rule,), 
            classification_network,
            'Backprop', 
            training_ints,testing_ints, measure,
            './ANN/BP/BACKPROP_LOG.csv',
            2000)

    ######################### simulated annealing #################

    for CE in [0.15,0.35,0.55,0.75,0.95]:
        for  T in [1e8,1e10,1e12]:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER],acti)
            nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
            oFile = "./ANN/SA/%s_%s_LOG.csv"%(CE,T)
            train(SimulatedAnnealing(T, CE, nnop), 
            classification_network, 
            'simulated annealing', 
            training_ints, testing_ints, measure,
            oFile,
            2000)
    
    ######################### random hill climbing #################

    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER],acti)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    train(RandomizedHillClimbing(nnop), 
        classification_network, 
        'RHC', 
        training_ints, testing_ints, measure,
        './ANN/RHC/RHC_LOG.csv',
        2000)

    ######################### genetic algorithm #################
    
    for P in [100]:
        for mate in [5, 15, 30]:
            for mutate in [5,15,30]:
                classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER],acti)
                nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
                oFile = "./ANN/GA/%s_%s_%s_LOG.csv"%(P, mate, mutate)
                train(StandardGeneticAlgorithm(P, mate, mutate, nnop), 
                    classification_network, 
                    'GA', 
                    training_ints, testing_ints, measure,
                    oFile,
                    2000)



if __name__ == "__main__":
    main()