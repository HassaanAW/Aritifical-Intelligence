
# HASSAAN AHMAD WAQAR - 22100137
# Set up library imports.
import random
from collections import Counter
from itertools import chain

# install bitstring 
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("bitstring")

# import bitstring
from bitstring import *

########################################################
'''
    - This module assumes you first read the Jupyter notebook. 
    - You are free to add other members functions in class GeneticAlgorithm
      as long as you do not modify the code already written. If you have justified
      reasons for making modifications in code, come talk to us. 
    - Our implementation uses recursive solutions and some flavor of 
      functional programming (maps/lambdas); You're not required to do so.
      Just Write clean code. 
'''
########################################################

class GeneticAlgorithm(object):

    def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, verbose):
        self.wall_bit_string_raw = "01010101011001101101010111011001100101010101100101010101"
        self.wall_bit_string = ConstBitStream(bin = self.wall_bit_string_raw)
        self.population_size = POPULATION_SIZE
        self.chromosome_length = CHROMOSOME_LENGTH # this is the length of self.wall_bit_string
        self.terminate = False
        self.verbose = verbose # In verbose mode, fitness of each individual is shown. 


    def run_genetic_alg(self):
        '''  
        The pseudo you saw in slides of Genetic Algorithm is implemented here. 
        Here, You'll get a flavor of functional 
        programming in Python- Those who attempted ungraded optional tasks in tutorial
        have seen something similar there as well. 
        Those with experience in functional programming (Haskell etc)
        should have no trouble understanding the code below. Otherwise, take our word that
        this is more or less similar to the generic pseudocode in Jupyter Notebook.

        '''
        "You may not make any changes to this function."

        # Creation of Population
        solutions = self.generate_candidate_sols(self.population_size) # arg passed for recursive implementation.

        # Evaluation of individuals
        parents = self.evaluate_candidates(solutions)

        while(not self.terminate):
            # Make pairs
            pairs_of_parents = self.select_parents(parents)

            # Recombination of pairs.
            recombinded_parents = list(chain(*map(lambda pair: \
                self.recombine_pairs_of_parents(pair[0], pair[1]), \
                    pairs_of_parents))) 

            # Mutation of each individual
            mutated_offspring = list(map(lambda offspring: \
                self.mutate_offspring(offspring), recombinded_parents))

            # Evaluation of individuals
            parents = self.evaluate_candidates(mutated_offspring) # new parents (offspring)
            if self.verbose and not self.terminate:
                self.print_fitness_of_each_indiviudal(parents)

######################################################################
###### These two functions print fitness of each individual ##########

# *** "Warning" ***: In this function, if an individual with 100% fitness is discovered, algorithm stops. 
# You should implement a stopping condition elsewhere. This codition, for example,
# won't stop your algorithm if mode is not verbose.
    def print_fitness_of_one_individual(self, _candidate_sol):
        _WallBitString = self.wall_bit_string
        _WallBitString.pos = 0
        _candidate_sol.pos = 0
        
        matching_bit_pairs = 0
        try:
            if not self.terminate:
                while (_WallBitString.read(2).bin == _candidate_sol.read(2).bin):
                    matching_bit_pairs = matching_bit_pairs + 1
                print('Individual Fitness: ', round((matching_bit_pairs)/28*100, 2), '%')
        except: # When all bits matched. 
            pass
            return

    def print_fitness_of_each_indiviudal(self, parents):
        if parents:
            for _parent in parents:
                self.print_fitness_of_one_individual(_parent)

###### These two functions print fitness of each individual ##########
######################################################################

    def select_parents(self, parents):
        '''
        args: parents (list) => list of bitstrings (ConstbitStream)
        returns: pairs of parents (tuple) => consecutive pairs.
        '''

        # **** Start of Your Code **** #
        length = len(parents) 
        listform = []

        loop = int(self.population_size/2)
        for i in range(0,loop):
            rand = random.randint(0,length-2)
            listform.append( (parents[rand], parents[rand + 1]) )
        
        return listform
    

        # parent_tuple = tuple(listform)
        # return parent_tuple

        # rand = random.randint(0,length-1)
        #     parent_tuple = (parents[rand], parents[rand+1], parents[rand +2])
        
        pass
        # **** End of Your Code **** #


    # A helper function that you may find useful for `generate_candidate_sols()`
    def random_num(self):
        random.seed()
        return random.randrange(2**14) ## for fitting in 14 bits.

    def generate_candidate_sols(self, n): 
        '''
        args: n (int) => Number of cadidates solutions to generate. 
        retruns: (list of n random 56 bit ConstBitStreams) 
                 In other words, a list of individuals: Population.

        Each cadidates solution is a 56 bit string (ConstBitStreams object). 

        One clean way is to first get four 14 bit random strings then concatenate
        them to get the desired 56 bit candidate. Repeat this for n candidates.
        '''

        # **** Start of Your Code **** #
        candidates_list = []

        for j in  range(0,n):
            concat = ""
            for i in range(0,56):
                num = str(random.randint(0,1))
                concat = concat + num
            concat = ConstBitStream(bin = concat)
            candidates_list.append(concat)

        return candidates_list
        
        pass
        # **** End of Your Code **** # 

    def recombine_pairs_of_parents(self, p1, p2):
        """
        args: p1, and p2  (ConstBitStream)
        returns: p1, and p2 (ConstBitStream)

        split at .6-.9 of 56 bits (CHROMOSOME_LENGTH). i.e. between 31-50 bits
        """
        # p1 = bin(int(str(p1)[2:],16)).zfill(56)
        p1 = p1.bin
        # p2 = bin(int(str(p2)[2:],16)).zfill(56)
        p2 = p2.bin
        # print(p1)
        # print(p2)
        rand = random.randint(30,50)
        x = p1[rand:]
        y = p2[rand:]
        # print("x",x)
        # print("y",y)
        p1 = p1[:rand] + y
        p2 = p2[:rand] + x

        # print(p1)
        # print(p2)
        p1 = ConstBitStream(bin = p1)
        p2 = ConstBitStream(bin = p2)

        return p1, p2

        

        # **** Start of Your Code **** #
        pass
        # **** End of Your Code **** #

    def mutate_offspring(self, p):
        ''' 
            args: individual (ConstBitStream)
            returns: individual (ConstBitStream)
        '''
        # p = bin(int(str(p)[2:],16)).zfill(56)
        p = p.bin
        #print(p)
        #p = p[2:]
        
        temp = p
        for i in range(0, len(p)):
            mutation = random.randint(90,100)
            probability = random.randint(1,100)
            if probability > mutation:
                if p[i] == '0':
                    temp = temp[:i] + '1' + temp[i+1:]    
                else:
                    temp = temp[:i] + '0' + temp[i+1:] 
        
        # print(temp)
        # print(len(temp))
        temp = ConstBitStream(bin = temp)
        return temp
        


        # **** Start of Your Code **** #
        pass
        # **** End of Your Code **** #

    def sort_second(self,val):
        return val[1]

    def evaluate_candidates(self, candidates): 
        '''
        args: candidate solutions (list) => each element is a bitstring (ConstBitStream)
        
        returns: parents (list of ConstBitStream) => each element is a bitstring (ConstBitStream) 
                    but elements are not unique. Fittest candidates will have multiple copies.
                    Size of 'parents' must be equal to population size.  
        '''
        # match for each candidate/total number of possible matches = fitness
        # (total matches / total possible matches) / population = average fitness 
        # fitness / average fitness = expected number in next gen

        
        # print("hello", candidates)
        # print("hello", candidates[0][1:2])
        ind_match_list = []
        fitness_list = []
        expected_list = []
        possible_matches = 28
        total_possible_matches = 28 
        num = len(candidates)
        total_matches = 0
        
        for i in range(0,num):
            current = candidates[i]
            # current = bin(int(str(current)[2:],16)).zfill(56)
            current = current.bin
            current = str(current)
            ind_match = 0
            for j in range(0, 55, 2):
                if current[j:j+2] == self.wall_bit_string_raw[j:j+2]:
                    ind_match = ind_match + 1
                    total_matches = total_matches + 1

            ind_match_list.append(ind_match)
            fitness_list.append( ind_match/possible_matches )

        avg_fitness = (total_matches/total_possible_matches)/ self.population_size

        for k in range(0,num):
            expected_list.append( round(fitness_list[k]/avg_fitness) )

        final_list = []
        for k in range(0, num):
            final_list.append( [candidates[k],expected_list[k] ] )

        final_list.sort(key = self.sort_second, reverse = True)
        
        newlist = []
        x = 0
        for i in range(0,8):
            for j in range(0,8):
                cure = final_list[x][0]
                newlist.append(cure)
            x = x + 1
        
        if len(newlist) < self.population_size:
            difference = self.population_size - len(newlist)
           
            while(difference != 0):
                newlist.append(final_list[0][0])
                difference = difference - 1
                newlist.append(final_list[1][0])
                difference = difference - 1


        random.shuffle(newlist)
 
        #print("length", len(newlist))
        return newlist

        # **** Start of Your Code **** #
        pass
        # **** End of Your Code **** # 





