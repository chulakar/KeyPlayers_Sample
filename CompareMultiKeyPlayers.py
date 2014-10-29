#!/usr/bin/env python
import array
import random
import csv
import time
import math
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import networkx as nx
import numpy
import SamplingAlgorithms
import itertools
import os
import subprocess



#------------------------------------------------------------

#Do sampling




def graph2bin(Network, num):
    random.seed(time.clock())
    random.shuffle(Network.nodes())
    ini_nodes = random.sample(set(Network.nodes()), num)
    #print ini_nodes
    return (int(i in ini_nodes) for i in range(len(Network.nodes())))

def evaluate(ind):
    #print ind
    key_nodes = []
    for i in range(len(ind)):
        if (ind[i]==1):
            key_nodes.append(i)
    sum_eigVec = sum_eig(key_nodes)
    #sum_betCen = sum_bet(key_nodes)
    sum_distance = sum_dist(key_nodes)
    return sum_eigVec, sum_distance
    #return sum_eigVec, sum_betCen
    #posDistance = getPosDistance(key_nodes)
    #negDistance = getNegDistance(key_nodes)
    #return posDistance, negDistance

def writeFitness_hof(hof, j, val, allHOFs):
    #filename_csv = str(val)+"Fitness_values_hof_5_facebook_"+str(j)+"_"+".csv"
    #with open(filename_csv, 'a') as test_file:
        #file_writer = csv.writer(test_file)
    for ind in hof:
        write_list = []
        key_nodes = []
        for i in range(len(ind)):
            if (ind[i]==1):
                key_nodes.append(i)
                write_list.append(i)
        sum_eigVec = sum_eig(key_nodes)
        #sum_betCen = sum_bet(key_nodes)
        sum_distance = sum_dist(key_nodes)
        write_list.append(sum_eigVec)
        write_list.append(sum_distance)
        #write_list.append(sum_betCen)
        allHOFs.append(write_list)

        #posDistance = getPosDistance(key_nodes)
        #negDistance = getNegDistance(key_nodes)
        #write_list.append(posDistance)
        #write_list.append(negDistance)
        #file_writer.writerow(write_list)
    #test_file.close()
        #return sum_eigVec, sum_distance
def writeFitness_pop(pop, i):
    filename_csv = "Fitness_values_pop_5_lesmis_"+str(i)+".csv"
    with open(filename_csv, 'wb') as test_file:
        file_writer = csv.writer(test_file)
        for ind in hof:
            write_list = []
            key_nodes = []
            for i in range(len(ind)):
                if (ind[i]==1):
                    key_nodes.append(i)
                    write_list.append(i)
            sum_eigVec = sum_eig(key_nodes)
            #sum_betCen = sum_bet(key_nodes)
            sum_distance = sum_dist(key_nodes)
            write_list.append(sum_eigVec)
            write_list.append(sum_distance)
            #write_list.append(sum_betCen)

            #posDistance = getPosDistance(key_nodes)
            #negDistance = getNegDistance(key_nodes)
            #write_list.append(posDistance)
            #write_list.append(negDistance)
            file_writer.writerow(write_list)
    test_file.close()

def sum_eig(key_nodes):
    sum_eig = 0.0
    for node in key_nodes:
        sum_eig = sum_eig + eigenvector[node]
    return sum_eig

#def sum_bet(key_nodes):
#    sum_bet = 0.0
#    for node in key_nodes:
#        sum_bet = sum_bet + betweenness[node]
#    return sum_bet

def sum_eig_former(key_nodes):
    sum_eig = 0.0
    for node in key_nodes:
        sum_eig = sum_eig + eigenvector_former[node]
    return sum_eig

#def sum_bet_former(key_nodes):
#    sum_bet = 0.0
#    for node in key_nodes:
#        sum_bet = sum_bet + betweenness_former[node]
#    return sum_bet

def sum_dist(key_nodes):
    sum_dist = 0.0
    for node in key_nodes:
        for node_a in key_nodes:
            dis = float(nx.dijkstra_path_length(G_, node_a, node))
            sum_dist = sum_dist + dis
    ave_dist = sum_dist/len(key_nodes)
    return ave_dist

def sum_dist_former(key_nodes):
    sum_dist = 0.0
    for node in key_nodes:
        for node_a in key_nodes:
            dis = float(nx.dijkstra_path_length(G, node_a, node))
            sum_dist = sum_dist + dis
    ave_dist = sum_dist/len(key_nodes)
    return ave_dist


def getNegDistance(selected):
    #print "Inside Get NEG distance\n"
    G_ = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
    G_.remove_nodes_from(selected)
    totaldistance = 0.0
    for node1 in G_.nodes():
        for node2 in G_.nodes():
            if node2 > node1:
                if (nx.has_path(G_, node1, node2)):
                    distance = nx.dijkstra_path_length(G_, node1, node2)
                    #print " ".join(map(str, ["Distance between ", node1, " - ", node2, " = ", distance]))
                    totaldistance = totaldistance + (1 / float(distance))
        #print " ".join(map(str, ["Total Distance ", totaldistance]))
    numnodes = len(G_.nodes())
    D_f = 1 - ((2 * totaldistance) / (numnodes * (numnodes - 1)))
    #print "Selected - "+str(selected)+" ++ D_f - "+ str(D_f)
    return D_f




def getPosDistance(selected):
    #print "Inside Get POS distance\n"
    totaldistance = 0.0
    #logging.info(" ".join(map(str, ["Selected Nodes - ", selected])))
    restnodes = set(G_.nodes()) - set(selected)
    for node1 in restnodes:
        mindis = 1000
        for node2 in selected:
            if (nx.has_path(G_, node1, node2)):
                distance = nx.dijkstra_path_length(G_, node1, node2)
                if (distance < mindis):
                    mindis = distance

        totaldistance = totaldistance + (1 / float(mindis))

    D_r = totaldistance / len(G_.nodes())
    #print "Selected - "+str(selected)+" ++ D_r - "+ str(D_r)
    return D_r





def main():
    #print "Inside Main"
    random.seed(time.clock())
    NGEN = 100
    MU = 50 #Number of populations
    LAMBDA = 100 #Number of individuals in the population
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", tools.mean)
    stats.register("std", tools.std)
    stats.register("min", min)
    stats.register("max", max)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    return pop, stats, hof


def findNomDoms(fileName, val, j, readfile):
    script = "nondom.py"
    saveto = readfile+"_"+val+"_"+j+"_nondoms.csv"
    runScript = script+" "+fileName+" "+"-d , -o 5-6 -M --output "+saveto
    #os.system(runScript)
    subprocess.Popen(runScript, shell=True)
    return saveto


def converttoFinalList(saveto, listBefore, listAfter):
    finalList = []
    reader = csv.reader(open(saveto, "rb"))
    for row in reader:
        if row != None and row != "":
            tempList = []
            #print row
            for i in range(0,5):
                valtoCheck = int(row[i])
                corValue = listBefore[valtoCheck]
                tempList.append(corValue)
            #print tempList
            finalList.append(tempList)
    return finalList

def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            #print values[0], values[1]
            dict[int(values[0])] = float(values[1])
        return(dict)


if __name__ == '__main__':
    precents = [0.1]
    readfile = 'facebook.csv'
    print "for "+readfile
    num_keyPlayers = 5
    G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
    start = 0
    G_ = nx.convert_node_labels_to_integers(G, first_label=start)
    eigenvector_former = readDict("facebook_eigenFile.txt", ",")
    print "Done calculating Eigenvector Centrality fullNetwork"
    #betweenness_former = readDict("facebook_betweenFile.txt", ",")
    #print "Done calculating betweenness Centrality fullNetwork"

    for val in precents:
        print "For value - "+str(val)
        for j in range(0, 10):
            allHOFs = []
            print "For network "+str(j)
            G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
            start = 0
            G_ = nx.convert_node_labels_to_integers(G, first_label=start)
            numNodes = len(nx.nodes(G_))
            precentToKeep = val
            keptNodes = int(math.ceil(float(numNodes)*precentToKeep))
            listBefore = []
            listAfter = []
            while True:
                nbunch = SamplingAlgorithms.SimilaritySample(G_, keptNodes)
                #nbunch = SamplingAlgorithms.SimilaritySample(G_)
                #nbunch = SamplingAlgorithms.topDegreeSubset("All_facebook.csv.csv", keptNodes)
                G_ = G_.subgraph(nbunch)
                connectedComponents = nx.number_connected_components(G_)
                print "Num components: "+str(connectedComponents)
                if connectedComponents > 1:
                    print "Taking the highest sub graph"
                    nbef = len(G_.nodes())
                    print "Nodes before - "+str(len(G_.nodes()))
                    highestCompNodes = 0
                    for comp in nx.connected_component_subgraphs(G_):
                        compNodes = len(comp.nodes())
                        if compNodes > highestCompNodes:
                            highestCompNodes = compNodes
                            G_ = comp
                    print "Nodes after - "+str(len(G_.nodes()))
                    naft = len(G_.nodes())
                    if naft > int(0.5*nbef):
                        break
                    else:
                        print "try again"
                        G_ = nx.convert_node_labels_to_integers(G, first_label=start)
                        continue
                    #G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
                    #start = 0
                    #G_ = nx.convert_node_labels_to_integers(G, first_label=start)
                    #continue
                else:
                    break
            writeme =  readfile+"_Old Nodes Order_"+str(val)+"_"+str(j)+"-"+str(G_.nodes())+"\n"
            listBefore = G_.nodes()
            print writeme
            start = 0
            G_ = nx.convert_node_labels_to_integers(G_, first_label=start)
            listAfter = G_.nodes()
            writeme =  readfile+"_New Nodes Order_"+str(val)+"_"+str(j)+"-"+str(G_.nodes())+"\n"
            print writeme
            eigenvector = nx.eigenvector_centrality_numpy(G_)
            print "Done calculating Eigenvector Centrality"
            #betweenness = nx.betweenness_centrality(G_)
            #print "Done calculating betweenness Centrality"



            for i in range(0,3):
                print "Run "+str(i)
                creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0,))
                creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
                toolbox = base.Toolbox()
                # Structure initializers
                toolbox.register("graph", graph2bin, Network=G_, num=num_keyPlayers)
                toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.graph)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("evaluate", evaluate)

                #toolbox.register("mate", tools.cxSelfPoints)
                toolbox.register("mate", tools.cxOnePoint)
                #toolbox.register("mate", tools.cxTwoPoints)
                toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
                #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
                toolbox.register("select", tools.selNSGA2)



                #print "Calling Main"
                [pop, stats, hof]=main()
                writeFitness_hof(hof, j, val, allHOFs)

            #Get allHOFs - remove dulplicates - find non doms - replace by order - find obj values - write to file
            allHOFs.sort()
            dupRemovedallHOFs = list(allHOFs for allHOFs,_ in itertools.groupby(allHOFs))
            filename_csv = readfile+"_"+str(val)+"_TestFile_"+str(j)+".csv"
            with open(filename_csv, 'wb') as filewrite:
                mycsv_writer = csv.writer((filewrite), quoting=csv.QUOTE_MINIMAL)
                for item in dupRemovedallHOFs:
                    mycsv_writer.writerow(item)
                filewrite.close()
            saveto = findNomDoms(filename_csv, str(val), str(j), readfile)
            time.sleep(5)
            finalLists = converttoFinalList(saveto, listBefore, listAfter)
            finalAll = []
            for keyplrset in finalLists:
                tempList = []
                tempList.extend(keyplrset)
                sumEigkey = sum_eig_former(keyplrset)
                #sumBetkey = sum_bet_former(keyplrset)
                sumDistkey = sum_dist_former(keyplrset)
                tempList.append('{:.9f}'.format(sumEigkey))
                tempList.append('{:.9f}'.format(sumDistkey))
                finalAll.append(tempList)
            final_filename_csv = readfile+"_"+str(val)+"finalFile_"+str(j)+".csv"
            with open(final_filename_csv, 'wb') as filewriter:
                my_final_csv_writer = csv.writer((filewriter), quoting=csv.QUOTE_MINIMAL)
                for item in finalAll:
                    my_final_csv_writer.writerow(item)
                filewriter.close()







    print "Done"