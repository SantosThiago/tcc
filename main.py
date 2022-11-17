import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import time
import pandas as pd

class Graph(object):

    def __init__(self, node_List,num_Vehicles,capacity):
        self.node_List=node_List
        self.num_Vehicles=num_Vehicles
        self.capacity=capacity

    def distance(self, node1, node2):
        x1=node1.get_Pos()[0]
        x2=node2.get_Pos()[0]
        y1=node1.get_Pos()[1]
        y2=node2.get_Pos()[1]

        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return d

    def route_Distance(self, graph):
        total_Distance = 0

        for route in graph.keys():
            for i in range(len(graph[route])-2):
                j = i + 1
                a = graph[route][i]
                b = graph[route][j]
                d = self.distance(self.node_List[a], self.node_List[b])
                total_Distance += d

        return total_Distance

    def plot_Nodes(self):
        x=[]
        y=[]
        for i in range(len(self.node_List)):
            x.append(self.node_List[i].get_Pos()[0])
            y.append(self.node_List[i].get_Pos()[1])

        for i in range(len(self.node_List)):
            plt.annotate(node_List[i].get_Name(),xy=(x[i]-1,y[i]-2),color="blue")
            plt.plot(x[i],y[i],"go")

    def plot_Edges(self,graph):

        for route in graph.values():
            x=[]
            y=[]

            for i in range(len(route)):
                x.append(self.node_List[route[i]].get_Pos()[0])
                y.append(self.node_List[route[i]].get_Pos()[1])
                plt.plot(x, y, c=np.random.rand(3,))

        self.plot_Nodes()

    def imgOutput(self,graph,fileName):
        self.plot_Nodes()
        self.plot_Edges(graph)
        plt.savefig(fileName)
        plt.close()

    def sorted_byTime(self,graph):
        for i in graph.keys():
            by_Time = []
            for j in range(len(graph[i])):
                node=graph[i][j]
                by_Time.append([self.node_List[node].get_end_Time(),graph[i][j]])

            by_Time=sorted(by_Time)
            aux=[]

            for list in by_Time:
                aux.append(list[1])

            graph[i]=aux

        return graph

    def salesman(self,graph):
        current_Vehicle = 1
        current_Capacity = self.capacity
        total_Vehicles = self.num_Vehicles

        for i in graph.keys():
            l=0
            finish=0
            cluster_Visits=[0]
            current_Node=graph[i][l]
            while(True):
                distances = []

                for j in range(len(graph[i])):

                    node=graph[i][j]
                    d = self.distance(self.node_List[current_Node], self.node_List[node])
                    if d != 0:
                        distances.append([d,node])

                minor=min(distances)
                index=distances.index(minor)

                while minor[1] in cluster_Visits:
                    del distances[index]
                    minor = min(distances)
                    index = distances.index(minor)

                cluster_Visits.append(minor[1])

                if current_Capacity >= self.node_List[minor[1]].get_Demand():
                    current_Capacity -= self.node_List[minor[1]].get_Demand()

                else:
                    total_Vehicles -= 1
                    if total_Vehicles != 0:
                        current_Capacity = self.capacity
                        current_Vehicle += 1

                    else:
                        print("No more vehicles avaible")
                        break

                finish+=1
                current_Node=minor[1]

                if (finish == len(graph[i])):
                    break

            cluster_Visits.append(0)
            graph[i]=cluster_Visits
            l+=1

        return graph

    def salesman2(self,graph):
        current_Vehicle = 1
        current_Capacity = self.capacity
        total_Vehicles = self.num_Vehicles

        for i in graph.keys():
            l = 0
            finish = 0
            cluster_Visits = [0]
            current_Node = graph[i][l]
            if len(graph[i]) > 1:
                while (True):
                    distances = []

                    for j in range(len(graph[i])):
                        node = graph[i][j]
                        d = self.distance(self.node_List[current_Node], self.node_List[node])
                        if d != 0:
                            distances.append([d, node])

                    minor = min(distances)
                    index = distances.index(minor)

                    while minor[1] in cluster_Visits:
                        del distances[index]
                        minor = min(distances)
                        index = distances.index(minor)

                    cluster_Visits.append(minor[1])

                    if current_Capacity >= self.node_List[minor[1]].get_Demand():
                        current_Capacity -= self.node_List[minor[1]].get_Demand()

                    else:
                        total_Vehicles -= 1
                        if total_Vehicles != 0:
                            current_Capacity = self.capacity
                            current_Vehicle += 1

                        else:
                            print("No more vehicles avaible")
                            break

                    finish += 1
                    current_Node = minor[1]



                    if (finish == len(graph[i])):
                        break

            elif len(graph[i])==1:
                if graph[i][0] not in cluster_Visits:
                    cluster_Visits.append(graph[i][0])

                    if current_Capacity >= self.node_List[graph[i][0]].get_Demand():
                        current_Capacity -= self.node_List[graph[i][0]].get_Demand()

                    else:
                        total_Vehicles -= 1
                        if total_Vehicles != 0:
                            current_Capacity = self.capacity
                            current_Vehicle += 1

                        else:
                            print("No more vehicles avaible")
                            break

            cluster_Visits.append(0)
            graph[i] = cluster_Visits
            l += 1

        return graph

    #Heurística de varredura, utiliza o ângulo para a cluterziração dos nós
    def sweep(self):
        current_Cluster=1 #identificação dos clusters
        current_Capacity=self.capacity #capacidade atual para controle da capacidade de cada veículo
        x_Dep = self.node_List[0].get_Pos()[0] #posição x do depósito
        y_Dep = self.node_List[0].get_Pos()[1] #posição y do depósito
        polar_Angles=[] # lista dos ângulos
        nodes=[] #lista dos nós
        dict_clusters={} #dicionário dos clusters, realiza a paridade chave/valor entre os clusters e os nós daquele cluster
                         #exemplo: {1: [5,7,1,6],2: [4,2,8,3] } , cluster 1 possui os nós 5,7,1,6 ; cluster 2 possui os nós 4,2,8,3

        #percorre a lista de nós e realiza o cálculo dos ângulos polares de cada nó para que possa ser feita a clusterização do menor para o maior ângulo
        for i in range(len(self.node_List)-1):
            i=i+1 #a iteração começa a partir de 1 vai até n-1, onde n é o tamanho da lista de nós, porque na posição 0 está o depósito
            x_Node=self.node_List[i].get_Pos()[0] - x_Dep #define a posição x do nó para o cálculo do ângulo polar
            y_Node=self.node_List[i].get_Pos()[1] - y_Dep #define a posição y do nó para o cálculo do ângulo polar
            p=(x_Node)/(np.sqrt((x_Node)**2 + (y_Node)**2)) #fórmula de cálculo do ângulo polar 1
            polar_Angle=np.arccos(p)*180/np.pi #fórmula de cálculo do ângulo polar 2

            # caso a posição y do nó seja negativa, multiplica seu ângulo polar por -1
            if y_Node<0:
                polar_Angle*=-1

            polar_Angles.append([polar_Angle,i]) #insere uma lista de par, onde o primeiro elemento é o ângulo e o segundo o nó que possui aquele ângulo
                                                 #exemplo: [[ângulo7,nó3],[ângulo3,nó8]]

        polar_Angles=sorted(polar_Angles) #ordena os nós baseado de forma decrescente baseado no ângulo
                                          #exemplo: [[ângulo3,nó8], [ângulo7,nó3]]

        #percorre a lista de ângulos e realiza a distruibiação dos nós para os clusters
        for i in range(len(polar_Angles)):
            node=polar_Angles[i][1] #variável que cada nó

            #caso a capacidade atual seja maior que que a demanda do nó observado, a capacidade daquele veículo é diminuida pelo valor da demanda
            if current_Capacity>=self.node_List[node].get_Demand():
                current_Capacity-=self.node_List[node].get_Demand()

            # caso a capacidade do veículo seja menor que a demanda do nó
            else:
                dict_clusters[current_Cluster] = nodes #define que o cluster atual recebe todos nós que foram alocados para ele
                nodes=[] #reseta a lista de nós
                current_Capacity=self.capacity #reseta a capacidade atual
                current_Cluster+=1 #atualiza o cluster atual
                current_Capacity -= self.node_List[node].get_Demand() #atualiza a capacidade do veículo pelo ter sido atualizado e um nó estar esperando ser alocado

            nodes.append(node) #adiciona um nó na lista de nós, o nó sempre é adicionado independente da condições
                               #mesmo que um cluster não possua capacidade de receber um nó, quando o cluster atual é atualizado no else
                               #ele recebe o nó que não pode ser inserido no cluster anterior

        dict_clusters[current_Cluster]=nodes #define o último cluster receba todos nós que foram alocados para ele

        dict_clusters=self.sorted_byTime(dict_clusters) #chama a função de ordenação por tempo que recebe um grafo como parâmetro
                                                        #retorna o grafo ordenado
                                                        #ordena os nós de cada cluster em relação ao tempo de fechamento de cada nó

        dict_clusters=self.salesman(dict_clusters)  #chama a função do caixeiro viajante que recebe um grafo como parâmetro
                                                    #retorna o grafo completo
                                                    #realiza o caxeiro viajante de cada cluster

        return dict_clusters #retorna o grafo completo

    def saving(self):
        current_Cluster=1
        current_Vehicle=1
        current_Capacity=self.capacity
        total_Vehicles = self.num_Vehicles
        visits=[0,0]
        clusters=[]
        pair_List=[]
        gain_List=[]
        by_Time = []
        leftovers=[]
        dict_clusters={}

        for i in range(1,len(self.node_List)):
            for j in range(2,len(self.node_List)):
                if i!=j:
                    pair_List.append([i,j])

        for pair in pair_List:
            i=pair[0]
            j=pair[1]
            d1 = self.distance(self.node_List[0], self.node_List[i])
            d2 = self.distance(self.node_List[0], self.node_List[j])
            d3 = self.distance(self.node_List[i], self.node_List[j])
            d=d1+d2-d3
            gain_List.append([d,pair])

        gain_List=sorted(gain_List,reverse=True)
        visits.insert(1,gain_List[0][1][1])
        visits.insert(1,gain_List[0][1][0])

        while (len(visits)<((len(self.node_List))+1)):
            for elem in gain_List:
                    gain = elem[0]
                    edge = elem[1]

                    if edge[1] == visits[1] and edge[0] not in visits:
                        visits.insert(1, edge[0])

                    elif edge[0] == visits[-2] and edge[1] not in visits:
                            visits.insert(-1,edge[1])

                    else:
                        leftovers.append(elem)

            gain_List=leftovers.copy()

        for i in range(1,len(visits)-1):
            node=visits[i]
            if current_Capacity >= self.node_List[node].get_Demand():
                current_Capacity -= self.node_List[node].get_Demand()

            else:
                dict_clusters[current_Cluster] = clusters
                clusters = []
                current_Capacity = self.capacity
                current_Cluster += 1
                current_Capacity -= self.node_List[node].get_Demand()

            clusters.append(node)

        dict_clusters[current_Cluster] = clusters

        dict_clusters=self.sorted_byTime(dict_clusters)
        dict_clusters=self.salesman2(dict_clusters)

        return dict_clusters

    def fisher_jaikumar(self):
        current_Cluster = 1
        current_Vehicle = 1
        current_Capacity = self.capacity
        total_Vehicles = self.num_Vehicles
        visits = []
        seeds=[]
        clusters=[]
        capacity=[]
        dict_clusters = {}
        distances = []
        route=[]
        aux=np.random.choice(self.node_List,self.num_Vehicles)

        for elem in aux:
            seeds.append(elem.get_Name())

        for seed in seeds:
            capacity.append(current_Capacity)
            for i in range(1,len(self.node_List)):
                d=self.distance(self.node_List[i],self.node_List[seed])
                d1 = self.distance(self.node_List[0], self.node_List[i])
                d2 = self.distance(self.node_List[0], self.node_List[seed])
                d3 = self.distance(self.node_List[i], self.node_List[seed])
                d = d1 + d3 - d2
                distances.append([d,[i,seed]])

            distances = sorted(distances)

        for elem in distances:
            node=elem[1][0]
            seed=elem[1][1]

            ind=seeds.index(seed)

            if node not in visits:
                if capacity[ind] >= self.node_List[node].get_Demand():
                    capacity[ind] -= self.node_List[node].get_Demand()
                    clusters.append([ind,node])
                    visits.append(node)

        for i in range(len(visits)):
            node = visits[i]
            if current_Capacity >= self.node_List[node].get_Demand():
                current_Capacity -= self.node_List[node].get_Demand()

            else:
                dict_clusters[current_Cluster] = route
                route = []
                current_Capacity = self.capacity
                current_Cluster += 1
                current_Capacity -= self.node_List[node].get_Demand()

            route.append(node)

        dict_clusters[current_Cluster] = route
        dict_clusters = self.sorted_byTime(dict_clusters)
        dict_clusters = self.salesman2(dict_clusters)

        return dict_clusters

    def two_opt(self,g):
        for route in g.keys():
            improved = True

            while improved:
                improved = False

                for i in range(len(g[route])-4):
                    proxi = i + 1
                    for j in range(i+2, len(g[route])-2):
                        proxj = j + 1
                        a = g[route][i]
                        b = g[route][proxi]
                        c = g[route][j]
                        d = g[route][proxj]
                        if ((j != i) and (j != i - 1) and (j != i + 1)):
                            d1 = self.distance(self.node_List[a], self.node_List[b])
                            d2 = self.distance(self.node_List[c], self.node_List[d])
                            d3 = self.distance(self.node_List[a], self.node_List[c])
                            d4 = self.distance(self.node_List[b], self.node_List[d])

                            if (d1 + d2) > (d3 + d4):
                                g[route][proxi], g[route][j] = g[route][j], g[route][proxi]
                                improved = True

        return g

class Node(object):

    def __init__(self,node,name,demand,end_Time):
        self.x=node[0]
        self.y=node[1]
        self.name=name
        self.demand=demand
        self.end_Time=end_Time

    def get_Pos(self):
        return self.x,self.y

    def get_Name(self):
        return self.name

    def get_Demand(self):
        return self.demand

    def get_end_Time(self):
        return self.end_Time

def createNodeList(pos_List,demand_List):
    node_List = []
    j = 0

    for i in range(len(pos_List)):
        node = Node(pos_List[i], j,demand_List[i],end_Time[i])
        node_List.append(node)
        j = j + 1

    return node_List

def output(g,total_Distance,fileName):
    f = open(fileName, "w")

    for j in g.keys():
        string = str(j) + ":" + str(g[j]) + "\n"
        string2 = "\nTotal distance:" + str(round(total_Distance,1))
        f.write(string)
    f.write(string2)
    f.close()

def read_File(arq):
    pos_List=[]
    demand_List=[]
    end_Time=[]
    x=[]
    y=[]

    with open(arq,"r") as file:
        File = csv.DictReader(file,delimiter=";")

        for row in File:
            x.append(int(row["X"]))
            y.append(int(row["Y"]))
            demand_List.append(int(row["DEMAND"]))
            end_Time.append(int(row["DUEDATE"]))

        for i in range(len(x)):
            pos_List.append([x[i],y[i]])

        return pos_List,demand_List,end_Time

if __name__ == '__main__':
    start=time.time()
    path= "C:/Users/thiag/Desktop/tcc/instances/Vrp-Set-Solomon/"
    files=glob.glob(path+"*.txt")
    fileNames=[]
    i=0
    state=5

    if state==1:
        for file in files:
            split1=file.split(".")
            split2=split1[0].split("\\")
            split3=split2[1].split("V")
            split4=split3[0].split("C")
            split5=split3[1].split("I")
            capacity=int(split4[1])
            vehicles=int(split5[0])
            fileNames.append(split2[1])
            pos_List, demand_List, end_Time = read_File(file)
            node_List = createNodeList(pos_List, demand_List)
            graph = Graph(node_List, vehicles, capacity)
            g = graph.sweep()
            fileName="C:/Users/thiag/Desktop/tcc/results/Sweep/" + fileNames[i] + ".sweep"
            fileName2="C:/Users/thiag/Desktop/tcc/results/Sweep/" + "Sweep - " + fileNames[i] + ".png"
            total_Distance=graph.route_Distance(g)
            output(g,total_Distance,fileName)
            graph.imgOutput(g,fileName2)
            fileName3 = "C:/Users/thiag/Desktop/tcc/results/Sweep/" + fileNames[i] + "B" + ".sweep"
            fileName4 = "C:/Users/thiag/Desktop/tcc/results/Sweep/" + "Sweep - " + fileNames[i] + "B" + ".png"
            g2=graph.two_opt(g)
            total_Distance2=graph.route_Distance(g2)
            output(g2, total_Distance2, fileName3)
            graph.imgOutput(g2, fileName4)
            i+=1

    elif state==2:
        for file in files:
            split1=file.split(".")
            split2=split1[0].split("\\")
            split3=split2[1].split("V")
            split4=split3[0].split("C")
            split5=split3[1].split("I")
            capacity=int(split4[1])
            vehicles=int(split5[0])
            fileNames.append(split2[1])
            pos_List, demand_List, end_Time = read_File(file)
            node_List = createNodeList(pos_List, demand_List)
            graph = Graph(node_List, vehicles, capacity)
            g = graph.saving()
            fileName="C:/Users/thiag/Desktop/tcc/results/Saving/" + fileNames[i] + ".saving"
            fileName2="C:/Users/thiag/Desktop/tcc/results/Saving/" + "Saving - " + fileNames[i] + ".png"
            total_Distance=graph.route_Distance(g)
            output(g,total_Distance,fileName)
            graph.imgOutput(g, fileName2)
            fileName3 = "C:/Users/thiag/Desktop/tcc/results/Saving/" + fileNames[i] + "B" + ".saving"
            fileName4 = "C:/Users/thiag/Desktop/tcc/results/Saving/" + "Saving - " + fileNames[i] + "B" + ".png"
            graph.imgOutput(g,fileName2)
            g2=graph.two_opt(g)
            total_Distance2=graph.route_Distance(g2)
            output(g2, total_Distance2, fileName3)
            graph.imgOutput(g2, fileName4)
            i+=1

    elif state==3:
        for file in files:
            split1=file.split(".")
            split2=split1[0].split("\\")
            split3=split2[1].split("V")
            split4=split3[0].split("C")
            split5=split3[1].split("I")
            capacity=int(split4[1])
            vehicles=int(split5[0])
            fileNames.append(split2[1])
            pos_List, demand_List, end_Time = read_File(file)
            node_List = createNodeList(pos_List, demand_List)
            graph = Graph(node_List, vehicles, capacity)
            l=[]
            for j in range(50):
                g = graph.fisher_jaikumar()
                distance=graph.route_Distance(g)
                l.append([distance,g])
            best_distance=min(l)
            best_g=best_distance[1]
            fileName="C:/Users/thiag/Desktop/tcc/results/Fisher-Jakumar/" + fileNames[i] + ".fisher_jakumar"
            fileName2="C:/Users/thiag/Desktop/tcc/results/Fisher-Jakumar/" + "Fisher_Jakumar - " + fileNames[i] + ".png"
            total_Distance=graph.route_Distance(best_g)
            output(best_g,total_Distance,fileName)
            graph.imgOutput(best_g, fileName2)
            fileName3 ="C:/Users/thiag/Desktop/tcc/results/Fisher-Jakumar/" + fileNames[i] + "B" + ".fisher_jakumar"
            fileName4 ="C:/Users/thiag/Desktop/tcc/results/Fisher-Jakumar/" + "Fisher_Jakumar - " + fileNames[i] + "B" + ".png"
            graph.imgOutput(best_g,fileName2)
            g2=graph.two_opt(best_g)
            total_Distance2=graph.route_Distance(g2)
            output(g2, total_Distance2, fileName3)
            graph.imgOutput(g2, fileName4)
            i+=1

    elif state==4:
        columns=("Sweep",'Sweep-Two Opt',"Saving",'Saving-Two Opt',"Fisher-Jakumar",'Fisher-Jakumar-Two Opt')
        rows_1000 = []
        rows_700 = []
        rows_200 = []
        distances_Sweep_200 = []
        distances_Saving_200 = []
        distances_fisher_jakumar_200 = []
        distances_Sweep_opt_200 = []
        distances_Saving_opt_200 = []
        distances_fisher_jakumar_opt_200 = []
        distances_Sweep_700 = []
        distances_Saving_700 = []
        distances_fisher_jakumar_700 = []
        distances_Sweep_opt_700 = []
        distances_Saving_opt_700 = []
        distances_fisher_jakumar_opt_700 = []
        distances_Sweep_1000 = []
        distances_Saving_1000 = []
        distances_fisher_jakumar_1000 = []
        distances_Sweep_opt_1000 = []
        distances_Saving_opt_1000 = []
        distances_fisher_jakumar_opt_1000 = []

        for file in files:
            print(i+1,'/',len(files))
            split1=file.split(".")
            split2=split1[0].split("\\")
            split3=split2[1].split("V")
            split4=split3[0].split("C")
            split5=split3[1].split("I")

            if split4[1]=='1000':
                rows_1000.append(split2[1])
                capacity = int(split4[1])
                vehicles = int(split5[0])
                pos_List, demand_List, end_Time = read_File(file)
                node_List = createNodeList(pos_List, demand_List)
                graph = Graph(node_List, vehicles, capacity)
                g = graph.sweep()
                g2 = graph.saving()
                l=[]
                for j in range(50):
                    g3 = graph.fisher_jaikumar()
                    distance = graph.route_Distance(g3)
                    l.append([distance, g3])
                best_distance = min(l)
                new_g3 = best_distance[1]
                total_Distance = graph.route_Distance(g)
                total_Distance2 = graph.route_Distance(g2)
                total_Distance3 = graph.route_Distance(new_g3)
                distances_Sweep_1000.append(round(total_Distance))
                distances_Saving_1000.append(round(total_Distance2))
                distances_fisher_jakumar_1000.append(round(total_Distance3))
                g4 = graph.two_opt(g)
                g5 = graph.two_opt(g2)
                g6= graph.two_opt(new_g3)
                total_Distance4 = graph.route_Distance(g4)
                total_Distance5 = graph.route_Distance(g5)
                total_Distance6 = graph.route_Distance(g6)
                distances_Sweep_opt_1000.append(round(total_Distance4))
                distances_Saving_opt_1000.append(round(total_Distance5))
                distances_fisher_jakumar_opt_1000.append(round(total_Distance6))



            elif split4[1] == '700':
                rows_700.append(split2[1])
                capacity = int(split4[1])
                vehicles = int(split5[0])
                pos_List, demand_List, end_Time = read_File(file)
                node_List = createNodeList(pos_List, demand_List)
                graph = Graph(node_List, vehicles, capacity)
                g = graph.sweep()
                g2 = graph.saving()
                l=[]
                for j in range(50):
                    g3 = graph.fisher_jaikumar()
                    distance = graph.route_Distance(g3)
                    l.append([distance, g3])
                best_distance = min(l)
                new_g3 = best_distance[1]
                total_Distance = graph.route_Distance(g)
                total_Distance2 = graph.route_Distance(g2)
                total_Distance3 = graph.route_Distance(new_g3)
                distances_Sweep_700.append(round(total_Distance))
                distances_Saving_700.append(round(total_Distance2))
                distances_fisher_jakumar_700.append(round(total_Distance3))
                g4 = graph.two_opt(g)
                g5 = graph.two_opt(g2)
                g6 = graph.two_opt(new_g3)
                total_Distance4 = graph.route_Distance(g4)
                total_Distance5 = graph.route_Distance(g5)
                total_Distance6 = graph.route_Distance(g6)
                distances_Sweep_opt_700.append(round(total_Distance4))
                distances_Saving_opt_700.append(round(total_Distance5))
                distances_fisher_jakumar_opt_700.append(round(total_Distance6))


            elif split4[1] == '200':
                rows_200.append(split2[1])
                capacity = int(split4[1])
                vehicles = int(split5[0])
                pos_List, demand_List, end_Time = read_File(file)
                node_List = createNodeList(pos_List, demand_List)
                graph = Graph(node_List, vehicles, capacity)
                g = graph.sweep()
                g2 = graph.saving()
                l=[]
                for j in range(50):
                    g3 = graph.fisher_jaikumar()
                    distance = graph.route_Distance(g3)
                    l.append([distance, g3])
                best_distance = min(l)
                new_g3 = best_distance[1]
                total_Distance = graph.route_Distance(g)
                total_Distance2 = graph.route_Distance(g2)
                total_Distance3 = graph.route_Distance(new_g3)
                distances_Sweep_200.append(round(total_Distance))
                distances_Saving_200.append(round(total_Distance2))
                distances_fisher_jakumar_200.append(round(total_Distance3))
                g4 = graph.two_opt(g)
                g5 = graph.two_opt(g2)
                g6 = graph.two_opt(new_g3)
                total_Distance4 = graph.route_Distance(g4)
                total_Distance5 = graph.route_Distance(g5)
                total_Distance6 = graph.route_Distance(g6)
                distances_Sweep_opt_200.append(round(total_Distance4))
                distances_Saving_opt_200.append(round(total_Distance5))
                distances_fisher_jakumar_opt_200.append(round(total_Distance6))

            i+=1

        df1 = pd.DataFrame(zip(distances_Sweep_1000,distances_Sweep_opt_1000, distances_Saving_1000,distances_Saving_opt_1000,distances_fisher_jakumar_1000,distances_fisher_jakumar_opt_1000))
        df2 = pd.DataFrame(zip(distances_Sweep_700,distances_Sweep_opt_700, distances_Saving_700,distances_Saving_opt_700,distances_fisher_jakumar_700,distances_fisher_jakumar_opt_700))
        df3 = pd.DataFrame(zip(distances_Sweep_200,distances_Sweep_opt_200, distances_Saving_200,distances_Saving_opt_200,distances_fisher_jakumar_200,distances_fisher_jakumar_opt_200))
        dfs=[df1,df2,df3]
        row_L=[rows_1000,rows_700,rows_200]

        for i in range(3):
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            table=ax.table(cellText=dfs[i].values, rowLabels=row_L[i], colLabels=columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(18)
            table.scale(2, 2)
            #fig.tight_layout()
            plt.show()

    elif state==5:
        Sweep_optimization = []
        Saving_optimization = []
        Fisher_jaikumar_optimization = []
        distances_Sweep = []
        distances_Saving = []
        distances_fisher_jakumar = []
        distances_Sweep_opt = []
        distances_Saving_opt = []
        distances_fisher_jakumar_opt = []

        for file in files:
            print(i+1,'/',len(files))
            split1=file.split(".")
            split2=split1[0].split("\\")
            split3=split2[1].split("V")
            split4=split3[0].split("C")
            split5=split3[1].split("I")
            capacity=int(split4[1])
            vehicles=int(split5[0])
            pos_List, demand_List, end_Time = read_File(file)
            node_List = createNodeList(pos_List, demand_List)
            graph = Graph(node_List, vehicles, capacity)
            g = graph.sweep()
            g2 = graph.saving()
            l = []
            for j in range(50):
                g3 = graph.fisher_jaikumar()
                distance = graph.route_Distance(g3)
                l.append([distance, g3])
            best_distance = min(l)
            new_g3 = best_distance[1]
            total_Distance = graph.route_Distance(g)
            total_Distance2 = graph.route_Distance(g2)
            total_Distance3 = graph.route_Distance(new_g3)
            distances_Sweep.append(round(total_Distance))
            distances_Saving.append(round(total_Distance2))
            distances_fisher_jakumar.append(round(total_Distance3))
            g4 = graph.two_opt(g)
            g5 = graph.two_opt(g2)
            g6 = graph.two_opt(new_g3)
            total_Distance4 = graph.route_Distance(g4)
            total_Distance5 = graph.route_Distance(g5)
            total_Distance6 = graph.route_Distance(g6)
            distances_Sweep_opt.append(round(total_Distance4))
            distances_Saving_opt.append(round(total_Distance5))
            distances_fisher_jakumar_opt.append(round(total_Distance6))
            diff1 = total_Distance - total_Distance4
            diff2 = total_Distance2 - total_Distance5
            diff3 = total_Distance3 - total_Distance6
            percent1 = (diff1 * 100) / total_Distance
            percent2 = (diff2 * 100) / total_Distance2
            percent3 = (diff3 * 100) / total_Distance3
            Sweep_optimization.append(round(percent1,1))
            Saving_optimization.append(round(percent2,1))
            Fisher_jaikumar_optimization.append(round(percent3,1))
            i+=1

        print(sorted(Sweep_optimization))
        print(sorted(Saving_optimization))
        print(sorted(Fisher_jaikumar_optimization))

    else:
        file='C:/Users/thiag/Desktop/tcc/instances/Vrp-Set-Solomon\\C200V25I2.txt'
        split1 = file.split(".")
        split2 = split1[0].split("\\")
        split3 = split2[1].split("V")
        split4 = split3[0].split("C")
        split5 = split3[1].split("I")
        capacity = int(split4[1])
        vehicles = int(split5[0])
        fileNames.append(split2[1])
        pos_List, demand_List, end_Time = read_File(file)
        node_List = createNodeList(pos_List, demand_List)
        graph = Graph(node_List, vehicles, capacity)
        g = graph.sweep()
        fileName = "C:/Users/thiag/Desktop/tcc/results/" + fileNames[i] + ".sweep"
        fileName2 = "C:/Users/thiag/Desktop/tcc/results/" + "Sweep - " + fileNames[i] + ".png"
        total_Distance = graph.route_Distance(g)
        print('sweep:',total_Distance)
        output(g, total_Distance, fileName)
        graph.imgOutput(g, fileName2)
        fileName3 = "C:/Users/thiag/Desktop/tcc/results/" + fileNames[i] + "B" + ".sweep"
        fileName4 = "C:/Users/thiag/Desktop/tcc/results/" + "Sweep - " + fileNames[i] + "B" + ".png"
        g2 = graph.two_opt(g)
        total_Distance2 = graph.route_Distance(g2)
        print('sweep_opt:',total_Distance2)
        output(g2, total_Distance2, fileName3)
        graph.imgOutput(g2, fileName4)

        i += 1

    end=time.time()
    execution_Time=end-start
    print("Execution time:",round(execution_Time,1),"seconds")