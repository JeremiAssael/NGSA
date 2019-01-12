import pandas as pd
import csv
import tqdm
import networkx as nx
import pickle
import tqdm

df_corpus = pd.read_csv('node_info.csv', header=None)
df_corpus.columns = ['ID', 'year', 'title', 'authors', 'name of journal',  'abstract']
print(df_corpus.head())


with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

G = nx.Graph()

degrees_dict = {}
# print(training_set)

for row in range(6200, len(training_set)):
    line = training_set[row]
    line = line[0].split(' ')
    if line[2] == '1':
        node_1 = line[0]
        node_2 = line[1]
        G.add_edge(node_1, node_2)
        try:
            degrees_dict[node_1] += 1
        except KeyError:
            degrees_dict[node_1] = 1
        try:
            degrees_dict[node_2] += 1
        except KeyError:
            degrees_dict[node_2] = 1
    else:
        node_1 = line[0]
        node_2 = line[1]
        if node_1 not in G.nodes:
            G.add_node(node_1)
        if node_2 not in G.nodes:
            G.add_node(node_2)

triangles_dict = nx.triangles(G)
print('TRIANGLES:', triangles_dict)

nbr_nodes = len(degrees_dict.keys())
# for key in degrees_dict.keys():
#     degrees_dict[key] = degrees_dict[key] / nbr_nodes

print(degrees_dict)

df_corpus['degree'] = [0]*len(df_corpus)
df_corpus['triangles number'] = [0]*len(df_corpus)
df_corpus['common_neighbours'] = [0]*len(df_corpus)

for i in range(len(df_corpus)):
    node = df_corpus['ID'][i]
    degree = 0
    try:
        degree = degrees_dict[str(node)]
    except KeyError:
        pass
    df_corpus.set_value(i, 'degree', degree)
    triangle = 0
    try:
        triangle = triangles_dict[str(node)]
    except KeyError:
        pass
    df_corpus.set_value(i, 'triangles number', triangle)



# print('________________________')
# dict_total_nb = {}
# for i in tqdm.tqdm(range(len(df_corpus))):
#     cn_dict = {}
#     for j in range(len(df_corpus)):
#         try:
#             nb = nx.common_neighbors(G, str(df_corpus['ID'][i]), str(df_corpus['ID'][j]))
#         except:
#             print(df_corpus['ID'][i])
#             print(df_corpus['ID'][j])
#         cn_dict[df_corpus['ID'][j]] = nb
#     # df_corpus.set_value(i, 'common_neighbours', cn_dict)
#     dict_total_nb[df_corpus['ID'][i]] = cn_dict

df_corpus.to_csv('node_info_complementary.csv', index=False)
#
# with open('dict_nb', 'wb') as file:
#     my_pickler = pickle.Pickler(file)
#     my_pickler.dump(dict_total_nb)