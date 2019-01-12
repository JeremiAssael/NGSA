import random
import numpy as np
import jgraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.metrics import classification_report
import nltk
import csv
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from itertools import islice


nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################
#
# random_predictions = np.random.choice([0, 1], size=len(testing_set))
# random_predictions = zip(range(len(testing_set)), random_predictions)
#
# with open("random_predictions.csv", "w") as pred:
#     csv_out = csv.writer(pred)
#     for row in random_predictions:
#         csv_out.writerow(row)
#
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

G = nx.Graph()

for row in range(6200, len(training_set)):
    line = training_set[row]
    line = line[0].split(' ')
    node_1 = line[0]
    node_2 = line[1]
    if line[2] == '1':
        G.add_edge(node_1, node_2)
    else:
        if node_1 not in G.nodes:
            G.add_node(node_1)
        if node_2 not in G.nodes:
            G.add_node(node_2)

training_set = [element[0].split(" ") for element in training_set]

with open("node_info_complementary.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

#edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

#nodes = IDs

## create empty directed graph
#g = igraph.Graph(directed=True)
 
## add vertices
#g.add_vertices(nodes)
 
## add edges
#g.add_edges(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=6200 + int(round(len(training_set)*0.05)))
training_set_reduced = [training_set[i] for i in to_keep]
print('Initial Len:', len(training_set_reduced))
training_set_to_test = training_set_reduced[:6200]
training_set_reduced = training_set_reduced[6200:6200+int(round(len(training_set)*0.05))]

print('New Len:', len(training_set_reduced), len(training_set_to_test))
# we will use three basic features:

# number of overlapping words in title
overlap_title = []
overlap_abstract = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []
degree_source = []
degree_target = []
triangles_source = []
triangles_target = []
shortest_path = []
counter = 0
nb_nb = []
for i in range(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    try:
        nb = nx.common_neighbors(G, source, target)
        nb_nb.append(len(sorted(nb)))
    except:
        print('--')
        print(index_source)
        print(source)
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]

    degree_source.append(source_info[6])
    degree_target.append(target_info[6])

    triangles_source.append(source_info[7])
    triangles_target.append(target_info[7])

    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_abstract = source_info[5].lower().split(" ")
    # remove stopwords
    source_abstract = [token for token in source_abstract if token not in stpwds]
    source_abstract = [stemmer.stem(token) for token in source_abstract]

    target_abstract = target_info[5].lower().split(" ")
    target_abstract = [token for token in target_abstract if token not in stpwds]
    target_abstract = [stemmer.stem(token) for token in target_abstract]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    overlap_abstract.append(len(set(source_abstract).intersection(set(target_abstract))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    try:
        paths = k_shortest_paths(G, source, target, 2)
        # print(paths[0])
        # print(paths[1])
        if len(paths[0]) == 2:
            shortest_path.append(len(paths[1]))
        else:
            shortest_path.append(len(paths[0]))
    except:
        shortest_path.append(-1)
    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")
# print(nb_nb)

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, overlap_abstract, temp_diff, comm_auth, degree_source, degree_target,
                              triangles_source, triangles_target, nb_nb, shortest_path]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
classifier = svm.LinearSVC()

# train
classifier.fit(training_features, labels_array)

clf = RandomForestClassifier(random_state=0, max_depth=10, n_estimators=3000)
clf.fit(training_features, labels_array)


model_xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=3000,
 max_depth=10,
 min_child_weight=10,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
model_xgb.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set


overlap_title_test = []
overlap_abstract_test = []
temp_diff_test = []
comm_auth_test = []
degree_target_test = []
degree_source_test = []
triangles_source_test = []
triangles_target_test = []
shortest_path_test = []
nb_nb_test = []
   
counter = 0
for i in range(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    # source = training_set_reduced[i][0]
    # target = training_set_reduced[i][1]
    #
    # index_source = IDs.index(source)
    # index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]

    degree_source_test.append(source_info[6])
    degree_target_test.append(target_info[6])
    try:
        nb = nx.common_neighbors(G, source, target)
        nb_nb_test.append(len(sorted(nb)))
    except:
        nb_nb_test.append(0)

    triangles_source_test.append(source_info[7])
    triangles_target_test.append(target_info[7])
    
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_abstract = source_info[5].lower().split(" ")
    source_abstract = [token for token in source_abstract if token not in stpwds]
    source_abstract = [stemmer.stem(token) for token in source_abstract]

    target_abstract = target_info[5].lower().split(" ")
    target_abstract = [token for token in target_abstract if token not in stpwds]
    target_abstract = [stemmer.stem(token) for token in target_abstract]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    overlap_abstract_test.append(len(set(source_abstract).intersection(set(target_abstract))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
   
    counter += 1
    if counter % 1000 == True:
        print (counter, "testing examples processsed")

    try:
        paths = k_shortest_paths(G, source, target, 2)
        if len(paths[0]) == 2:
            shortest_path_test.append(len(paths[1]))
        else:
            shortest_path_test.append(len(paths[0]))
    except:
        shortest_path_test.append(-1)

print(shortest_path_test)
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test, overlap_abstract_test, temp_diff_test, comm_auth_test,
                             degree_source_test, degree_target_test, triangles_source_test, triangles_target_test,
                             nb_nb_test, shortest_path_test]).T

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(clf.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

with open("improved_predictions.csv", "w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)

overlap_title_test = []
overlap_abstract = []
temp_diff_test = []
comm_auth_test = []
degree_target_test = []
degree_source_test = []
triangles_source_test = []
triangles_target_test = []
shortest_path_test = []
nb_nb_test = []

counter = 0
for i in range(len(training_set_to_test)):
    source = training_set_to_test[i][0]
    target = training_set_to_test[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # source = training_set_reduced[i][0]
    # target = training_set_reduced[i][1]
    #
    # index_source = IDs.index(source)
    # index_target = IDs.index(target)

    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    degree_source_test.append(source_info[6])
    degree_target_test.append(target_info[6])

    triangles_source_test.append(source_info[7])
    triangles_target_test.append(target_info[7])
    try:
        nb = nx.common_neighbors(G, source, target)
        nb_nb_test.append(len(sorted(nb)))
    except:
        nb_nb_test.append(0)
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_abstract = source_info[5].lower().split(" ")
    # remove stopwords
    source_abstract = [token for token in source_abstract if token not in stpwds]
    source_abstract = [stemmer.stem(token) for token in source_abstract]

    target_abstract = target_info[5].lower().split(" ")
    target_abstract = [token for token in target_abstract if token not in stpwds]
    target_abstract = [stemmer.stem(token) for token in target_abstract]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    overlap_abstract.append(len(set(source_abstract).intersection(set(target_abstract))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))

    try:
        paths = k_shortest_paths(G, source, target, 2)
        if len(paths[0]) == 2:
            shortest_path_test.append(len(paths[1]))
        else:
            shortest_path_test.append(len(paths[0]))
    except:
        shortest_path_test.append(-1)
    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")

testing_features = np.array([overlap_title_test, overlap_abstract, temp_diff_test, comm_auth_test, degree_source_test,
                             degree_target_test, triangles_source_test, triangles_target_test, nb_nb_test,
                             shortest_path_test]).T

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))
predictions_rf = list(clf.predict(testing_features))
predictions_xgb = list(model_xgb.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_to_test]
labels = list(labels)
labels_array = np.array(labels)

print(classification_report(labels_array, predictions_SVM))
print(classification_report(labels_array, predictions_rf))
print(classification_report(labels_array, predictions_xgb))


df_r = pd.read_csv('improved_predictions.csv', names=['id', 'category'])
df_r.to_csv('imp_pre.csv', index=False)
