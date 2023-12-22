+++
title = 'Graph Machine Learning : a deep dive'
date = 2023-12-18T11:59:55+01:00
draft = false
math = true
+++

<!-- 

![Alt text](../../images_post_1/kookaburra.jpg)

![Alt text](../../kookaburra.jpg) -->

# Acknowledgments

A heartfelt thanks to Professor Jure Leskovec and Stanford for offering the exceptional resource, CS224W. The majority of the content and images used in this blog post are derived from this course. I highly recommend following [this course](https://web.stanford.edu/class/cs224w/) if you wish to experience the same level of enjoyment and learning about Graph Machine Learning as I did.

Let's dive into it now.


# Introduction

Graphs are everywhere. A lot of today's problem can be formulated as
graph problems whether they are theoretical or practical. The reason is
simply that graphs are a very versatile mathematical structure.\
A graph is a collection of nodes connected by edges where each node
abstractly represents a certain entity and edges represent relationships
between these entities. There can be several types of nodes in a graph
and several types of relationships between them, that is what allows
them to model a lot of complex systems where components interact with
each other.\
From social networks and recommendation systems to transportation
networks, biology, and even language processing, graph representations
offer a powerful way to model and analyze complex systems.\
Graph problems are often defined at a specific level : node-level,
edge-level, subgraph-level and graph-level, we specify some very popular
problems at each level below

-   Node-level : Classifying persons in a social network as a male or
    female or even predict the age feature of each person.

-   Edge-level : Given the Amazon bipartite graph of items and users,
    suggest what items a particular user would like by training a model
    for a link prediction task.

-   Subgraph-level : Finding communities or clusters of similar nodes
    within a graph.

-   Graph-level : Chemical molecules constructed of simple atoms (nodes)
    and linked together by chemical bonds (edges) can be seen as simple
    graphs. One can then train a model to classify between toxic and
    non-toxic molecules.

Supervised ML requires data but what data do we have? When solving graph
problems, we most often have information about the nodes themselves,
which we refer to as features or attribute-features. These features can
include attributes like age, the number of friends, IP address, and so
on. Additionally, we have information about the edges in the graph,
including details about the relationships between nodes. As expected, we
also have access to the fundamental structural information of the
graphs, which is their topology. In some cases such as in the
semi-supervised setting, we may even have labels for a few nodes.\
As we said, nodes often have features and one can therefore wonder why
we do not solve graph problems as a usual tabular-data problem where the
dataset consists of all nodes features. Well, simply because we have
that relational structure information and we want to leverage it. How we
are going to do so will be be explained in the main part of the blog.\
What actually makes graphs challenging to study is :

-   They have an arbitrary size and therefore a simple MLP whose input
    size is fixed cannot possibly be fed different graphs.

-   There is no starting reference point one can use to order the nodes
    such as in a sentence or an image which are well structured objects.

-   They are dynamic, nodes and edges can both appear and disappear.

Thinking about it, graphs are one level of abstraction above images and text. Indeed, consider Figure [1](#fig:1). Images can be thought of as a simple grid-like graph, where pixels are nodes connected to their neighboring pixels. Text, on the other hand, can be seen as a simple linked list-like graph, with words or characters connected in a sequential order.

Even though we already have powerful tools to deal with images and text, developing tools to deal with graphs is not only beneficial for graph-specific tasks but also holds the potential for broader applications. Graph-based techniques can serve as a unifying framework that transcends different data types. Tools designed for analyzing and processing graphs can be adapted and extended to work with images and text as well.

<div style="text-align: center;">
  <img src="../../images/graph_vs_image_vs_text.PNG" alt="Graphs VS images VS text" width="800"/>
  <p><em>Figure 1: Graphs vs Images vs Text</em></p>
</div>

<!-- | ![Alt text](../../images/graph_vs_image_vs_text.PNG) |
|:--:|
| Figure 1: Graphs vs Images vs Text | -->



NB : If an image is not sourced, it means it was taken from the lecture [CS224W](https://web.stanford.edu/class/cs224w/), we just omit it for readability.




This blog aims at quickly reviewing traditional methods to solve graph problems and then delve into the graph machine learning world.  
Also, as I believe practical examples and problems are necessary for theory comprehension, we'll solve a simple problem of node classification using the Planetoid dataset. It is a simple graph-data set where each node represents a research paper and 2 nodes are connected if the papers cite each other. Each node contains 1433 features describing the paper and a 7-class label categorizing its research area.
Along this blog I'll post links to Colab notebooks where the problem is solved using the so-far learnt methods. You can also find all the notebooks along with an experiment notebook at the very end of the post.

Before jumping into it, let's briefly review graphs and define some clear notations that will be useful to us later.

# Graphs

A **graph** $G$ is mathematically defined as a couple $(V, E)$, where:

-   $V$ a represents a finite set of nodes (or vertices).

-   $E$ represents a finite set of edges. Each edge is a pair of nodes,
    i.e., $E \subseteq \{(u, v) \,|\, u, v \in V\}$.

Graphs can be directed or not, weighted or not, cyclic or not, connected
or not. Even though GML methods generally apply to all graphs, we'll
only consider simple undirected graphs.\
From a computer perspective, graphs are usually stored either with an
adjacency matrix or with an edge list :

-   The adjacency matrix A of the undirected and unweighted graph G is a
    matrix A $\in \{0,1\}^{|V| \times |V|}$ where $A_{i,j}=1$ if
    $(i,j) \in E$ and $A_{i,j}=0$ otherwise

-   The edge list of the undirected and unweighted graph G is a list
    that contains all edges of the graph without redundancy. For
    example, $[(a,b), (c,d), ...]$ where a,b,c,d $\in V$ and $(a,b)$,
    $(c,d) \in E$

We also said that in our context, each node had a feature vector, an
attribute vector, that would describe it independently of the graph
structure, we call it $\mathbf{x} \in \mathcal{R}^{m}$. For instance,
the feature vector for a particular node in a social network could be
$\mathbf{x} = [1, 20, 85, 765, 45]$ where the values represent
attributes such as gender (1 for male), age (20 years), number of
friends (85), posts (765), and daily average usage time (45 minutes).\

The feature matrix of the graph, which is nothing but the stack of
feature vectors of all nodes is
$\mathbf{X} \in \mathcal{R}^{m \times |V|}$.

$\textbf{Notebook 1}$ : A classifier that does not look at graph relations [MLP classifier on graph-independent features](https://colab.research.google.com/drive/1Dr1LqLq00RsmGwn5byqU7-_FNy6jBuIn?usp=sharing).

# What is GML and why GML

You probably all know that GML is a sub-field of ML that focuses on the
development of algorithms and models to make predictions on
graph-structured data but you may not know why we actually need it.
There are actually a few reasons we need GML.\
First, traditional algorithms tend to have a large trade-off between the
algorithm ability to solve a task with good precision and its time
complexity, they are thus usually not deployed on very large graphs
except for a few cases.\
Second, most ML pipelines contain a data-preprocessing and feature
design step that is usually more impactful on performance than the
choice of the model itself.

One conceivable approach involves appending graph-related information to
the feature vector, denoted as $\mathbf{x}$, and subsequently inputting
it into a machine learning model. For instance, in a node classification
task, one might append the corresponding row from the adjacency matrix,
describing the node's neighborhood, to the feature vector. The augmented
vector could then be fed into a fixed-size MLP responsible for node
classification.

However, this approach encounters two notable challenges. Firstly, the
adjacency row provides only first-order information about the node,
lacking details about the neighbors of its neighbors. Second, designing
appropriate graph features is a task dependent on the specific problem
and is often resource-intensive. The chosen graph features can introduce
bias into the model, influencing it towards a particular learning mode.

In short, the need of GML rises from

-   The limited capacities of traditional methods

-   The significant amount of time and effort expended on
    designing/tailoring appropriate graph features for each project,
    which can ultimately introduce unintended bias into our models


<p align="center">
  <img src="../../images/pipeline_change.png" alt="New pipeline" width="600"/>
</p>
<p align="center"><em>Figure 2: We want to go from the traditional ML pipeline based on feature engineering to a more automatic pipeline which learns the useful features itself</em></p>



<!-- | ![New pipeline](../../images/pipeline_change.png) |
|:--:|
| Figure 2: We want to go from the traditional ML pipeline based on feature engineering to a more automatic pipeline which learns the useful features itself |
 -->





Instead of carefully designing and engineering features, wouldn't it be
more advantageous to develop a graph model capable of autonomously
deriving the most suitable features for the specific downstream task? I
guess yes, an overview of the previous paragraph is on figure
[2](#fig:2).
That is why Graph Neural Networks (GNNs) were designed, instead of
feeding feeding a ML model carefully designed features, we would simply
feed the whole graph to a GNN and feed the embeddings it produces to a
ML model.\
Let's already make things clear : GNNs **were designed to embed data**
and not to directly classify, they simply produce embeddings which are
1D vectors of real numbers.\
Technically they can also be used as classifiers by setting their final
embedding dimension to k in a k-class prediction problem but one would
have to pass them through a softmax layer to have a well defined
probability distribution. So yeah, they can be used as classifiers but
they are better understood as simple encoders whose point is to
meaningfully compress data.\
Embeddings, as suggested by fig
[3](#fig:3), are defined at the node level but
we can also use them to define embeddings at the graph level by simply
aggregating all the embeddings of the nodes in the graph.\
I repeat : GNNs simply map a node to an embedding or a graph to an
embedding.\
The resulting embeddings can be used for visualization, classification,
prediction or any downstream task.\
Thinking about it, fully CNNs are not pure classifiers either, they
provide a bunch of feature maps and it is then usually the job of an MLP
to use these feature maps to classify the input image.\
In this blog, we'll see how to define GNNs, how they work, how to train
them and compare them to traditional models.

<p align="center">
  <img src="../../images/basic_node_embedding.PNG" alt="Node embedding" width="600"/>
</p>
<p align="center"><em>Figure 3: Embedding of the node $u$ using a GNN</em></p>


<!-- | ![Node embedding](../../images/basic_node_embedding.PNG) |
|:--:|
| Figure 3: Embedding of the node $u$ using a GNN | -->


What will we cover ? We'll start by reviewing a few traditional methods
for extracting relevant features and passing knowledge through the
network. We'll later use these methods to benchmark GNNs . You can skip
that part if you're only here to learn about GNNs. Essentially the next
3 sections are all about compressing graph information to small
embeddings. This is what GML is all about actually and if there is one
thing you should remember from this blog, this is it.\
We then focus on defining in detail what a GNN is, how it is composed
and how we can stack GNN layers as well as describe a few main
implementations of GNNs. We then theoretically analyze them and
implement a few ones along the way.

# Traditional graph features  

We here study the traditional techniques to extract relevant information
about nodes, edges and sub-graphs in a graph. These graph-features along
with attribute-features are both aggregated to define the aggregated
node feature vector that will be used for the downstream task. They can
be thought of as embeddings of nodes, edges and sub-graphs.
$$\mathbf{x_{agg}} = AGG(\mathbf{x_{att-features},x_{graph-features})}$$
In this section, we simply seek to define a good graph-features vector
$\mathbf{x_{graph-features}}$ that will contain knowledge about its
neighbourhood, about the graph topology but **no** information inherent
to the node itself. Actually we can define graph-features at the node
level but also at the edge-level and graph-level.

## Traditional graph features : node-level

We here define some graph features that could be useful for describing a
specific node $v \in V$, they typically include :

-   Node degree

-   Node eigen centrality : A node $v$ is important if it is surrounded
    by important nodes $u \in N(v)$, the eigen centrality of a node $v$
    is $$c_v = \frac{1}{\lambda} \sum_{u \in N(v)}c_u$$ and the
    centrality vector is the eigenvector of the adjacency matrix with
    largest eigenvalue $$\lambda \mathbf{c} = \mathbf{A} \mathbf{c}$$

-   Node between centrality : A node $v$ is important if it lies on many
    shortest paths between all nodes in V, the between centrality of a
    node $v$ is
    $$c_v = \sum_{s\neq v\neq t} \frac{\\#(Shortest\ paths\ between\ s\ and\ t\ that\ contain\ v)}{\\#(Shortest\ paths\ between\ s\ and\ t)}$$
    

-   Closeness centrality : A node $v$ is important if it has a small
    shortest path to all other nodes in V, the closeness centrality of a
    node $v$ is
    $$c_v = \sum_{u\neq v} \frac{1}{\\#(Shortest\ paths\ between\ u\ and\ v)}$$
    See figure [3](#fig:3) for a practical example of the
    centrality measures

<div style="text-align: center; display: flex; flex-direction: row; justify-content: center;">
  <img src="../../images/eigen_centrality.png" alt="Eigen centrality" width="250"/>
  <img src="../../images/between_centrality.png" alt="Between centrality" width="250"/>
  <img src="../../images/closeness_centrality.png" alt="Closeness centrality" width="250"/>
</div>
<p style="text-align: center;"><em>Figure 3: Three centrality measures computed on the same graph</em></p>
<p style="text-align: center;">Image source: [aksakalli.github.io](https://aksakalli.github.io/2017/07/17/network-centrality-measures-and-their-visualization.html)</p>


<!-- 
    <div align="center">

    | ![Eigen Centrality](../../images/eigen_centrality.png) | ![Between Centrality](../../images/between_centrality.png) | ![Closeness Centrality](../../images/closeness_centrality.png) |
    | --- | --- | --- |
    | *Figure 3: Three centrality measures computed on the same graph* | | |

    Image source: [Network Centrality Measures and Their Visualization](https://aksakalli.github.io/2017/07/17/network-centrality-measures-and-their-visualization.html)

    </div> -->

-   Clustering coefficient : Measures the degree of inter-connectedness
    among the neighbors of vertex $v$, the clustering coefficient of a
    node $v$ is
    $$e_v = \frac{\\#(edges\ among\ neighbouring\ nodes)}{\binom{k_v}{2}}$$

    where $k_v$ denotes the number of neighbours of $v$

-   Graphlet degree vector : Given a list of graphlets \[a,b,c,\...\]
    and a node $v \in V$, we define GDV(v) = \[x,y,z\] as the graphlet
    degree vector which which means that v participates in

    -   x instances of the graphlet a

    -   y instances of the graphlet b

    -   z instances of the graphlet c

### A small digression on graphlets

A graphlet is a rooted, connected, non-isomorphic subgraph, which can be
thought of as a motif. There is 1 graphlet of size 2, 3 graphlets of
size 3, and 6 graphlets of size 4, and so on. We say that a node $v$
participates in a graphlet, denoted as $g$, if we can find the graphlet
$g$ in the neighborhood of node $v$ with one of the anchors of $g$
located at node $v$.\
The first 72 condensed graphlets and the GDV of a simple graph are
represented on figure [4](#fig:4)


<p align="center">
  <img src="../../images/graphlets.png" alt="Graphlets" width="600"/>
</p>
<p align="center"><em>Figure 4: First 72 graphlets drawn in a condensed way</em></p>


<!-- | ![Graphlets](../../images/graphlets.png) |
|:--:|
| Figure 4: First 72 graphlets drawn in a condensed way | -->

## Traditional graph features : edge-level

A very common task in graph problems is link prediction, this task
however requires features defined at the link level on existing edges.\
We here define some edge features that could be useful for describing a
specific edge $(u,v) \in E$, they typically include :

-   Common neighbours : Counts the number of common neighbors between u
    and v $$|N(u) \cap N(v)|$$

-   Jaccard coefficient : $$\frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

-   Adamic-Adar index :
    $$\sum_{x \in N(u) \cap N(v)} \frac{1}{\log(k_x)}$$
    where $k_x$ denotes the degree of node $x$.

    The issue with these 3 coefficients is that for all vertices $u$ and $v$ that are more than 2 hops away from each other, $N(u) \cap N(v) = \emptyset$. It may be beneficial for us to design features that characterize the link between 2 nodes even if a direct edge between them is not in $E$

-   Katz index : Counts the number of paths of all lengths between $u$
    and $v$ $$S_{u,v} = \sum_{l=1}^{l=\infty} \beta^l A_{u,v}^l$$ where
    $\beta \in [0,1]$ and A is the adjacency matrix of the graph

## Traditional graph features : graph-level

Instead of designing a feature vector for a graph, we instead design
kernels. A kernel $K(G,G') \in \mathcal{R}$ measures the similarity
between G and G'. The kernel K is defined by means of a feature mapping
function $\phi$ such that $K(G,G') = \phi(G)^T \phi(G')$ where $\phi$
maps graphs to vectors and the inner product is used as the usual
similarity measure.\
A simple kernel function is the bag of node degree function which takes
a graph G and maps it to a degree vector where the i-th output counts
the number of nodes with degree i in G. An example is given on figure
[5](#fig:5)


<p align="center">
  <img src="../../images/bag_of_node_degree.png" alt="Bag of node degree" width="600"/>
</p>
<p align="center"><em>Figure 5: Bag of node degree kernel function</em></p>


<!-- | ![Bag of node degree](../../images/bag_of_node_degree.png) |
|:--:|
| Figure 5: Bag of node degree kernel function | -->



Taking it further, instead of mapping a graph to a bag of node degree,
we could map it to a bag of graphlets.\
Given a graph G and a graphlets list $\mathcal{G} = (g_1, g_2, g_n)$,
the graphlets count vector $f_{\mathcal{G}} \in \mathcal{N}^n$ is
defined as
$$(f_{\mathcal{G}})_i = \\#(g_i \subseteq G)\ \forall i \in [1,n]$$ 
That
is, for each graphlet $g_i$, you simply count the number of instances of
$g_i$ you find in G. Note that in this setting, the graphlets are not
constrained to being connected nor rooted. An example is given on figure
[6](#fig:graphlet_kernel).

<p align="center">
  <img src="../../images/graphlet_kernel.png" alt="Graphlet kernel" width="400"/>
</p>
<p align="center"><em>Figure 6: Computing the graphlets vector for a simple graph G</em></p>


<!-- | ![Graphlet kernel](../../images/graphlet_kernel.png) |
|:--:|
| Figure 6: Computing the graphlets vector for a simple graph G | -->



NB : Counting graphlets of size k in G is $O(n^k)$!

A more advanced and more efficient kernel is the Weisfeiler-Lehman
kernel, also known as the color-refinement algorithm. However, a
detailed discussion of it is beyond the scope of this context. For more
information, you can refer to the presentation available at the
following link: [Weisfeiler-Lehman kernel](https://ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/CA10_WeisfeilerLehman.pdf)

The graph-features we've seen at the 3 levels can be all aggregated to
the attribute-features and the aggregated features can be seen as
equivalent to embeddings except that they are designed separately for
each problem while embeddings tend to be learnt rather than defined.\
Now that you have pre-processed your data and carefully designed
appropriate features, you can feed them to a ML model to solve the
problem at hand.


$\textbf{Notebook 2 :}$ A classifier using graph-dependent features along with graph-independent features [MLP classifier on graph-dependent and graph-independent features](https://colab.research.google.com/drive/1yRzcHPtHrlpeH000xilYn_5hlo0xVcgM?usp=sharing).

$\textbf{Notebook 3 :}$ : Same as notebook 2 except we pass the processed features through an MLP that preserves size [MLP classifier on MLP embeddings](https://colab.research.google.com/drive/1He6LH7bAVbWfnPyIUZgPsbA5DKN831nt?usp=sharing)

# Embeddings by random walks

In this section, we seek to define powerful embeddings independently of
the graph problem at hand. We first review the traditional techniques
used to compute embeddings and then see how GNN's were created from
there.\
Given a graph G, we want an encoding function f :
$v \mapsto \mathcal{R}^d$ that maps a node to an embedding which is a
1-d vector of length d where d is an arbitrary parameter, an overview is
on figure [7](#fig:7).


<p align="center">
  <img src="../../images/embeddings_of_u_and_v.png" alt="Embeddings of u and v" width="600"/>
</p>
<p align="center"><em>Figure 7: Embeddings $z_u=f(u)$ and $z_v=f(v)$</em></p>


<!-- | ![Embeddings of u and v](../../images/embeddings_of_u_and_v.png) |
|:--:|
| Figure 7: Embeddings $z_u=f(u)$ and $z_v=f(v)$ | -->


We want a meaningful encoder though, an encoder that maps similar points
to similar embeddings. I believe it makes sense, if 2 nodes are close
neighbours and have similar attribute-features, we want them to be close
and similar in the embedding space. What does it actually mean for 2
nodes to be similar and for 2 embeddings to be similar ?\
In the embedding space as well as in latent spaces in general,
similarity is often measured by means of a dot product, note that in an
attention layer, the similarity between the key and the query is also
measured by a dot product.\
In the original network, defining similarity is more challenging, and we
cannot merely rely on the dot-product of attribute features. This
approach has limitations for two key reasons. First, it does not
consider the underlying graph structure, which is essential in many
graph-related problems. Second, some attributes are categorical, and
converting them into n-ary indicators, i.e one-hot encodings, may not
provide a robust solution, as the scales of these attributes can vary
significantly. To define similarity in a graph, we'll refer to an
important method in graph problems : random walks.\
To make things more tangible, let me introduce you to a basic encoder :
the shallow look-up encoder $\mathbf{Z}$ $\in R^{d \times |V|}$. It
essentially maps nodes to embeddings this way : if node 5 wants its
embedding, I simply give him my fifth column. Mathematically,

$$ENC(v) = z_v = Z * v$$ 

Where $v \in \{0,1\}^{|V|}$ is essentially an
indicator vector filled with 0's, except at its v-th index, which
contains a 1.\
In this setup, we aim to optimize the embedding of each node directly.
We seek the matrix $\mathbf{Z}$ that can best encode these nodes. We
optimize $\mathbf{Z}$ by maximizing:

$$\max z_u \cdot z_v \quad \text{subject to} \quad (u,v) \text{ being similar nodes}$$

Ok great formulation except we yet don't know what it mathematically
means for 2 nodes to be similar. Well, that's where random walks come
into play.

## Random walk for node embeddings

Before delving into the details, we warn you that the random walks we
consider **do not** consider the node attribute-features nor its label
to construct embeddings.\
A random walk is a stochastic process used to traverse a network from
one neighbor to another by following a certain strategy or policy,
denoted as R, that instructs which neighbor to visit at each node.
Consider the example on figure [8](#fig:8).

<p align="center">
  <img src="../../images/random_walk_example.png" alt="Random walks" width="500"/>
</p>
<p align="center"><em>Figure 8: Starting at node 8, an example of RW of length 10 is $\{8,12,5,19,17,10,19,5,15,16\}$. An example of RW of length 10 that starts at 8 could also be $\{8,1,4,2,12,11,13,9,16,18\}$.</em></p>


<!-- | ![Random walks](../../images/random_walk_example.png) |
|:--:|
| Figure 8: Starting at node 8, an example of RW of length 10 is $\{8,12,5,19,17,10,19,5,15,16\}$. An example of RW of length 10 that starts at 8 could also be $\{8,1,4,2,12,11,13,9,16,18\}$. | -->


The idea is that if 2 nodes meet each other on a lot of random walks,
i.e, their frequency of co-occurrence on random walks is high, they are
**similar**. On figure
[8](#fig:8), nodes 12 and 8 will tend to
co-occur quite a lot on their random walks because they are pretty close
while nodes 21 and 3 will very rarely co-occur on their random walks,
simply because they are far from each other.\
If $u$ and $v$ tend to co-occur a lot on each other's random walks, we
want $z_u^T \cdot z_v$ to be high while if $u$ and $v$ rarely co-occur
on each other's random walks, we want $z_u^T \cdot z_v$ to be low. That
is, we want $$z_u^T \cdot z_v \propto P_R(v|u)$$ Where $P_R(v|u)$
denotes the probability of observing $v$ by starting a random walk from
$u$ while following a strategy R.\
We can cast the problem of finding optimal embeddings as a simple
maximum likelihood problem: Given $G = (V, E)$, and let $N_R(u)$ denote
the neighborhood of $u$ using a random walk strategy $R$, we aim to
learn an encoder $f$ such that $z_u = f(u)$ with the objective:

$$\max_f \sum_{u \in V} \log P(N_R(u) | z_u)$$

Equivalently,

$$\max_f \sum_{u \in V} \sum_{v \in N_R(u)} \log P(v | z_u)$$

What does this mean?

We want to train an encoder $f$ in such a way that the embedding $z_u$
is as predictive as possible of its neighborhood. If $z_u$ is
sufficiently accurate, we can use it to predict with good accuracy the
actual neighborhood in the initial graph.\
Regarding $P(v | z_u)$, as it is a distribution, we can simply
parameterize it by a softmax layer :

$$P(v | z_u) = \frac{exp(z_u^T \cdot z_v)}{\sum_{n \in V} exp(z_u^T \cdot z_n) }$$

To frame the problem in a ML way, we can define the following loss
function directly derived from the MLE objective :

$$L = -\sum_{u \in V} \sum_{v \in N_R(u)} \log P(v | z_u)$$ 
And we can find the optimal embeddings by minimizing this loss function with
gradient descent.\
The overall recipe to find optimal embeddings using random walks is :

1.  Run several random walks by following a strategy R for each node and
    store them

2.  Collect $N_R(u)$ the multi-set (set with repeating elements) of
    nodes as the merge of all random walks for each node

3.  Initialize randomly the encoder f

4.  Optimize f by minimizing L with GD :
    $z_u \leftarrow z_u - \lambda \frac{\partial L}{\partial z_u}$

So far we assumed the random walk strategy/policy was given, let's now
breakdown 2 of the most popular ones : DeepWalk and Node2Vec.

## Random walk strategies

As we saw above, we can define the notion of similarity between 2 nodes
in the original network by means of their frequency of co-occurrence in
each others' random walks. We can then train embeddings using that
notion of similarity. We have to note that the embeddings will be
extremely dependent on the random walk strategy R that was used. If the
random walk strategy is voluntarily poorly designed, the similarity
definition used will be meaningless and so will be the embeddings.\
A good random walk strategy is essential to solid downstream
performances.\
The first and simplest strategy is DeepWalk : the transition
distribution across neighbours is uniform. That is, the random walks
from a node are simply created by moving from the current node to the
next by choosing a neighbor uniformly at random.

A clear issue we can directly see is the expressivity of DeepWalk :
there are no tunable parameters, the transition distribution is always
uniform.\
Node2Vec overcomes this issue by defining the \"return\" and the
\"walk-away\" parameters that allow one to bias at will the random
walks. In fixed length random walks, it is not easy to have both a good
local view of the network and a good global view of the network. The
reason is simply that as you have a finite number of hops per walk, you
can either decide to move around your neighborhood a lot or try to
explore most of the network by leaving your neighborhood.\
A BFS algorithm with fixed length (fixed number of nodes to discover)
will give you a good local view of the neighborhood while a DFS
algorithm will give you a weak local but decent global view of the
network. On figure [9](#fig:9), we observe that the DFS strategy allows to
dive deep in the network while the BFS strategy simply learns about the
neighborhood. One is not strictly better than the other, the valuable
asset is that one might prefer shallow random walks and deep random
walks depending on the downstream task.


<p align="center">
  <img src="../../images/node2vec.PNG" alt="Node2Vec" width="600"/>
</p>
<p align="center"><em>Figure 9: 2 network discovery strategies with a length of 3: BFS and DFS</em></p>


<!-- | ![Node2Vec](../../images/node2vec.PNG) |
|:--:|
| Figure 9: 2 network discovery strategies with a length of 3: BFS and DFS | -->


These biases are explicited by means of the \"return\" parameter p and
the \"walk-away\" parameter q. To understand them, consider the
following example : on figure
[10](#fig:10), suppose we just moved from $s_1$ to $w$.
The random walker has 4 options :

1.  Go back to $s_1$ increase the local view around $s_1$

2.  Go to $s_2$ which is as far from $s_1$ as $w$ is, i.e, they are both
    one hop away from $s_1$

3.  Go to $s_3$ which means getting far from $s_1$

4.  Go to $s_4$ which means getting far from $s_1$

If we want our walks to only explore the shallow neighborhood around us,
we have to set a low value of p and a high value of q such that walkers
don't move far away from us. On the other hand, if we want our walks to
deeply explore the network, we need to set of high value of p and a low
value of q such that walkers tend to go as far as possible from us.
Note that in this case the transition distribution is not a proper
distribution as it is not normalized.

<p align="center">
  <img src="../../images/node2vec_p_q.PNG" alt="Node2Vec parameters" width="600"/>
</p>
<p align="center"><em>Figure 10: Node2Vec p and q parameters</em></p>


<!-- | ![Node2Vec parameters](../../images/node2vec_p_q.PNG) |
|:--:|
| Figure 10: Node2Vec p and q parameters | -->


## Shortcomings of random walks

As you have probably guessed, random walks for embeddings have quite a
few limitations, otherwise GNN's would have most likely not been
necessary. You might even have already identified some weak points. We
briefly enumerate some of them below :

1.  Node features are not considered : The nodes' and edges' labels and
    features are not considered. Embeddings are learnt such that they
    allow to predict the neighborhood of a node pretty well. They rely
    on neighborhoods defined by random walks but random walks **DO NOT**
    consider at all the features and labels information we have about
    nodes and edges.

2.  Inference and evaluation are not easy : It is not clear how one
    computes embeddings for nodes that were not in the training set.
    Typically, one would simply feed the node to the encoder $f$ but how
    exactly? In the case of a shallow encoder this is impossible since
    it stores a column per training node and returns column k when fed
    node k however it stores no column embeddings for nodes that it did
    not see in the training set.

3.  Sensitivity to parameters : The quality of embeddings is highly
    dependent on the p and q parameters values and tuning them
    appropriately is usually time-taking. The size of random walks
    parameters is also crucial and tough to tune.

4.  Complexity : Running a large number of random walks on a large
    number of nodes can be particularly long.

# Message Passing

One last concept we haven't covered in this blog which might help you
understand GNN is message passing. Message passing is a communication
method between nodes to help them know more about the global information
on the graph. It is sometimes referred to as \"belief propagation\" and
you'll most likely understand it better in the context of node
classification.\
The basic idea is as follows: each node gathers messages from its
neighbors to update its belief regarding itself and its belief of
others. Then, it passes these updated beliefs forward to its neighbors.\
Some notations for message passing :

-   Label-label potential matrix $\psi$ :
    $\psi(Y_i,Y_j) \propto P(node j_c=Y_j|neighbour\ node \ i_c = Y_i)$.
    This matrix captures the class dependency between a node and its
    neighbours, i.e, if I'm of class $Y_j$, what are the class
    distributions of each of my neighbours.

-   Prior belief $\phi$ : $\phi(Y_j) \propto P(node j_c=Y_j)$.

-   Message $m_{i \to j}(Y_j)$ : i's message/estimate/belief of node $j$
    belonging to class $Y_j$.

-   $\mathcal{L}$ is the set of classes/labels.

The algorithm to propagate beliefs through the node is the following :

1.  Initialize all messages to 1.

2.  Repeat for each node :
    $$
    m_{i \to j}(Y_j) = \sum_{Y_i \in \mathcal{L}} \psi(Y_i, Y_j) \cdot \phi(Y_i) \cdot \prod_{k \in N(i) \setminus j} m_{k \to i}(Y_i) \quad \forall Y_j \in \mathcal{L}
    $$

    This equation is best read from right to left : $i$ collects and
    aggregates the beliefs about him from its neighbours (according to
    each of them what's $P(i_c=Y_i)$), multiplies it with its prior
    belief of belonging to class $Y_i$ and tells $j$ how $i$'s label
    should influence $j$'s label through the label-label matrix. See
    figure [11](#fig:1) for an example.

    <p align="center">
      <img src="../../images/message_passing_schema.PNG" alt="Message passing" width="300"/>
    </p>
    <p align="center"><em>Figure 11: Message passing example where in red are messages from $i$'s neighbours that $i$ has to aggregate to update its belief and in green is the message that $i$ will pass to $j$ which quantifies how much $i$ believes that $j$ is of class $Y_j$



    <!-- | ![Message passing](../../images/message_passing_schema.PNG) |
    |:--:|
    | Figure 11: Message passing example where in red are messages from $i$'s neighbours that $i$ has to aggregate to update its belief and in green is the message that $i$ will pass to $j$ which quantifies how much $i$ believes that $j$ is of class $Y_j$ | -->



3.  After convergence, the class probabilities are given by $b_i(Y_j)$ :
    node $i$'s belief of being of class $Y_j$ where
    $$b_i(Y_j) = \phi(Y_j)*\prod_{k \in N(i)} m_{k \to i}(Y_j)$$

A slight downside to this algorithm is that it tends to amplify wrong
beliefs in small-sized cycled graphs because the same story get spread
again and again. In large-sized graphs this is not a problem, as cycles
are very long, the wrong belief might be spread close to its origin but
will quickly dissipate as it dives deep in the network. Indeed, as the
incorrect belief navigates through the network, the accumulated
corrections and conflicting information tend to outweigh and counteract
the initial wrong belief.

# GNN basics

## GCN

We now finally delve into GNN's, actually we'll first study one of the
simplest and most important GNN : Graph Convolutional Network.\
In CNNs, a kernel operates by applying pixel-wise transformations
(multiplications) and aggregating (summing) information from neighboring
pixels. This process defines new information for the current pixel, but
at another level. The neighbours pixels are not removed or deleted but
instead a new level of information is created to store the results of
the operation : a feature map. A simple example is shown on figure
[12](#fig:12).


<p align="center">
  <img src="../../images/feature_map.jpg" alt="Feature map" width="600"/>
</p>
<p align="center"><em>Figure 12: Feature map computation example as a result of a convolution with a 3x3 kernel on a 5x5 image</em></p>
<p align="center">Image source: [baeldung.com](https://www.baeldung.com/cs/cnn-feature-map)</p>


<!-- | ![Feature map](../../images/feature_map.jpg) |
|:--:|
| Figure 12: Feature map computation example as a result of a convolution with a 3x3 kernel on a 5x5 image |
| Image source: [www.baeldung.com](https://www.baeldung.com/cs/cnn-feature-map) | -->





In GCN's, it pretty similar. First we determine which neighbours we'll
query and second, we transform and aggregate (or propagate depending on
the point of view) their information. As there is no grid structure,
neighbours over which computation are done are defined by a k-hop
neighborhood, this step is called determining the computation graph.
Then the information of all nodes in a k-hop neighborhood (smaller than
k included) is transformed and propagated.\
To make it more clear, let's consider the following example.\
**E.g** : Suppose that we want to learn the embeddings of $A$ in the
following graph [13](#fig:13). We know the embeddings of $A$ should
depend on the features of its neighbors and also on the features of $A$
but let's forget $A$'s features for now.

<p align="center">
  <img src="../../images/gcn_input_graph.PNG" alt="GCN input graph" width="400"/>
</p>
<p align="center"><em>Figure 13: GCN input graph</em></p>



<!-- | ![GCN input graph](../../images/gcn_input_graph.PNG) |
|:--:|
| Figure 13: GCN input graph | -->



The computation graph defined in a 1-hop neighborhood around $A$ is
represented on figure
[14](#fig:14) where the neighbours send their
features (messages) to $A$ which are aggregated and transformed by a
box. What happens in the box is completely arbitrary, its a modelization
choice. You can decide to simply square the mean input signals or you
could pass it through whatever function. To find decent embeddings, one
good idea is to define the boxes as neural networks. As NN's can
approximate a lot of functions, they can also approximate the function
that is the most fit to this task, that is, the function that will
produce an optimal embedding. Obviously we don't know that function.



<p align="center">
  <img src="../../images/gcn_first_embeddings.PNG" alt="GCN first embeddings" width="600"/>
</p>
<p align="center"><em>Figure 14: Computation graph defined in a 1-hop neighborhood around $A$. Neighbor features are aggregated and then transformed by a box</em></p>


<!-- | ![GCN first embeddings](../../images/gcn_first_embeddings.PNG) |
|:--:|
| Figure 14: Computation graph defined in a 1-hop neighborhood around $A$. Neighbor features are aggregated and then transformed by a box | -->


What if we wanted to compute the embedding of $A$ using a 2-hop
neighborhood instead ? It's quite simple, we simply have to first define
the computation graph on figure
[15](#fig:15). We observe that the $2^{nd}$
order embedding of $A$ is directly dependent on the $1^{st}$ order
embedding of its neighbours. That is, to compute layer-2 embeddings, one
must first compute layer-1 embedding. Actually, for a computation graph
defined on a m-hop neighborhood, we'll have m layers of embeddings. That
is, a single node has several embeddings, it has one embedding at each
layer :

-   Layer-0 embedding of node $u$ is its attribute feature vector
    $\mathbf{x_u}$.

-   Layer-k embedding of node $u$ is the result of an aggregation of
    information of all nodes which are less than k hops away from $u$.


<p align="center">
  <img src="../../images/gcn_second_embeddings.PNG" alt="GCN second embeddings" width="600"/>
</p>
<p align="center"><em>Figure 15: Computation graph defined in a 2-hop neighborhood around $A$</em></p>


<!-- | ![GCN second embeddings](../../images/gcn_second_embeddings.PNG) |
|:--:|
| Figure 15: Computation graph defined in a 2-hop neighborhood around $A$ | -->


By now you should have understood that each node defines its own set of
computation graphs, i.e, the 1-hop computation graph of $A$ is not the
same as the 1-hop computation graph of $C$ and neither are their k-hop
computation graphs. If we were to draw the 2-hop computation graph of
every node, we would get that : figure
[16](#fig:16).

<div style="text-align: center;">
  <img src="../../images/gcn_second_embeddings_all.PNG" alt="GCN second embeddings for all" width="900"/>
  <p><em>Figure 16: All 2-hop computation graphs for nodes in [13](#fig:13)</em></p>
</div>

<!-- | ![GCN second embeddings for all](../../images/gcn_second_embeddings_all.PNG) |
|:--:|
| Figure 16: All 2-hop computation graphs for nodes in [13](#fig:13) | -->




Let's now delve into the actual maths. We said previously that to
generate an embedding at the l+1-layer we simply aggregated and
transformed the neighbours embeddings at the l-layer. But how do we
actually do that ?

1.  Aggregate embeddings from neighbours

2.  Define the next-level embedding by applying a NN to transform the
    aggregated embedding

Mathematically it reads :\
Let $\mathbf{h_v^l}$ denote the l-level embedding of node $v$ :

$$
\mathbf{h}_v^0 = \mathbf{x}_v \quad \text{ (1)}
$$



$$\mathbf{h_v^{l+1}} = \sigma \left( \mathbf{W_l} \cdot \left(\sum_{u \in N(v)} \frac{\mathbf{h_u^l}}{|N(v)|}\right) + \mathbf{B_l} \cdot \mathbf{h_v^l}\right) \quad \text{ (2)}$$ 

$$\mathbf{z_v} = \mathbf{h_v^L} \quad \text{ (3)}$$

Where $\mathbf{W_l}, \mathbf{B_l} \in \mathcal{R}^{ d_l \times d_{l+1} }$

Equation [1](#h_0) defines
the 0-layer embedding as the attribute feature vector, equation
[3](#h_L) defines the final
embedding as the L-layer embedding. This is the one we'll use for the
downstream task.  
The most important : equation
[2](#h_l+1) defines the
next-level embedding of $v$ given the current-level embeddings of all of
$v$'s neighbours. It simply linearly transforms the average embedding of
neighbours through $\mathbf{W_l}$. It also linearly transforms the
current-level embedding of $v$ through $\mathbf{B_l}$, adds the two
transformations, and passes the result through an activation function.
Even though figures [14](#fig:14) and
[15](#fig:15) did not capture the current-level
embeddings of the node itself to compute the next-level embeddings, we
see through equation [2](#h_l+1) that it is indeed the case.  

The parameters $d_0, d_1, ..., d_L$ which define the embeddings length
are chosen arbitrarily by the modeler however $d_0$ must be equal to the
number of components in the attribute feature vector $\mathbf{x_v}$ and
$d_L$ refers to the final embedding size. Note that they are
**independent** of the graph dimensions.  

You'll observe that the weight matrices $\mathbf{W_l}$ and
$\mathbf{B_l}$ are not indexed by a node and apply thus to all nodes.
How can that be ? We saw that each computation graph was different and
the NN sometimes received messages from 3 neighbours and sometimes from
5 neighbours. Actually, the NN always receives one single input which is
the aggregated messages of all nodes. Whether there are 3 or 5
neighbours sending l-layer embeddings does not matter as they will all
be aggregated by the aggregator to construct a single embedding of
length $d_l$. That is what will be fed to the NN.\
We note however that $\mathbf{W_l}$ and $\mathbf{B_l}$ are indexed by l
which is the current depth. Indeed, for a given depth l, all the weight
matrices will be the same. That is why on figure
[15](#fig:15), we observe that all boxes at
layer-1 are dark grey, because they all store the same pair
$\mathbf{W_1}$ and $\mathbf{B_1}$ and the box at layer-2 is light grey
because it stores a pair $\mathbf{W_2}$ and $\mathbf{B_2}$. Yeah in this
example there is only one light grey box but trust me, at the $2^{nd}$
level all these light grey boxes would store the same parameters. A
schematic of the shared weights is given on figure
[17](#fig:17).


<div style="text-align: center;">
  <img src="../../images/gcn_shared_parameters.PNG" alt="GCN shared parameters" width="900"/>
  <p><em>Figure 17: GCN neural networks are the same at a given layer</em></p>
</div>

<!-- | ![GCN shared parameters](../../images/gcn_shared_parameters.PNG) |
|:--:|
| Figure 17: GCN neural networks are the same at a given layer | -->



Note that having a $\mathbf{W_l}$ and $\mathbf{B_l}$ per node would not
be in line with one of DL's most important principles : scalability.\
The role of the aggregation function is crucial in GNN, in GCN the
aggregation function is simply an average but we'll see examples where
the aggregation is more complex. No matter how complex it is, it still
has to satisfy a few properties such as the permutation-invariant
property. That is, changing the order of the inputs of the function
should have no impact on the output.\
In case you're wondering how we train the GCN, the method has not
changed : we simply feed the embeddings to a downstream model and
back-propagate the downstream loss to update the GCN's weights.\
**NB : the downstream model has to be differentiable !** If it's not,
the gradients cannot flow back to the GCN and thus weights are not
updated.

In this section, we introduced Graph Convolutional Networks which is
just one type of GNN. Let's use what we learnt to define the main
components of a generic GNN.

## A layer of GNN

A layer of GNN is made of 2 components : the **message** and the
**aggregation**.

### Message

The message function *MSG* receives an aggregated signal, transforms it,
and sends it to its neighbors.

$$MSG^{(l)}: \mathbb{R}^{d_{l-1}} \rightarrow \mathbb{R}^{d_l}$$ 

such that $$\mathbf{m}_u^l = MSG^l(\mathbf{h}_u^{l-1})$$ An example of
message function could be using a linear layer with
$MSG^l = \mathbf{W^l}$ such that
$\mathbf{m}_u^l = \mathbf{W^l}\mathbf{h}_u^{l-1}$

### Aggregation

As the name suggests, the aggregation function simply aggregates an
arbitrary number of messages. Given n messages of length $d_l$ where n
is the number of neighbours, the function returns a single signal of
length $d_l$ and this signal is what defines the embedding of the node
at the current layer l.

$$AGG^{(l)}: \mathbb{R}^{n \times d_l} \rightarrow \mathbb{R}^{d_l}$$

such that

$$\mathbf{h}_v^l = AGG^l (\{ \mathbf{m}_u^l \quad \forall  u \in N(v) \})$$

An example of aggregation function could be the average of signals, the
sum of signals or even the maximum of signals.\
However these simple aggregation functions all suffer from a very simple
problem : they do not consider the information from the node $v$ itself
but only information from its neighbours. A simple solution would be to
pass the embedding of the node itself or a transformed embedding, i.e, a
$\textbf{message}$. We thus define a message for the node itself
$\mathbf{m}_v^l = \mathbf{B^l}\mathbf{h}_v^{l-1}$ and then need to
aggregate the aggregated message with this message, so we need 2
aggregations.   
This means,
$$\mathbf{h}_v^l = AGG^1( AGG^2(\{ \mathbf{m}_u^l \quad \forall  u \in N(v) \}), \mathbf{m}_v^l) )$$
In GCN for example, you can identify on equation
[2](#h_l+1) that $AGG^1$
= Sum() and $AGG^2$ = Average() which is then passed to an activation
function.  

Now that we have seen the 2 main components of a layer of GNN, we'll
briefly review some of the most popular GNN layers. As a good exercise,
you can try to identify the message and aggregation functions in all the
layers we'll cover.

### Classical GNN layers

-   GCN :
    $$\mathbf{h_v^{l+1}} = \sigma \left( \mathbf{W_l} \cdot \left(\sum_{u \in N(v)} \frac{\mathbf{h_u^l}}{|N(v)|}\right) + \mathbf{B_l} \cdot \mathbf{h_v^l}\right)$$

-   GraphSAGE :
    $$
    \mathbf{h}_v^{l+1} = \sigma \left( \mathbf{W}_l \cdot \text{CONCAT}(\mathbf{h}_v^l, \text{AGG}(\mathbf{h}_u^l \ \forall u \in N(v))) \right)
    $$

    Where AGG() can be a simple sum but even a
    deep MLP, it just has to be good at aggregating.

-   Graph Attention Network :
    $$\mathbf{h_v^{l+1}} = \sigma \left( \sum_{u \in N(v)} \alpha_{vu} \mathbf{W^l} \cdot \mathbf{h_v^{l}}
                \right)$$ 
    Where $\alpha_{vu}$ is an attention factor
    that allows to give more importance to some neighbours and less to
    others.

Now that we have seen the main components of GNN layers and typical
examples of layers, what can we do ? Well, we can follow the deep
learning recipe and simply design any deep GNN that we want now in the
same way we learnt to design MLP's after we learnt about the perceptron.

# Deep GNNs

The simplest way to stack layers is to stack them sequentially as shown
on figure [18](#fig:18)

<p align="center">
  <img src="../../images/stacking_gnn_layers.PNG" alt="Stacking GNN layers" width="200"/>
</p>
<p align="center"><em>Figure 18: Stacking GNN layers to create a deep GNN</em></p>


<!-- | ![Stacking GNN layers](../../images/stacking_gnn_layers.PNG) |
|:--:|
| Figure 18: Stacking GNN layers to create a deep GNN | -->


It is the traditional approach used in deep learning however we should
already note something :

-   In MLP's and CNN's : depth means complexity and expressivity.

-   In GNN's : depth means number of hops, it simply defines the
    neighborhood a node will use to compute its embeddings.

The issue with this approach is that deep sequential GNN's suffer from the
over-smoothing problem where all node embeddings converge to the same
values. But why ?  
For k a large number of layers , the computation graph of all nodes
always consists of all nodes in the graph but arranged a bit
differently. For such k, the receptive field of a single node is
actually huge and thus nodes will all have a very similar receptive
field, a field that covers most nodes in the network. How can that be ?\
Well it is not very surprising, do you know about the 6 people theory?
According to this theory, any person on earth can be connected to any
other person on earth by at most 6 known intermediaries. From the graph
perspective, we may understand that any pair of nodes in the graph can
be connected by 6 intermediate neighbours on average.\
That is, when using 6 GNN layers, the computation graph of any node
almost always contains all nodes in the graph and therefore all nodes
have the same computation graph.  
If 2 computation graphs are identical,
their embeddings will be identical. An example of the problem is given
on figure[19](#fig:19).

<p align="center">
  <img src="../../images/over_smoothing.PNG" alt="Over-smoothing" width="600"/>
</p>
<p align="center"><em>Figure 19: A GNN consisting of 3 GNN layers will have the computation graphs of all nodes that almost entirely overlap</em></p>


<!-- | ![Over-smoothing](../../images/over_smoothing.PNG) |
|:--:|
| Figure 19: A GNN consisting of 3 GNN layers will have the computation graphs of all nodes that almost entirely overlap | -->


How do we solve that problem ? How do we make a GNN expressive if we are
only allowed to use a few GNN layers ?\
First, we can try making the GNN layers more expressive themselves !\
In equation [2](#h_l+1)
we simply use a 1-layer MLP, what if we used a 2-layer MLP as in
equation
[4](#more_expressive_layer) or even a 10-layer MLP? A schematic
of the solution is given on figure
[20](#fig:20).

$$\mathbf{h_v^{l+1}} = \sigma \left( \mathbf{W_{2,l}} \cdot \sigma \left( \mathbf{W_{1,l}} \cdot \left(\sum_{u \in N(v)} \frac{\mathbf{h_u^l}}{|N(v)|}\right) + \mathbf{B_{1,l}} \cdot \mathbf{h_v^l}\right) + \mathbf{B_{2,l}}\right) \quad \text{ (4)}$$


<div style="text-align: center;">
  <img src="../../images/over_smoothing_sol_1.PNG" alt="Over-smoothing sol 1" width="600"/>
  <p><em>Figure 20: First solution to the over-smoothing problem by making the transform layers more expressive through the use of an n-layer MLP</em></p>
</div>

<!-- | ![Over-smoothing sol 1](../../images/over_smoothing_sol_1.PNG) |
|:--:|
| Figure 20: First solution to the over-smoothing problem by making the transform layers more expressive through the use of an n-layer MLP | -->




Second, another solution to make the model more expressive simply
consists in adding layers that do **not** pass messages. These layers
could be as simple as MLP layers that would be applied to each node,
they simply transform either the features if they are applied at the
beginning or the embeddings if they are applied in the middle or at the
end of the GNN. A simple schema can be seen on figure
[21](#fig:21).

<p align="center">
  <img src="../../images/over_smoothing_sol_2.PNG" alt="Over-smoothing sol 2" width="300"/>
</p>
<p align="center"><em>Figure 21: Second solution to the over-smoothing problem by adding MLP layers applied to each node</em></p>


<!-- | ![Over-smoothing sol 2](../../images/over_smoothing_sol_2.PNG) |
|:--:|
| Figure 21: Second solution to the over-smoothing problem by adding MLP layers applied to each node | -->


Finally, a third solution consists in adding skip-connections to the
GNN. The origin of the over-smoothing problem lies in the fact that at
large depth, the receptive fields of all the nodes are the same but at a
shallow depth, **it is not the case**!. We cannot directly use the
shallow embeddings $\mathbf{h_v^l}$ with $l \leq 3$ because they are not
expressive enough but one thing we can do is pass them to the deep
embeddings $\mathbf{h_v^l}$ with $l \geq 6$ by a skip connection. What's
great is that even if the deep embeddings are all the same, adding the
shallow embeddings which are all different will make the resulting
embeddings all different. A schematic of this solution is given on
figure [22-left](#fig:22).  
Another solution based on the
same idea is to recurrently add skip connections at all GNN layers
successively as depicted on figure
[22-right](#fig:23).


<div style="text-align: center; display: flex; flex-direction: row; justify-content: center;">
  <img src="../../images/over_smoothing_sol_3_1.PNG" alt="Skip-connection 1" width="300"/>
  <img src="../../images/over_smoothing_sol_3_2.PNG" alt="Skip-connection 2" width="300"/>
</div>
<p style="text-align: center;"><em>Figure 22: Skip-connection from shallow embeddings to deep embeddings (left) and Skip-connection from embeddings at layer $l$ to embeddings at layer $l+1$ successively (right)</em></p>


<!-- <div align="center">

| ![Between Centrality](../../images/over_smoothing_sol_3_1.PNG) | ![Closeness Centrality](../../images/over_smoothing_sol_3_2.PNG) |
| --- | --- |
| *Figure 22: Skip-connection from shallow embeddings to deep embeddings (left) and Skip-connection from embeddings at layer $l$ to embeddings at layer $l+1$ successively (right)* | |

</div> -->




We have now covered the main material for Graph Neural Networks
architecture and hopefully you should now be able to pseudo-code one of
them for a simple task. That's great but to solve a ML problem you need
2 main components : a model and a dataset. In many cases, poor model
performance is not necessarily due to an inadequate architecture but
often stems from either a low-quality dataset or insufficient data. In
the following section, we will introduce several techniques to augment
your training data, allowing you to train your GNNs better.


$\textbf{Notebook 4 :}$ A classifier based on GCN embeddings [GCN classifier](https://colab.research.google.com/drive/1G1Ox6-a_CaK30dLHWnKwxPOJJkXXHQAS?usp=sharing)

$\textbf{Notebook 5 :}$ A classifier based on GraphSAGE embeddings [GraphSage classifier](https://colab.research.google.com/drive/1RwCB99_pyq7nBrm74Bz8t9-ZcHBhVaDR?usp=sharing)

# Graph augmentation

You've probably heard of data augmentation when you learnt about
training CNNs, we usually enlarge the support of the data distribution
by transforming some data samples which makes the model less prone to
over-fitting. These transformations include random cropping, random
flipping, random rotation, color jittering etc\...\
Graph augmentation is based on the same idea except in graphs we have 2
types of features : attribute-features and graph-features sometimes
called structural features. We thus present augmentation techniques for
these 2 types of features separately.\
Thinking about it, augmenting the graph seems like a good idea because
it is very unlikely that the input graphs happens to be the optimal
graph to learn embeddings from for our task.

### Attribute-features augmentation

If the input graph lacks node features, it is a good idea to simply
design some. It can be as simple as designing a one-hot encoding vector
for each of them to computing more advanced statistics such as the node
eigen centrality or the 3-graphlet score as we saw in section
[4]. What's great is that any
features you could think of can potentially be used however you should
remind that you'll have to compute them while pre-processing test
samples in the test/inference phase and that can slow down inference a
lot.

### Structural features augmentation

There are several cases where you may want to structurally augment your
graph :

-   If the graph is too sparse, message passing becomes inefficient
    because nodes have very few neighbours and it is hard to propagate
    information.\
    A solution could be to connect 2-hop neighbours by virtual edges,
    i.e, replace $A \leftarrow A + A^2$.\
    Another solution is to add a virtual node that is connected to all
    other nodes such that the distance between any pair of nodes is
    upper bounded by 2. This allows a more fluid message passing.

-   If the graph is too dense, message passing will be computationally
    intensive and it is not always worth it to add nodes in the
    computation graph that add very little information to the
    computation graph because one of their very similar neighbour is
    already present.\
    A solution consists in sampling neighbours to create the computation
    graph. Instead of adding all neighbours of the current node, we
    uniformly (or not) sample half of them and we recursively do the
    same until we reached the desired depth.

Ok great we've seen how GNNs work and even how we can augment their
training data but how do we actually train them ?

# Training GNNs

As I've said repeatedly, the encoder is trained by feeding the
embeddings to a downstream model and back-propagating the downstream
loss all up to the encoder's weights. Still, we may want to look at what
kind of downstream model we could use for different prediction tasks.
These downstream models are often called prediction heads.\
NB : Remember that the downstream has to be differentiable for the loss to
properly back-propagate its way up to the encoder.

## Node-level prediction

Given the d-dimensional embedding $\mathbf{h^L_v} \in \mathcal{R}^d$, we
can use it for a k-class prediction or k-target regression using a
linear prediction head. Classifying a node based on its embedding simply
consists in computing
$$\mathbf{\hat{y}_v} = PredictionHead(\mathbf{h^L_v}) = \mathbf{W}_H \cdot \mathbf{h^L_v}$$
where $\mathbf{W}_H \in \mathcal{R}^{k \times d}$ is a trainable matrix
and $\mathbf{\hat{y}_v} \in \mathcal{R}^k$ is the vector of interest
which can be passed through a softmax to become a proper
distribution.

## Edge-level prediction

Given $\mathbf{h}^L_v$ and $\mathbf{h}^L_u$, we want to predict the values of an edge between the nodes $v$ and $u$ out of $k$ possible outcomes. If $k=2$, it can be as simple as "Exists" vs "Not exists", but in a social network, we could have $k=4$ possible outcomes: "Is friend with", "Has talked to", "Has blocked", "Has ignored". We can once again simply use a linear prediction, except it will have to accept the concatenated embeddings or at least an aggregation of them.

$$
\mathbf{\hat{y}}_{uv} = \text{PredictionHead}(\mathbf{h}^L_v, \mathbf{h}^L_u) = \mathbf{W}_H \cdot \text{CONCAT}(\mathbf{h}^L_v, \mathbf{h}^L_u)
$$
where $\mathbf{W}_H \in \mathcal{R}^{k \times 2d}$ is a trainable matrix and $\mathbf{\hat{y}}_uv \in \mathcal{R}^k$ is the vector of interest
which can be passed through a softmax to become a proper
distribution.


## Graph-level prediction

Given the d-dimensional embeddings
$\{ \mathbf{h^L_v} \in \mathcal{R}^d \quad \forall v \in V \}$, we need
to aggregate them before passing them to a prediction head. Simply
concatenating them as we did before is not a scalable option as the
prediction head matrix will be huge for large graphs. A good solution
consists in first aggregating all the embeddings and then feeding the
aggregated embedding to a prediction head.
$$\mathbf{\hat{y}_G} = PredictionHead(\{ \mathbf{h^L_v} \in \mathcal{R}^d \quad \forall v \in V \}) = \mathbf{W}_H \cdot AGG(\{ \mathbf{h^L_v} \in \mathcal{R}^d \quad \forall v \in V \})$$

## Loss function

When we train a ML model, it is necessary to define a loss. You've
probably heard that dozens of times and have been doing it hundreds of
times but it's technically not correct. We do not define a loss, we
derive it given the model we assumed and the optimization strategy. Most
often we look for the optimal parameters by maximizing the likelihood of
observing the dataset given the model. By playing with the maximum
likelihood equations and shaking the terms, we can derive the loss
function. Anyway, it was a comment that bore only slight relevance.  

When training GNNs, the loss used is the one used by the downstream
model, it is nothing special. If your downstream task is about
classification, you'll use the cross-entropy loss, if it's about
regression, you'll use MSE. You see, nothing special.\
Once you've trained your model, the next step consists in evaluating it.
We could use the loss function computed on the test set as an evaluation
metric but this is hardly enough. Suppose the average test cross-entropy
loss is of 0.6 in a classification task, it does not say much about the
model itself even though it's a decent metric to compare models. A more
interpretable metric would be the accuracy, the precision, the recall
and the ROC curve plot. For regression tasks, the evaluation metrics to
use would be the MSE, MAE, etc\...

## Splitting the dataset

We've mentionned above the existence of a test set. It is a good
practice in ML to split your dataset in a train-val-test split to have
an honest idea of a model's performance. In a classical dataset such as
ImageNet, it is easy to split your dataset into a train-val-test as each
sample has been sampled **independently** from the data distribution.
Sample n°23 and n°24 have no direct dependency and therefore we can
randomly define 3 subsets as the train set, validation set and test set.  
In other settings such as time-series it is more complex as the temporal
aspect introduces dependencies between consecutive samples. In
time-series datasets, the order of data points matters, and sampling
must be done with careful consideration of these dependencies. Unlike
independent and identically distributed (i.i.d.) datasets like ImageNet,
where each sample is unrelated, time-series data often exhibits temporal
patterns and trends. In such cases, a random split is meaningless.\
Similarly, it is tough to split graph datasets since randomly selecting
a set of nodes would likely lead to overly sparse graphs or even
unconnected graphs. The structure of graph data introduces
inter-dependencies among nodes, and a random split might disrupt these
relationships, negatively impacting the model's ability to generalize to
unseen data. Therefore, in graph datasets, specialized techniques such
as graph-based sampling or link-based splitting are often employed to
ensure a representative and meaningful distribution across the train,
validation, and test sets. See [lecture 5](https://web.stanford.edu/class/cs224w/slides/05-GNN3.pdf) for details.

# The best GNN

We've seen a plethora of GNNs, and it is natural to wonder which one of
them works best, or at least, which one is most expressive. These two
criteria, effectiveness and expressiveness, differ in the sense that
effectiveness gauges the model's performance on specific tasks, while
expressiveness reflects the model's capacity to capture and learn
complex relationships and patterns in data.  
Expressiveness might me measured by the ability of a model to
distinguish different graph structures. For example, given 2 different
graph structures, can your model learn different embeddings for them if
they both have the same attribute feature vectors ?\
Consider the following simple graph on figure [23](#fig:23) where all nodes share the same
attribute feature vector, that is why they all share the same color.

<p align="center">
  <img src="../../images/GIN_input_graph.PNG" alt="GIN input graph" width="400"/>
</p>
<p align="center"><em>Figure 23: Simple input graph</em></p>


<!-- | ![GIN input graph](../../images/GIN_input_graph.PNG) |
|:--:|
| Figure 23: Simple input graph | -->


Node $2$ and node $4$ seem similar as they both have 2 neighbours
however node $2$ has 1 neighbour of degree 2 and 1 neighbour of degree
3. Node $4$'s situation is different, it has 1 neighbour of degree 1 and
1 neighbour of degree 3. If we were to draw the computational graphs of
node $2$ and node $4$, we would see that they are different at the
second layer.\
On the other hand, node $1$ and node $2$ have the **exactly same**
computation graph because they have the same neighbourhood structure,
see figure [24](#fig:24).


<p align="center">
  <img src="../../images/GIN_2_comp_graphs.PNG" alt="GIN 2 comp graph" width="400"/>
</p>
<p align="center"><em>Figure 24: Computational graphs of node $1$ and node $2$ which are identical omitting the id's. Note that a GNN does not care about id's unless one manually encodes them as attribute features in the pre-processing stage</em></p>


<!-- | ![GIN 2 comp graphs](../../images/GIN_2_comp_graphs.PNG) |
|:--:|
| Figure 24: Computational graphs of node $1$ and node $2$ which are identical omitting the id's. Note that a GNN does not care about id's unless one manually encodes them as attribute features in the pre-processing stage | -->


If 2 computational graphs are the same, the embeddings will be the same
and our GNN won't be able to distinguish them (sad result :( ).\
However, some GNN's are not even able to distinguish different
computation graphs, that is, even though 2 nodes have different
neighbourhood structures, their embeddings will be the same. This is
very undesired! We want different computation graphs to be mapped to
different embeddings by the GNN as depicted on figure
[25](#fig:25). This property is called injectivity.\
**Definition :** A function f : X $\mapsto$ Y is injective if it maps
different elements of X to different elements of Y. That is,
$\neg \exists x_1 \in X, x_2 \in X$ distinct such that $f(x_1)=f(x_2)$.



<p align="center">
  <img src="../../images/GIN_embeddings.PNG" alt="GIN embeddings" width="600"/>
</p>
<p align="center"><em>Figure 25: Scalar embeddings represented by colors of different nodes of the input graph</em></p>


<!-- | ![GIN embeddings](../../images/GIN_embeddings.PNG) |
|:--:|
| Figure 25: Scalar embeddings represented by colors of different nodes of the input graph | -->


We actually want GNNs to be injective!
This is kind of a high-level requirement because GNNs are made of
several layers and in each layer several operations take place. We'd
like to know which fundamental operations in GNNs should be injective
such that the whole GNN is injective.

## Injectivity Importance

**Key Observation:** Sub-trees of the same depth can be differentiated
from one another in a bottom-up fashion.

Consider Figure
[26](#fig:26), where $\mathbf{h}_1^2$,
the $2^{nd}$ level embedding of node $1$, is an aggregation from the
$1^{st}$ level embeddings of nodes $2$ and $5$. It is crucial to note
that $\mathbf{h}_4^2$ will also be aggregated from its 2 neighbors'
$1^{st}$ level embeddings. If these embeddings were all the same, i.e.,
$\mathbf{h}_2^1=\mathbf{h}_5^1=\mathbf{h}_3^1=\mathbf{h}_5^1$, it would
not be possible to aggregate them differently.

However, these embeddings should not be equal! On the left tree, nodes
$2$ and $5$ have **5** neighbors to aggregate from, while on the right
tree, nodes $3$ and $5$ have **4** neighbors to aggregate from. If the
aggregation function led to
$\mathbf{h}_2^1 \neq \mathbf{h}_5^1 \neq \mathbf{h}_3^1$, then the
$1^{st}$ level embeddings would have been different, and recursively,
the aggregation function would have led to different $2^{nd}$ level
embeddings.

The requirement becomes clearer when looking at the tree bottom-up.
Since the 0-level embeddings are different (5 vs. 4), and both 2-level
embeddings share 2 neighbors, the only way to allow the 2-level
embeddings to be different is by making this information percolate up,
bubble up. However, when computing the 2-level embeddings, the latter do
not have access to the information that the 0-level embeddings are
different because they went through the 1-level embedding. The GNN will
only be able to produce different 2-level embeddings if the 1-level
embeddings of both trees are different! To achieve that, the mapping
from 0-level to 1-level embeddings has to be **injective**!

Actually, only the aggregation step in the mapping has to be injective.\
**If the aggregation function is injective, we can guarantee that the
GNN will be injective at all levels**. Particularly, if you have 2 trees
with different 0-level embeddings, their last-level embeddings will also
be different because each aggregation step will lead to different
$l+1$-level embeddings.\
To go even further, the complexity and expressivity of GNNs usually boil
down to the complexity and expressivity of the aggregation function.


<p align="center">
  <img src="../../images/from_leaf_node_to_root_node.PNG" alt="GIN embeddings" width="600"/>
</p>
<p align="center"><em>Figure 26: Differentiating embeddings by observing the computation graph bottom-up</em></p>


<!-- | ![From leaf node to root node](../../images/from_leaf_node_to_root_node.PNG) |
|:--:|
| Figure 26: Differentiating embeddings by observing the computation graph bottom-up | -->


## Designing the most expressive GNN

Before designing the most expressive GNN into that, we have to realize
that a neighbor aggregation function can be abstracted as a function
over a multi-set which is nothing but a set which can have repeating
elements. In a traditional set, this is not possible, i.e,
$\{1,2,3\} = \{1,2,3,3\}$.\
In GCN, the aggregation function was simply an average of the received
messages. The average function is **not injective**. Simply consider
$AVG(\{3,5\})=AVG(\{3,4,5\})$.\
In GraphSAGE, the aggregation function is a maximum of the received
messages. The max function is **not injective**. Simply consider
$MAX(\{1,2,3\})=MAX(\{1,2,2,3\})$.\
Therefore, GCN and GraphSAGE are not the most expressive GNNs, but who
is?\
Well at least it has to be a GNN whose aggregation function is injective
but how can we design that ?\
Fortunately there is a great theorem that will help us here : the lemma
5 of Xu et al. ICLR 2019  :

Any injective multi-set function $g(X)$ can be decomposed as

$$g(X) = \phi \left( \sum_{x \in X} f(x) \right)$$

Where $\phi$ and $f$ are both non-linear functions and $X$ is a multi-set.\
As you could guess, $\phi$ and $f$ are very specific functions which we
thus decide to approximate by a 1-layer MLP with a latent dimension of
sufficient size such that
$$g(X) = MLP_{\phi} \left( \sum_{x \in X} MLP_f(x) \right)$$

A GNN that would use such an aggregation function is called a Graph
Isomorphism Network (GIN) and it is the most expressive GNN of
message-passing GNNs we've seen so far.

# Roads to Explore

We've explored the fundamental aspects of Graph Machine Learning (GML), and I encourage you to delve deeper into this subject. 

GML is far from being a solved field; on the contrary, it faces numerous limitations that hinder its application in various contexts. These constraints slow down its widespread adoption and effectiveness. Persistent challenges are still unsolved. Perhaps, you could be one of those exploring these roads.

### Key roads:

1. **Scalability:**
   Handling large-scale graphs efficiently is a significant limitation today as real-world graphs become increasingly massive. What do you do if your graph doesn't fit on a GPU? Do you simply pick a few nodes here and there?

2. **Heterogeneity:**
   Most existing Graph ML methods focus on homogeneous graphs, where all nodes and edges are of the same type. Handling heterogeneous graphs with diverse node and edge types is a growing area of interest.

3. **Dynamical Graphs:**
   Many real-world graphs are dynamic, evolving over time. Adapting existing methods to handle temporal aspects and changes in the graph structure is a tough challenge.

4. **Interpretability:**
   Interpreting graph-based model decisions is challenging due to the complex, non-linear relationships within graphs. The complexity of GNNs makes it difficult to provide clear and intuitive explanations for model predictions.

5. **Transfer Learning:**
   Transfer learning in GML is not a small feat due to the diversity of graph structures, including heterogeneous nodes and edges. Adapting pre-trained models to handle varying graph sizes, dynamic topologies, and different node and edge semantics is definitely not trivial.

6. **Uncertainty Estimation:**
   How can uncertainty estimation techniques be integrated into graph-based models to enhance reliability in predictions, particularly in scenarios involving noisy or incomplete data?


# Implementations
Here are links to all colab notebooks that we've given through-out this blog :
#### I) First model : Classifier on graph-indepent features

[MLP classifier on graph-independent features](https://colab.research.google.com/drive/1Dr1LqLq00RsmGwn5byqU7-_FNy6jBuIn?usp=sharing)


#### II) Second model : Classifier on graph-independent features and  graph-dependent features

[MLP classifier on graph-dependent and graph-independent features](https://colab.research.google.com/drive/1yRzcHPtHrlpeH000xilYn_5hlo0xVcgM?usp=sharing)

#### III) Third model : Classifier on embeddings defined transformed features

[MLP classifier on MLP embeddings](https://colab.research.google.com/drive/1He6LH7bAVbWfnPyIUZgPsbA5DKN831nt?usp=sharing)

#### IV) Fourth model : Classifier on GCN embeddings


[GCN classifier](https://colab.research.google.com/drive/1G1Ox6-a_CaK30dLHWnKwxPOJJkXXHQAS?usp=sharing)

#### V) Fifth model : Classifier on GraphSAGE embeddings

[GraphSage classifier](https://colab.research.google.com/drive/1RwCB99_pyq7nBrm74Bz8t9-ZcHBhVaDR?usp=sharing)

#### VI) Experiments on GNNs vs MLPs

[Experiments](https://colab.research.google.com/drive/1o-K1HIv2XHRzHhl7YXdH4ydndWxRec2n?usp=sharing)
