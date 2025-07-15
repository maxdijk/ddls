Code for Master's Thesis Learning Decision Diagrams for Classification via Local Search.

I'll add more detailed instructions later.

local_search_final.py handles the Bayesian search. It initializes the diagram, saves it to a file and then calls bdd.exe using the correct commands.

The printed results are in the following format:
solver,filename,metaheuristic,objective_cost,initialization_method,diagram_size,features,classes,samples,train_acc,test_acc,runtime,last_improvement,depth,question_length,avg_frag
