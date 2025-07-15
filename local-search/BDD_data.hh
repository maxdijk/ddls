#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>

class Node {
public:
    int id;
    int layer;
    std::shared_ptr<Node> leftChild;
    std::shared_ptr<Node> rightChild;

    Node(int nodeId) :
        leftChild(nullptr),
        rightChild(nullptr),
        id(nodeId), 
        layer(-1){
    }
};

class BDD_Input {
    friend std::ostream& operator<<(std::ostream& os, const BDD_Input& bs);
public:
    std::vector<std::shared_ptr<Node>> nodes;
    std::map<int, std::vector<int>> nodes_per_layer;
    std::shared_ptr<Node> root;
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> test_data;
    std::vector<std::vector<std::vector<float>>> data_sorted_per_feature;

    int seed;
    int diagramSize;
    int numberOfFeatures;
    int numberOfClasses;
    float baselineMultiplier;
    float alpha;

    bool greedyInitialization = false;
    std::vector<int> greedyFeatures;
    std::vector<float> greedyThresholds;

    BDD_Input() : nodes{}, root { nullptr } {}

    void setRoot(std::shared_ptr<Node>& rootNode);
    void addNode(std::shared_ptr<Node>& node);
    void loadDiagram(std::string fileName);
    void loadData(std::string filename_train, std::string filename_test);
};

class BDD_Output {
    friend std::ostream& operator<<(std::ostream& os, const BDD_Output& out);
    friend std::istream& operator>>(std::istream& is, BDD_Output& out);
    friend bool operator==(const BDD_Output& out1, const BDD_Output& out2);
public:
    BDD_Output(const BDD_Input& in) : inputBDD(in) {}
    BDD_Output& operator=(const BDD_Output& out);

    const BDD_Input& inputBDD;

    // Stores the values for the current solution
    std::vector<float> nodeThresholds;         
    std::vector<int> nodeFeatures;
    std::vector<bool> isFlipped;
    std::vector<int> leftChild;
    std::vector<int> rightChild;
 
	// Structure, that stores the classification counts for each node, meaning it stores the number of samples
	// of a given class for the left and right child.
    struct ClassificationCounts {
        std::vector<int> left_class_counts;
        std::vector<int> right_class_counts;
    };

    std::vector<int>& classifySample(std::shared_ptr<Node> node, const std::vector<float>& sample, const std::vector<int>& selectedFeatures, const std::vector<float>& selectedThresholds, std::map<int, ClassificationCounts>& classificationCounts) const;
    std::vector<int>& classifySampleUpdatedNode(std::shared_ptr<Node> node, const std::vector<float>& sample, int optimizing_id, int optimizing_feature, float optimizing_threshold, std::map<int, ClassificationCounts>& classificationCounts, bool& relevant_sample) const;
    
    float calculateAccuracy(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const;
    float calculateTestAccuracy(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const;
    int calculateMissclassified(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const;

    std::map<int, ClassificationCounts> BDD_Output::computeClassificationCounts(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data, const std::vector<int>& selectedFeatures, const std::vector<float>& selectedThresholds) const;
    std::map<int, BDD_Output::ClassificationCounts> computeClassificationCountsUpdatedNode(std::shared_ptr<Node> root, int optimizing_id, int optimizing_feature, float optimizing_threshold, std::vector<int>& relevant_samples, const std::vector<std::vector<float>>& data) const;

    float findOptimalFeatureThreshold(int node_id, int feature) const;

    mutable bool up_to_date = false;            // Is current state up to date with current_cost
    mutable float current_cost = -1;

	mutable bool prev_move_optimal = false;     // Was the previous move optimal?
    mutable float prev_optimal_cost = -1;

    void printDiagram() const;
    void outputDiagram() const;
    int getDepth(std::shared_ptr<Node> current_node) const;
    float getQuestionLength() const;
    float getFragmentation(int size) const;
};


class BDD_Move
{
    friend bool operator==(const BDD_Move& m1, const BDD_Move& m2);
    friend bool operator!=(const BDD_Move& m1, const BDD_Move& m2);
    friend bool operator<(const BDD_Move& m1, const BDD_Move& m2);
    friend std::ostream& operator<<(std::ostream& os, const BDD_Move& m);
    friend std::istream& operator>>(std::istream& is, BDD_Move& m);
public:
    BDD_Move(int n_id = -1, int prevF = -1, int newF = -1, float prevT = -1.0, float newT = -1.0) : 
        nodeID(n_id),
        prevFeature(prevF),
        newFeature(newF),
        prevThreshold(prevT),
        newThreshold(newT) {}

    int nodeID;
    int prevFeature;
    int newFeature;
    float prevThreshold;
    float newThreshold;
};


// No longer used, flips a <= to > or vice versa
class BDD_FlipMove
{
    friend bool operator==(const BDD_FlipMove& m1, const BDD_FlipMove& m2);
    friend bool operator!=(const BDD_FlipMove& m1, const BDD_FlipMove& m2);
    friend bool operator<(const BDD_FlipMove& m1, const BDD_FlipMove& m2);
    friend std::ostream& operator<<(std::ostream& os, const BDD_FlipMove& m);
    friend std::istream& operator>>(std::istream& is, BDD_FlipMove& m);
public:
    BDD_FlipMove(int n_id = -1, bool prevV = -1, bool newV= -1 ) :
        nodeID(n_id),
        prevValue(prevV),
        newValue(newV) {}

    int nodeID;
    bool prevValue;
    bool newValue;
};

// Move that changes the child of a node, used for edge moves
class BDD_EdgeMove
{
    friend bool operator==(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2);
    friend bool operator!=(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2);
    friend bool operator<(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2);
    friend std::ostream& operator<<(std::ostream& os, const BDD_EdgeMove& m);
    friend std::istream& operator>>(std::istream& is, BDD_EdgeMove& m);
public:
    BDD_EdgeMove(int n_id = -1, bool l_edge = -1, bool prev_child = -1, bool new_child = -1) :
        nodeID(n_id),
		left_edge(l_edge),
		prev_child_id(prev_child),
        new_child_id(new_child) {
    }

    int nodeID;
    bool left_edge;
    int prev_child_id;
	int new_child_id;
};
