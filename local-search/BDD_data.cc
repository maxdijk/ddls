#include "BDD_data.hh"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <unordered_set>
#include <random>
#include <unordered_map>



std::ostream& operator<<(std::ostream& os, const BDD_Input& in) {
    return os;
}

std::ostream& operator<< (std::ostream& os, const BDD_Output& out) {
    return os;
}

std::istream& operator>>(std::istream& is, BDD_Output& out) {
    return is;
}

bool operator==(const BDD_Output& out1, const BDD_Output& out2) {
    return out1.nodeThresholds == out2.nodeThresholds &&
        out1.nodeFeatures == out2.nodeFeatures &&
        out1.isFlipped == out2.isFlipped &&
        out1.leftChild == out2.leftChild &&
        out1.rightChild == out2.rightChild;
}


BDD_Output& BDD_Output::operator=(const BDD_Output& out) {
	nodeThresholds = out.nodeThresholds;
	nodeFeatures = out.nodeFeatures;
	isFlipped = out.isFlipped;
    leftChild = out.leftChild;
    rightChild = out.rightChild;
    return *this;
}

void BDD_Output::printDiagram() const {
    for (auto& node : inputBDD.nodes) {
        std::cout << "\nNodeID: " << node->id;
        std::cout << "\nFeature: " << nodeFeatures[node->id] << ", Value: " << nodeThresholds[node->id];
        if (node->leftChild && node->rightChild) {
            std::cout << "\nChildren: " << leftChild[node->id] << ", " << rightChild[node->id];
        }
    }
}

// Returns the current class counts using classificationCounts for the final edge reached by current sample
std::vector<int>& BDD_Output::classifySample(std::shared_ptr<Node> node, const std::vector<float>& sample, const std::vector<int>& selectedFeatures, const std::vector<float>& selectedThresholds, std::map<int, ClassificationCounts>& classificationCounts) const {
    while (node != nullptr) {
        bool traverse_left = (!isFlipped[node->id] && sample[selectedFeatures[node->id]] <= selectedThresholds[node->id])
            || (isFlipped[node->id] && sample[selectedFeatures[node->id]] > selectedThresholds[node->id]);

        if(node->leftChild == nullptr && node->rightChild == nullptr) {
			if (traverse_left) {
				return classificationCounts[node->id].left_class_counts;
			}
			else {
				return classificationCounts[node->id].right_class_counts;
			}
        }
        else {
            if (traverse_left) {
                node = inputBDD.nodes[leftChild[node->id]];
            }
            else {
                node = inputBDD.nodes[rightChild[node->id]];
            }
        }
		
    }
}

// Returns the current class counts using classificationCounts for the final edge reached by current sample, with an updated node
std::vector<int>& BDD_Output::classifySampleUpdatedNode(std::shared_ptr<Node> node, const std::vector<float>& sample, int optimizing_id, int optimizing_feature, float optimizing_threshold, std::map<int, ClassificationCounts>& classificationCounts, bool& relevant_sample) const {
    while (node != nullptr) {
        int cur_feature = nodeFeatures[node->id];
        float cur_threshold = nodeThresholds[node->id];

        if (node->id == optimizing_id) {
            relevant_sample = true;
            cur_feature = optimizing_feature;
            cur_threshold = optimizing_threshold;
        }

        bool traverse_left = (!isFlipped[node->id] && sample[cur_feature] <= cur_threshold)
            || (isFlipped[node->id] && sample[cur_feature] > cur_threshold);

        if (node->leftChild == nullptr && node->rightChild == nullptr) {
            if (traverse_left) {
                return classificationCounts[node->id].left_class_counts;
            }
            else {
                return classificationCounts[node->id].right_class_counts;
            }
        }
        else {
            if (traverse_left) {
                node = inputBDD.nodes[leftChild[node->id]];
            }
            else {
                node = inputBDD.nodes[rightChild[node->id]];

            }
        }

    }
}

// Using classification counts, which holds the counts of each class for the left and right edges of each node, compute the accuracy
float computeAccuracyFromCounts(const std::map<int, BDD_Output::ClassificationCounts>& classificationCounts, int totalSamples) {
    int correctPredictions = 0;

    for (const auto& entry : classificationCounts) {
        const BDD_Output::ClassificationCounts& cc = entry.second;

        int left_majority_class_count = 0;
        for (int count : cc.left_class_counts) {
            left_majority_class_count = std::max(left_majority_class_count, count);
        }

        // Majority class for the right edge
        int right_majority_class_count = 0;
        for (int count : cc.right_class_counts) {
            right_majority_class_count = std::max(right_majority_class_count, count);
        }

        // Add majority counts to correct predictions
        correctPredictions += left_majority_class_count;
        correctPredictions += right_majority_class_count;
    }

    // Return total accuracy
    return (totalSamples > 0) ? static_cast<float>(correctPredictions) / totalSamples : 0.0f;
}

float computeMisclassificationsFromCounts(const std::map<int, BDD_Output::ClassificationCounts>& classificationCounts, int totalSamples) {
    int correctPredictions = 0;

    for (const auto& entry : classificationCounts) {
        const BDD_Output::ClassificationCounts& cc = entry.second;

        int left_majority_class_count = 0;
        for (int count : cc.left_class_counts) {
            left_majority_class_count = std::max(left_majority_class_count, count);
        }

        // Majority class for the right edge
        int right_majority_class_count = 0;
        for (int count : cc.right_class_counts) {
            right_majority_class_count = std::max(right_majority_class_count, count);
        }

        // Add majority counts to correct predictions
        correctPredictions += left_majority_class_count;
        correctPredictions += right_majority_class_count;
    }


    return totalSamples - correctPredictions;
}


float BDD_Output::calculateTestAccuracy(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const {

    std::map<int, BDD_Output::ClassificationCounts> counts = computeClassificationCounts(root, inputBDD.data, nodeFeatures, nodeThresholds);

    int correct_classified = 0;
    for (const auto& sample : data) {
        std::vector<int>& final_edge_count = classifySample(root, sample, nodeFeatures, nodeThresholds, counts);
        auto maxIt = std::max_element(final_edge_count.begin(), final_edge_count.end());

        // Get the index of the max element
        int maxIndex = std::distance(final_edge_count.begin(), maxIt);

        int classLabel = static_cast<int>(sample.back());
        if (maxIndex == classLabel) {
            correct_classified += 1;
        }  
    }

    return correct_classified * 1.0f / data.size();
}

float BDD_Output::calculateAccuracy(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const {

    std::map<int, BDD_Output::ClassificationCounts> counts = computeClassificationCounts(root, data, nodeFeatures, nodeThresholds);
    float acc = computeAccuracyFromCounts(counts, inputBDD.data.size());

    return acc;
}

int BDD_Output::calculateMissclassified(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data) const
{
    std::map<int, BDD_Output::ClassificationCounts> counts = computeClassificationCounts(root, data, nodeFeatures, nodeThresholds);
	int missclassified = computeMisclassificationsFromCounts(counts, inputBDD.data.size());

    return missclassified;
}

std::map<int, BDD_Output::ClassificationCounts> BDD_Output::computeClassificationCounts(std::shared_ptr<Node> root, const std::vector<std::vector<float>>& data, const std::vector<int>& selectedFeatures, const std::vector<float>& selectedThresholds) const {
    std::map<int, ClassificationCounts> classificationCounts;

    for (const auto& sample : data) {
		std::vector<int>& final_edge_count = classifySample(root, sample, selectedFeatures, selectedThresholds, classificationCounts);
        int classLabel = static_cast<int>(sample.back());
        if (final_edge_count.size() <= classLabel) {
            final_edge_count.resize(inputBDD.numberOfClasses, 0);
        }
		final_edge_count[classLabel]++;
    }

    return classificationCounts;
}

// Compute the classification counts when using the new values for the node we are trying to optimize
std::map<int, BDD_Output::ClassificationCounts> BDD_Output::computeClassificationCountsUpdatedNode(std::shared_ptr<Node> root, int optimizing_id, int optimizing_feature, float optimizing_threshold, std::vector<int>& relevant_samples, const std::vector<std::vector<float>>& data) const {
    
    std::map<int, ClassificationCounts> classificationCounts;

    for (int i = 0; i < data.size(); i++) {
        const auto& sample = data[i];

        bool relevant_sample = false;
        std::vector<int>& final_edge_count = classifySampleUpdatedNode(root, sample, optimizing_id, optimizing_feature, optimizing_threshold, classificationCounts, relevant_sample);
        
        int classLabel = static_cast<int>(sample.back());
        
        if (final_edge_count.size() <= classLabel) {
            final_edge_count.resize(inputBDD.numberOfClasses, 0);
        }

        final_edge_count[classLabel]++;

		// Only if the sample has passed through the optimizing node, we add it to the relevant samples
        if (relevant_sample) {
            relevant_samples.push_back(i);
        }
    }

    return classificationCounts;
}


// Given a feature and the data, calculate the optimal threshold for the current node
float BDD_Output::findOptimalFeatureThreshold(int node_id, int feature) const
{   
    const std::vector<std::vector<float>>& sorted = inputBDD.data_sorted_per_feature[feature];
    float current_threshold = std::numeric_limits<float>::lowest();

    std::vector<int> relevant_samples;
    std::map<int, BDD_Output::ClassificationCounts> counts = computeClassificationCountsUpdatedNode(inputBDD.root, node_id, feature, std::numeric_limits<float>::lowest(), relevant_samples, sorted);
	
    float size = 0;
    for (int i = 0; i < nodeThresholds.size(); i++) {
        if (nodeThresholds[i] != std::numeric_limits<float>::lowest() && nodeThresholds[i] != std::numeric_limits<float>::max() && i != node_id) {
            size++;
        }
    }

    int cur_missclassified = computeMisclassificationsFromCounts(counts, inputBDD.data.size());
    float best_cost = (inputBDD.baselineMultiplier * cur_missclassified) + inputBDD.alpha * size;
    float best_threshold = current_threshold;
    float prev_smaller = std::numeric_limits<float>::lowest();

	// Iterate through all relevant samples and find the best threshold
    for (int index = 0; index < relevant_samples.size(); index++) {
        int i = relevant_samples[index];
		std::vector<float> sample = sorted[i];

		// If current threshold is equal to current sample, need to change since it must be strictly smaller
        if (current_threshold == sorted[i][feature]) {
			current_threshold = (sorted[i][feature] + prev_smaller) / 2.0f;
        }

        // Get the final edge that the sample visits
        bool relevant = false;
		std::vector<int>& final_edge_count = classifySampleUpdatedNode(inputBDD.nodes[node_id], sample, node_id, feature, current_threshold, counts, relevant);
        if (final_edge_count.size() <= sample.back()) {
            final_edge_count.resize(inputBDD.numberOfClasses, 0);
        }
        final_edge_count[static_cast<int>(sample.back())]--;


		// Update threshold to be the average of the current and next sample
        if (index < relevant_samples.size() - 1) {
            // If current and next sample have same value, update prev_smaller
            int j = relevant_samples[index + 1];
            if (sorted[i][feature] < sorted[j][feature]) {
                prev_smaller = sorted[i][feature];
            }
            current_threshold = (sorted[i][feature] + sorted[j][feature]) / 2.0f;
        }
        else {
            current_threshold = std::numeric_limits<float>::max();
            size = size - 1;
        }

        std::vector<int>& new_final_edge_count = classifySampleUpdatedNode(inputBDD.nodes[node_id], sample, node_id, feature, current_threshold, counts, relevant);
        if (new_final_edge_count.size() <= sample.back()) {
            new_final_edge_count.resize(inputBDD.numberOfClasses, 0);
        }
        new_final_edge_count[static_cast<int>(sample.back())]++;


        // If current threshold is equal to the current sample, that means it has the same value as its next sample
		// and we therefore cannot compute the accuracy yet
        // For example if we have 14 17 17 17 20, then when we set threshold to 17, first 17 might be classified correctly while
        // we dont take into account how the other 17s are classified
        
        int missclassified = computeMisclassificationsFromCounts(counts, inputBDD.data.size());
        float cost = (inputBDD.baselineMultiplier * missclassified) + inputBDD.alpha * (size+1);

		if (cost < best_cost && current_threshold != sorted[i][feature]) {
			best_cost = cost;
			best_threshold = current_threshold;
		}
	}

	prev_optimal_cost = best_cost;
    return best_threshold;
}


void BDD_Output::outputDiagram() const {
    
    std::vector<int> nodeCounts(inputBDD.diagramSize, 0);
    std::map<int, ClassificationCounts> class_counts;

    for (int i = 0; i < inputBDD.nodes.size(); i++) {
        class_counts[i].left_class_counts.resize(inputBDD.numberOfClasses);
        class_counts[i].right_class_counts.resize(inputBDD.numberOfClasses);
    }

    for (int i = 0; i < inputBDD.data.size(); i++) {
        std::shared_ptr<Node> node = inputBDD.root;
		std::vector<float> sample = inputBDD.data[i];
        while (true) {
            if (node == nullptr) {
                node = inputBDD.root;
                break;
            }
			nodeCounts[node->id] += 1;

            bool traverse_left = (!isFlipped[node->id] && sample[nodeFeatures[node->id]] <= nodeThresholds[node->id])
                || (isFlipped[node->id] && sample[nodeFeatures[node->id]] > nodeThresholds[node->id]);

            if (traverse_left) {
                class_counts[node->id].left_class_counts[sample.back()]++;
            }
            else {
                class_counts[node->id].right_class_counts[sample.back()]++;
            }

            if (node->leftChild == nullptr && node->rightChild == nullptr) {
                node = nullptr;
            } else if (traverse_left) {
                node = inputBDD.nodes[leftChild[node->id]];
            }
            else {
                node = inputBDD.nodes[rightChild[node->id]];
            }
        }
    }

    std::cout << "\n" << inputBDD.diagramSize << "\n";
    for (auto& node : inputBDD.nodes) {
        if (node->leftChild && node->rightChild) {
            std::cout << "normal," << node->id << "," << node->layer << "," << leftChild[node->id] << "," << rightChild[node->id] << "," << nodeFeatures[node->id] << "," << nodeThresholds[node->id] << "," << nodeCounts[node->id] << "\n";
        }
        else {
            std::cout << "final," << node->id << "," << node->layer << ","  << nodeFeatures[node->id] << "," << nodeThresholds[node->id] << "," << nodeCounts[node->id] << "\n";

        }

        for (size_t i = 0; i < class_counts[node->id].left_class_counts.size(); ++i) {
            std::cout << class_counts[node->id].left_class_counts[i];
            if (i != class_counts[node->id].left_class_counts.size() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "\n";

        for (size_t i = 0; i < class_counts[node->id].right_class_counts.size(); ++i) {
            std::cout << class_counts[node->id].right_class_counts[i];
            if (i != class_counts[node->id].right_class_counts.size() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "\n";

    };
}

int BDD_Output::getDepth(std::shared_ptr<Node> current_node) const
{   
    if (current_node->leftChild && current_node->rightChild) {
        if (nodeThresholds[current_node->id] == std::numeric_limits<float>::lowest() ||
            nodeThresholds[current_node->id] == std::numeric_limits<float>::max()) {
            return std::max(getDepth(inputBDD.nodes[leftChild[current_node->id]]), getDepth(inputBDD.nodes[rightChild[current_node->id]]));

        }
        else {
            return std::max(getDepth(inputBDD.nodes[leftChild[current_node->id]]), getDepth(inputBDD.nodes[rightChild[current_node->id]])) + 1;
        }
    }
    else {
        return 1;
    }   
}

float BDD_Output::getQuestionLength() const
{
    int total = 0;
    for (const auto& sample : inputBDD.data) {
        std::shared_ptr<Node> node = inputBDD.root;

        while (node != nullptr) {
            int feature = nodeFeatures[node->id];
            float threshold = nodeThresholds[node->id];
            float value = sample[feature];

            bool traverse_left = (!isFlipped[node->id] && value <= threshold)
                || (isFlipped[node->id] && value > threshold);

            if (threshold != std::numeric_limits<float>::lowest() &&
                threshold != std::numeric_limits<float>::max()) {
                total++;
            }

            if (node->leftChild == nullptr) {
                node = nullptr;
            }
            else if (traverse_left) {
                node = inputBDD.nodes[leftChild[node->id]];
            }
            else {
                node = inputBDD.nodes[rightChild[node->id]];
            }
        }
    }

    return 1.0f * total / inputBDD.data.size();
}

float BDD_Output::getFragmentation(int size) const
{   
	std::map<int, int> sample_count_per_node;

    for (const auto& sample : inputBDD.data) {
        std::shared_ptr<Node> node = inputBDD.root;

        while (node != nullptr) {
            int feature = nodeFeatures[node->id];
            float threshold = nodeThresholds[node->id];
            float value = sample[feature];

            bool traverse_left = (!isFlipped[node->id] && value <= threshold)
                || (isFlipped[node->id] && value > threshold);

			sample_count_per_node[node->id]++;

            if (node->leftChild == nullptr) {
                node = nullptr;
            }
            else if (traverse_left) {
                node = inputBDD.nodes[leftChild[node->id]];
            }
            else {
                node = inputBDD.nodes[rightChild[node->id]];
            }
        }
    }

    float avg_fragmentation = 0.0f;
    for (auto& node : inputBDD.nodes) {
        if (nodeThresholds[node->id] != std::numeric_limits<float>::lowest() &&
            nodeThresholds[node->id] != std::numeric_limits<float>::max()) {
			avg_fragmentation += static_cast<float>(sample_count_per_node[node->id]) / inputBDD.data.size();
        }
	}


    return avg_fragmentation / size;
}

    
void BDD_Input::setRoot(std::shared_ptr<Node>& rootNode) {
    root = rootNode;
    nodes.push_back(rootNode);
}

void BDD_Input::addNode(std::shared_ptr<Node>& node) {
    nodes.push_back(node);
}



void BDD_Input::loadDiagram(std::string fileName) {
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << fileName << std::endl;
        return;
    }

    std::string line;
    bool firstLine = true;  // Used to skip the first line

    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            diagramSize = std::stoi(line);

            for (int i = 0; i < diagramSize; i++) {
                if (i == 0) {
                    auto root = std::make_shared<Node>(i);
                    setRoot(root);
                }
                else {
                    addNode(std::make_shared<Node>(i));
                }
            }
        }
        else {
            std::stringstream ss(line);
            std::string type;

            std::getline(ss, type, ',');

            if (type == "normal") {
                std::string temp;
                int node_id, layer_num, left_id, right_id, feature;
                float threshold;

                std::getline(ss, temp, ',');
                node_id = std::stoi(temp);
                std::getline(ss, temp, ',');
                layer_num = std::stoi(temp);
                std::getline(ss, temp, ',');
                left_id = std::stoi(temp);
                std::getline(ss, temp, ',');
                right_id = std::stoi(temp);
                std::getline(ss, temp, ',');
                feature = std::stoi(temp);
                std::getline(ss, temp, ',');
                threshold = std::stof(temp);


				nodes[node_id]->layer = layer_num;
                nodes[node_id]->leftChild = nodes[left_id];
                nodes[node_id]->rightChild = nodes[right_id];
				greedyFeatures.push_back(feature);
				greedyThresholds.push_back(threshold);

                // Save the nodes per layer so we can easily get nodes in next layer
                nodes_per_layer[layer_num].push_back(node_id);
            }
            else {
                std::string temp;
                int node_id, layer_num, feature;
                float threshold;

                std::getline(ss, temp, ',');
                node_id = std::stoi(temp);
                std::getline(ss, temp, ',');
                layer_num = std::stoi(temp);
                std::getline(ss, temp, ',');
                feature = std::stoi(temp);
                std::getline(ss, temp, ',');
                threshold = std::stof(temp);

                nodes[node_id]->layer = layer_num;
                greedyFeatures.push_back(feature);
                greedyThresholds.push_back(threshold);

                nodes_per_layer[layer_num].push_back(node_id);
            }
        }
    }

}


std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));  // Convert string to float
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

int getMaxLastColumnValues(const std::vector<std::vector<float>>& data) {
    float maxVal = std::numeric_limits<float>::lowest();
    for (const auto& row : data) {
        if (!row.empty()) {
            maxVal = std::max(maxVal, row.back());
        }
    }

    return static_cast<int>(maxVal);
}

int getMostFrequentLastColumnCount(const std::vector<std::vector<float>>& data) {
    std::unordered_map<float, int> frequency;
    int maxCount = 0;

    for (const auto& row : data) {
        if (!row.empty()) {
            float value = row.back();
            frequency[value]++;
            maxCount = std::max(maxCount, frequency[value]);
        }
    }

    return maxCount;
}

void BDD_Input::loadData(std::string filename_train, std::string filename_test) {

    data = readCSV("../temp/" + filename_train);
    test_data = readCSV("../temp/" + filename_test);

    numberOfClasses = getMaxLastColumnValues(data) + 1;
	numberOfFeatures = data[0].size() - 1;
	baselineMultiplier = 1.0f / (data.size() - getMostFrequentLastColumnCount(data));

    // Save data sorted per feature
    for (int i = 0; i < data[0].size() - 1; i++) {
        std::vector<std::vector<float>> sorted_current_feature = data;
        std::sort(sorted_current_feature.begin(), sorted_current_feature.end(), [i](const std::vector<float>& a, const std::vector<float>& b) {
            return a[i] < b[i];
            });

        data_sorted_per_feature.push_back(sorted_current_feature);
    }

}

bool operator==(const BDD_Move& m1, const BDD_Move& m2) {
    return m1.nodeID == m2.nodeID
        && m1.prevFeature == m2.prevFeature
        && m1.newFeature == m2.newFeature
        && m1.prevThreshold == m2.prevThreshold
        && m1.newThreshold == m2.newThreshold;
}

bool operator!=(const BDD_Move& m1, const BDD_Move& m2) {
    return m1.nodeID != m2.nodeID
        || m1.prevFeature != m2.prevFeature
        || m1.newFeature != m2.newFeature
        || m1.prevThreshold != m2.prevThreshold
        || m1.newThreshold != m2.newThreshold;
}

bool operator<(const BDD_Move& m1, const BDD_Move& m2) {
    return m1.nodeID < m2.nodeID;
}

std::ostream& operator<<(std::ostream& os, const BDD_Move& m) {
    return os;
}

std::istream& operator>>(std::istream& is, BDD_Move& m) {
    return is;
}




bool operator==(const BDD_FlipMove& m1, const BDD_FlipMove& m2) {
    return m1.nodeID == m2.nodeID
        && m1.prevValue == m2.prevValue
        && m1.newValue == m2.newValue;
}

bool operator!=(const BDD_FlipMove& m1, const BDD_FlipMove& m2) {
    return m1.nodeID != m2.nodeID
        || m1.prevValue != m2.prevValue
        || m1.newValue != m2.newValue;
}

bool operator<(const BDD_FlipMove& m1, const BDD_FlipMove& m2) {
    return m1.nodeID < m2.nodeID;
}

std::ostream& operator<<(std::ostream& os, const BDD_FlipMove& m) {
    return os;
}

std::istream& operator>>(std::istream& is, BDD_FlipMove& m) {
    return is;
}

bool operator==(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2)
{   
    return m1.nodeID == m2.nodeID
        && m1.prev_child_id == m2.prev_child_id
        && m1.new_child_id == m2.new_child_id
        && m1.left_edge == m2.left_edge;
}

bool operator!=(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2)
{
    return m1.nodeID != m2.nodeID
        || m1.prev_child_id != m2.prev_child_id
        || m1.new_child_id != m2.new_child_id
        || m1.left_edge != m2.left_edge;
}

bool operator<(const BDD_EdgeMove& m1, const BDD_EdgeMove& m2)
{
    return m1.nodeID < m2.nodeID;
}

std::ostream& operator<<(std::ostream& os, const BDD_EdgeMove& m)
{
    return os;
}

std::istream& operator>>(std::istream& is, BDD_EdgeMove& m)
{
    return is;
}


