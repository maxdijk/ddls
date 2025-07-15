#include "BDD_helpers.hh"
#include <map>
#include <cmath>
#include <algorithm>
#include <random>


// Depending on the initialization method 
// this function initializes the BDD state with random features and thresholds or uses a greedy initialization.
// Or uses saved values from a previous ILS run.
void BDD_SolutionManager::RandomState(BDD_Output& st) {
	if (iteratedLocalSearch) {
		st.nodeThresholds = saved_nodeThresholds;
		st.nodeFeatures = saved_nodeFeatures;
		st.isFlipped = saved_isFlipped;
		st.leftChild = saved_leftChild;
		st.rightChild = saved_rightChild;
		return;
	}

	if (st.inputBDD.greedyInitialization) {
		GreedyState(st);
		return;
	}

	std::vector<float> newNodeThresholds;
	std::vector<int> newNodeFeatures;
	std::vector<bool> newisFlipped;
	std::vector<int> newLeftChild;
	std::vector<int> newRightChild;
	for (auto& node : st.inputBDD.nodes) {
		int randFeature = Random::Uniform<int>(0, st.inputBDD.numberOfFeatures-1);
		float randSplit = st.inputBDD.data[Random::Uniform<int>(0, st.inputBDD.data.size()-1)][randFeature];

		newNodeFeatures.push_back(randFeature);
		newNodeThresholds.push_back(randSplit);
		newisFlipped.push_back(false);

		if (node->leftChild) {
			newLeftChild.push_back(node->leftChild->id);
			newRightChild.push_back(node->rightChild->id);
		}
		else {
			newLeftChild.push_back(-1);
			newRightChild.push_back(-1);
		}
	}

	st.nodeFeatures = newNodeFeatures;
	st.nodeThresholds = newNodeThresholds;
	st.isFlipped = newisFlipped;
	st.leftChild = newLeftChild;
	st.rightChild = newRightChild;

	//st.printDiagram();
	//st.outputDiagram();
	//std::cout << "\Initial Accuracy: " << st.calculateAccuracy(st.inputBDD.root, st.inputBDD.data) << "\n";
}

bool BDD_SolutionManager::CheckConsistency(const BDD_Output& st) const {
	return true;
}

// Initializa the state of the BDD based on the stored features and thresholds from the initial greedy solution.
void BDD_SolutionManager::GreedyState(BDD_Output& st)
{	
	for (int i = 0; i < st.inputBDD.diagramSize; i++) {
		if (st.inputBDD.greedyFeatures[i] == -1) {
			st.nodeFeatures.push_back(0);
			st.nodeThresholds.push_back(std::numeric_limits<float>::lowest());
		}
		else {
			st.nodeFeatures.push_back(st.inputBDD.greedyFeatures[i]);
			st.nodeThresholds.push_back(st.inputBDD.greedyThresholds[i]);
		}

		st.isFlipped.push_back(false);

		auto node = st.inputBDD.nodes[i];
		if (node->leftChild) {
			st.leftChild.push_back(node->leftChild->id);
			st.rightChild.push_back(node->rightChild->id);
		}
		else {
			st.leftChild.push_back(-1);
			st.rightChild.push_back(-1);
		}
	}

	//st.printDiagram();
	//st.outputDiagram();
	//std::cout << "\Initial Accuracy: " << st.calculateAccuracy(st.inputBDD.root, st.inputBDD.data) << "\n";
}

// Make a random move that splits a node based on a random feature and threshold.
void BDD_RandomSplitNE::RandomMove(const BDD_Output& st, BDD_Move& mv) const {

	int randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size()-1);
	int randFeature = Random::Uniform<int>(0, st.inputBDD.numberOfFeatures - 1);
	float randSplit = st.inputBDD.data[Random::Uniform<int>(0, st.inputBDD.data.size() - 1)][randFeature];

	mv.nodeID = randNode;
	mv.prevFeature = st.nodeFeatures[randNode];
	mv.newFeature = randFeature;
	mv.prevThreshold = st.nodeThresholds[randNode];

	int rand = Random::Uniform<int>(0, st.inputBDD.data.size());
	if (rand == 0) {
		mv.newThreshold = std::numeric_limits<float>::lowest();
	} else if (rand == 1) {
		mv.newThreshold = std::numeric_limits<float>::max();
	} else {
		mv.newThreshold = randSplit;
	}

	st.prev_move_optimal = false;
}

void BDD_RandomSplitNE::MakeMove(BDD_Output& st, const BDD_Move& mv) const {
	st.nodeFeatures[mv.nodeID] = mv.newFeature;
	st.nodeThresholds[mv.nodeID] = mv.newThreshold;
	st.up_to_date = false;

}

void BDD_RandomSplitNE::FirstMove(const BDD_Output& st, BDD_Move& mv) const {
	NextMove(st, mv);
}


bool BDD_RandomSplitNE::NextMove(const BDD_Output& st, BDD_Move& mv) const {

	return false;
}

float compute_size(const BDD_Output& st)
{
	float total = 0;
	for (float threshold : st.nodeThresholds) {
		if (threshold != std::numeric_limits<float>::lowest() && threshold != std::numeric_limits<float>::max()) {
			total++;
		}
	}
	return total;
}

// Compute the cost of the BDD based on size and accuracy.
float BDD_SizeAndAccuracyCost::ComputeCost(const BDD_Output& st) const
{
	float missclassified = st.calculateMissclassified(st.inputBDD.root, st.inputBDD.data);

	float cost = (st.inputBDD.baselineMultiplier * missclassified) + st.inputBDD.alpha * compute_size(st);

	float acc = (st.inputBDD.data.size() - missclassified) / st.inputBDD.data.size();
	if (acc > best_accuracy) {
		convergence_acc.push_back(acc);
		convergence_time.push_back(std::chrono::high_resolution_clock::now() - start_time);
		best_accuracy = acc;
	}

	return cost;
}

void BDD_SizeAndAccuracyCost::PrintViolations(const BDD_Output& st, std::ostream& os) const {
	os << "PrintViolations Testing";
}

// Make a move that optimally splits a node based given a random feature.
void BDD_OptimalSplitNE::RandomMove(const BDD_Output& st, BDD_Move& mv) const
{
	int randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size() - 1);
	int randFeature = Random::Uniform<int>(0, st.inputBDD.numberOfFeatures - 1);

	float optimalSplit = st.findOptimalFeatureThreshold(randNode, randFeature);

	mv.nodeID = randNode;
	mv.prevFeature = st.nodeFeatures[randNode];
	mv.newFeature = randFeature;
	mv.prevThreshold = st.nodeThresholds[randNode];
	mv.newThreshold = optimalSplit;

	st.prev_move_optimal = true;
}

void BDD_OptimalSplitNE::MakeMove(BDD_Output& st, const BDD_Move& mv) const
{
	st.nodeFeatures[mv.nodeID] = mv.newFeature;
	st.nodeThresholds[mv.nodeID] = mv.newThreshold;
	st.up_to_date = false;
}

// Not needed for HC/SA/ILS
void BDD_OptimalSplitNE::FirstMove(const BDD_Output& st, BDD_Move& mv) const
{
	NextMove(st, mv);
}

// Not needed for HC/SA/ILS
bool BDD_OptimalSplitNE::NextMove(const BDD_Output& st, BDD_Move& mv) const
{
	return false;
}

void BDD_RandomFlipNE::RandomMove(const BDD_Output& st, BDD_FlipMove& mv) const
{
	int randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size() - 1);

	while (!st.inputBDD.nodes[randNode]->leftChild) {
		randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size() - 1);
	}
	mv.nodeID = randNode;
	mv.prevValue = st.isFlipped[randNode];
	mv.newValue = !mv.prevValue;

	st.prev_move_optimal = false;
}

void BDD_RandomFlipNE::MakeMove(BDD_Output& st, const BDD_FlipMove& mv) const
{
	st.isFlipped[mv.nodeID] = mv.newValue;
	st.up_to_date = false;

}

// Not needed for HC/SA/ILS
void BDD_RandomFlipNE::FirstMove(const BDD_Output& st, BDD_FlipMove& mv) const
{
	NextMove(st, mv);
}

// Not needed for HC/SA/ILS
bool BDD_RandomFlipNE::NextMove(const BDD_Output& st, BDD_FlipMove& mv) const
{
	return false;
}

// Calculate difference in cost between the current state and the new state after applying the BDD_Move.
float BDD_DeltaCostComponent::ComputeDeltaCost(const BDD_Output& st, const BDD_Move& mv) const
{
	BDD_Output new_st = st;
	new_st.nodeFeatures[mv.nodeID] = mv.newFeature;
	new_st.nodeThresholds[mv.nodeID] = mv.newThreshold;

	float cur_cost;
	if (st.up_to_date) {
		cur_cost = st.current_cost;
	}
	else {
		cur_cost = cc.ComputeCost(st);
		st.current_cost = cur_cost;
		st.up_to_date = true;
	}

	float new_cost;
	// If the previous move was optimal, we can use the stored cost
	if (st.prev_move_optimal) {
		new_cost = st.prev_optimal_cost;
		st.prev_move_optimal = false;
	}
	else {
		new_cost = cc.ComputeCost(new_st);
	}

	return new_cost - cur_cost;
}

float BDD_FlipDeltaCostComponent::ComputeDeltaCost(const BDD_Output& st, const BDD_FlipMove& mv) const
{
	BDD_Output new_st = st;
	new_st.isFlipped[mv.nodeID] = mv.newValue;

	float cur_cost;
	if (st.up_to_date) {
		cur_cost = st.current_cost;
	}
	else {
		cur_cost = cc.ComputeCost(st);
		st.current_cost = cur_cost;
		st.up_to_date = true;
	}

	float new_cost = cc.ComputeCost(new_st);
	return new_cost - cur_cost;
}

// Make an edge of a random node point towards a random node in next layer
void BDD_RandomEdge::RandomMove(const BDD_Output& st, BDD_EdgeMove& mv) const
{
	int randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size() - 1);
	// Ensure that the random node has children
	while (!st.inputBDD.nodes[randNode]->leftChild) {
		randNode = Random::Uniform<int>(0, st.inputBDD.nodes.size() - 1);
	}

	std::vector<int> next_layer = st.inputBDD.nodes_per_layer.at(st.inputBDD.nodes[randNode]->layer + 1);
	std::mt19937 gen(st.inputBDD.seed);                        
	std::uniform_int_distribution<> distr(0, next_layer.size() - 1);
	std::uniform_int_distribution<> left(0, 1);

	mv.nodeID = randNode;

	if (left(gen) == 0) {
		int new_child_id = next_layer[distr(gen)];
		while (new_child_id == st.rightChild[randNode]) {
			new_child_id = next_layer[distr(gen)];
		}

		mv.prev_child_id = st.leftChild[randNode];
		mv.new_child_id = new_child_id;
		mv.left_edge = true;
		return;
	}
	else {
		int new_child_id = next_layer[distr(gen)];
		while (new_child_id == st.leftChild[randNode]) {
			new_child_id = next_layer[distr(gen)];
		}

		mv.prev_child_id = st.rightChild[randNode];
		mv.new_child_id = new_child_id;
		mv.left_edge = false;
		return;
	}
	
	st.prev_move_optimal = false;
}

void BDD_RandomEdge::MakeMove(BDD_Output& st, const BDD_EdgeMove& mv) const
{
	if (mv.left_edge) {
		st.leftChild[mv.nodeID] = mv.new_child_id;
	}
	else {
		st.rightChild[mv.nodeID] = mv.new_child_id;
	}

	st.up_to_date = false;
}

// Not needed for HC/SA/ILS
void BDD_RandomEdge::FirstMove(const BDD_Output& st, BDD_EdgeMove& mv) const
{
	NextMove(st, mv);
}

// Not needed for HC/SA/ILS
bool BDD_RandomEdge::NextMove(const BDD_Output& st, BDD_EdgeMove& mv) const
{
	return false;
}

// Calculate difference in cost between the current state and the new state after applying the edge move.
float BDD_EdgeDeltaCostComponent::ComputeDeltaCost(const BDD_Output& st, const BDD_EdgeMove& mv) const
{
	BDD_Output new_st = st;
	if (mv.left_edge) {
		new_st.leftChild[mv.nodeID] = mv.new_child_id;
	}
	else {
		new_st.rightChild[mv.nodeID] = mv.new_child_id;
	}

	float cur_cost;
	if (st.up_to_date) {
		cur_cost = st.current_cost;
	}
	else {
		cur_cost = cc.ComputeCost(st);
		st.current_cost = cur_cost;
		st.up_to_date = true;
	}

	float new_cost = cc.ComputeCost(new_st);
	return new_cost - cur_cost;
}

void BDD_OptimalCompleteNE::RandomMove(const BDD_Output& st, BDD_Move& mv) const
{

	float best_objective = 10000.0;
	float best_threshold = -1.0;
	int best_feature = -1;
	int best_node = -1;
	
	for (auto& node : st.inputBDD.nodes) {
		for (int i = 0; i < st.inputBDD.numberOfFeatures; i++) {
			float optimalSplit = st.findOptimalFeatureThreshold(node->id, i);
			if (st.prev_optimal_cost < best_objective) {
				best_objective = st.prev_optimal_cost;
				best_threshold = optimalSplit;
				best_feature = i;
				best_node = node->id;
			}
		}
	}

	mv.nodeID = best_node;
	mv.prevFeature = st.nodeFeatures[best_node];
	mv.newFeature = best_feature;
	mv.prevThreshold = st.nodeThresholds[best_node];
	mv.newThreshold = best_threshold;

	st.prev_move_optimal = true;
	st.prev_optimal_cost = best_objective;
}

void BDD_OptimalCompleteNE::MakeMove(BDD_Output& st, const BDD_Move& mv) const
{
	st.nodeFeatures[mv.nodeID] = mv.newFeature;
	st.nodeThresholds[mv.nodeID] = mv.newThreshold;
	st.up_to_date = false;
}

// Not needed for HC/SA/ILS
void BDD_OptimalCompleteNE::FirstMove(const BDD_Output& st, BDD_Move& mv) const
{
	NextMove(st, mv);
}


// Not needed for HC/SA/ILS
bool BDD_OptimalCompleteNE::NextMove(const BDD_Output& st, BDD_Move& mv) const
{
	return false;
}
