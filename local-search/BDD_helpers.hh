#pragma once

#include "BDD_data.hh"
#include <easylocal.hh>

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <chrono>


using namespace EasyLocal::Core;



class BDD_SolutionManager : public SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>
{
public:
    BDD_SolutionManager(const BDD_Input& pin) : SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>(pin, "BDD_SolutionManager") {}
    void RandomState(BDD_Output& st);
    bool CheckConsistency(const BDD_Output& st) const;
    void GreedyState(BDD_Output& st);

	// Saved values necessary for iterated local search
	bool iteratedLocalSearch = false;
    std::vector<float> saved_nodeThresholds;
    std::vector<int> saved_nodeFeatures;
    std::vector<bool> saved_isFlipped;
	std::vector<int> saved_leftChild;
	std::vector<int> saved_rightChild;
};

class BDD_RandomSplitNE : public NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>
{
public:
    BDD_RandomSplitNE(const BDD_Input& pin, SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>& psm) :
        NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>(pin, psm, "BDD_MoveNeighborhoodExplorerr") {}
    void RandomMove(const BDD_Output& st, BDD_Move& mv) const override;
    void MakeMove(BDD_Output& st, const BDD_Move& mv) const override;
    void FirstMove(const BDD_Output& st, BDD_Move& mv) const override;
    bool NextMove(const BDD_Output& st, BDD_Move& mv) const override;
};


class BDD_OptimalSplitNE : public NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>
{
public:
    BDD_OptimalSplitNE(const BDD_Input& pin, SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>& psm) :
        NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>(pin, psm, "BDD_OptimalSplitNE") {}
    void RandomMove(const BDD_Output& st, BDD_Move& mv) const override;
    void MakeMove(BDD_Output& st, const BDD_Move& mv) const override;
    void FirstMove(const BDD_Output& st, BDD_Move& mv) const override;
    bool NextMove(const BDD_Output& st, BDD_Move& mv) const override;
};


class BDD_OptimalCompleteNE : public NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>
{
public:
    BDD_OptimalCompleteNE(const BDD_Input& pin, SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>& psm) :
        NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_Move, DefaultCostStructure<float>>(pin, psm, "BDD_OptimalCompleteNE") {}
    void RandomMove(const BDD_Output& st, BDD_Move& mv) const override;
    void MakeMove(BDD_Output& st, const BDD_Move& mv) const override;
    void FirstMove(const BDD_Output& st, BDD_Move& mv) const override;
    bool NextMove(const BDD_Output& st, BDD_Move& mv) const override;
};

class BDD_RandomFlipNE : public NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_FlipMove, DefaultCostStructure<float>>
{
public:
    BDD_RandomFlipNE(const BDD_Input& pin, SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>& psm) :
        NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_FlipMove, DefaultCostStructure<float>>(pin, psm, "BDD_RandomFlipNE") {}
    void RandomMove(const BDD_Output& st, BDD_FlipMove& mv) const override;
    void MakeMove(BDD_Output& st, const BDD_FlipMove& mv) const override;
    void FirstMove(const BDD_Output& st, BDD_FlipMove& mv) const override;
    bool NextMove(const BDD_Output& st, BDD_FlipMove& mv) const override;
};

class BDD_RandomEdge : public NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_EdgeMove, DefaultCostStructure<float>>
{
public:
    BDD_RandomEdge(const BDD_Input& pin, SolutionManager<BDD_Input, BDD_Output, DefaultCostStructure<float>>& psm) :
        NeighborhoodExplorer<BDD_Input, BDD_Output, BDD_EdgeMove, DefaultCostStructure<float>>(pin, psm, "BDD_RandomEdge") {}
    void RandomMove(const BDD_Output& st, BDD_EdgeMove& mv) const override;
    void MakeMove(BDD_Output& st, const BDD_EdgeMove& mv) const override;
    void FirstMove(const BDD_Output& st, BDD_EdgeMove& mv) const override;
    bool NextMove(const BDD_Output& st, BDD_EdgeMove& mv) const override;
};

// Cost component used for computing the cost of a BDD solution
class  BDD_SizeAndAccuracyCost : public CostComponent<BDD_Input, BDD_Output, float>
{
public:
    BDD_SizeAndAccuracyCost(const BDD_Input& in, int w, bool hard) : CostComponent<BDD_Input, BDD_Output, float>(in, w, hard, "BDD_SizeAndAccuracyCost") {}
    float ComputeCost(const BDD_Output& st) const;
    void PrintViolations(const BDD_Output& st, std::ostream& os = std::cout) const;

    mutable float best_accuracy = 0;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    mutable std::vector<float> convergence_acc;
    mutable std::vector<std::chrono::duration<double>> convergence_time;
};

// Delta cost component used for computing the delta cost of a BDD_Move
class BDD_DeltaCostComponent
    : public DeltaCostComponent<BDD_Input, BDD_Output, BDD_Move, float>
{
public:
    BDD_DeltaCostComponent(const BDD_Input& in, BDD_SizeAndAccuracyCost& cc)
        : DeltaCostComponent<BDD_Input, BDD_Output, BDD_Move, float>(in, cc, "BDD_DeltaCost") {}
    float ComputeDeltaCost(const BDD_Output& st, const BDD_Move& mv) const;
};

// Delta cost component used for computing the delta cost of a BDD_FlipMove
class BDD_FlipDeltaCostComponent
    : public DeltaCostComponent<BDD_Input, BDD_Output, BDD_FlipMove, float>
{
public:
    BDD_FlipDeltaCostComponent(const BDD_Input& in, BDD_SizeAndAccuracyCost& cc)
        : DeltaCostComponent<BDD_Input, BDD_Output, BDD_FlipMove, float>(in, cc, "BDD_DeltaCost") {}
    float ComputeDeltaCost(const BDD_Output& st, const BDD_FlipMove& mv) const;
};

// Delta cost component used for computing the delta cost of a BDD_EdgeMove
class BDD_EdgeDeltaCostComponent
    : public DeltaCostComponent<BDD_Input, BDD_Output, BDD_EdgeMove, float>
{
public:
    BDD_EdgeDeltaCostComponent(const BDD_Input& in, BDD_SizeAndAccuracyCost& cc)
        : DeltaCostComponent<BDD_Input, BDD_Output, BDD_EdgeMove, float>(in, cc, "BDD_DeltaCost") {}
    float ComputeDeltaCost(const BDD_Output& st, const BDD_EdgeMove& mv) const;
};