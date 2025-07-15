#include "BDD_helpers.hh"
#include "BDD_data.hh"

#include <chrono>
#include <string>
#include <cmath>

using namespace std::chrono;
using namespace EasyLocal::Debug;


int main(int argc, const char* argv[])
{
#if !defined(NDEBUG)
    std::cout << "This code is running in DEBUG mode" << std::endl;
#endif
    ParameterBox main_parameters("main", "Main Program options");
    Parameter<std::string> method("method", "Solution method (empty for tester)", main_parameters);
    Parameter<std::string> initialization("initialization", "Greedy or random", main_parameters);
    Parameter<float> alpha_value("alpha", "Alpha", main_parameters);
    Parameter<int> ils_perturbations("ILS_perturbations", "Number of pertrubations for each ILS round", main_parameters);
    Parameter<int> ils_iterations("ILS_iterations", "Number of iterations for ILS", main_parameters);
    Parameter<std::string> file_train("filename_train", "Datasetname for training", main_parameters);
    Parameter<std::string> file_test("filename_test", "Datasetname for evaluating", main_parameters);
    Parameter<unsigned int> seed("seed", "Random seed", main_parameters);
    Parameter<float> r_move("random_move", "Random move", main_parameters);
    Parameter<float> o_move("optimal_move", "Optimal move", main_parameters);
    Parameter<float> c_move("complex_move", "Complex move", main_parameters);
    Parameter<float> e_move("edge_move", "Edge move", main_parameters);

    // CommandLineParameters::Parse(argc, argv, true, false);       // Used this once normally
    CommandLineParameters::Parse(argc, argv, false, true);


	std::string filename_train = file_train.ToString();
	std::string filename_test = file_test.ToString();

    // Create and initial diagram
    BDD_Input diagram = BDD_Input();

    if (seed.IsSet())
    {
        Random::SetSeed(seed);
        diagram.seed = seed;
    }

    diagram.loadDiagram("../temp/diagram");
    diagram.loadData(filename_train, filename_test);

    if (initialization == std::string("greedy")) {
        diagram.greedyInitialization = true;
    }

    if (alpha_value.IsSet()) {
        diagram.alpha = alpha_value;
    }


    BDD_SizeAndAccuracyCost cc = BDD_SizeAndAccuracyCost(diagram, 1, false); // 3rd should be false, because not hard constraint
	BDD_DeltaCostComponent deltaCC = BDD_DeltaCostComponent(diagram, cc);
	BDD_FlipDeltaCostComponent deltaFlipCC = BDD_FlipDeltaCostComponent(diagram, cc);
	BDD_EdgeDeltaCostComponent deltaEdgeCC = BDD_EdgeDeltaCostComponent(diagram, cc);

    BDD_SolutionManager sm(diagram);
    
    BDD_RandomSplitNE randomNE(diagram, sm);
    BDD_OptimalSplitNE optimalNE(diagram, sm);
    BDD_OptimalCompleteNE optimalCompleteNE(diagram, sm);
    BDD_RandomFlipNE randomFlipNE(diagram, sm);
	BDD_RandomEdge randomEdgeNE(diagram, sm);


    sm.AddCostComponent(cc);

    randomNE.AddCostComponent(cc);
    optimalNE.AddCostComponent(cc);
	optimalCompleteNE.AddCostComponent(cc);
	randomFlipNE.AddCostComponent(cc);
	randomEdgeNE.AddCostComponent(cc);

    randomNE.AddDeltaCostComponent(deltaCC);
    optimalNE.AddDeltaCostComponent(deltaCC);
	optimalCompleteNE.AddDeltaCostComponent(deltaCC);
    randomFlipNE.AddDeltaCostComponent(deltaFlipCC);
	randomEdgeNE.AddDeltaCostComponent(deltaEdgeCC);


    SetUnionNeighborhoodExplorer<BDD_Input, BDD_Output, DefaultCostStructure<float>, decltype(randomNE), decltype(optimalNE), decltype(randomEdgeNE)> multiNeighborhoodExplorer(diagram, sm, "MULTI", randomNE, optimalNE, randomEdgeNE, {r_move, o_move, e_move});

    HillClimbing<BDD_Input, BDD_Output, decltype(multiNeighborhoodExplorer)::MoveType, DefaultCostStructure<float>> hc(diagram, sm, multiNeighborhoodExplorer, "HC1");
    SimulatedAnnealing<BDD_Input, BDD_Output, decltype(multiNeighborhoodExplorer)::MoveType, DefaultCostStructure<float>> sa(diagram, sm, multiNeighborhoodExplorer, "SA1");
    
    SimpleLocalSearch<BDD_Input, BDD_Output, DefaultCostStructure<float>> BDD_solver(diagram, sm, "BDD_solver");

	if (method == std::string("HC")) {
		BDD_solver.SetRunner(hc);
	}
	else if (method == std::string("SA")) {
		BDD_solver.SetRunner(sa);
	}
	else {
		BDD_solver.SetRunner(hc);
	}

    CommandLineParameters::Parse(argc, argv, false, true);

    SolverResult<BDD_Input, BDD_Output, DefaultCostStructure<float>> result(diagram);
      

    if (!(method == std::string("ILS"))) {
        auto start = std::chrono::high_resolution_clock::now();
        result = BDD_solver.Solve();
        BDD_Output out = result.output;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        out.printDiagram();
        out.outputDiagram();
       
        float train_acc = result.output.calculateAccuracy(out.inputBDD.root, diagram.data);
        float test_acc = result.output.calculateTestAccuracy(out.inputBDD.root, diagram.test_data);

        float size = 0;
        // Calculate number of activated nodes
        for (float threshold : result.output.nodeThresholds) {
            if (threshold != std::numeric_limits<float>::lowest() && threshold != std::numeric_limits<float>::max()) {
                size++;
            }
        }

        std::cout << "\n";

        int depth = result.output.getDepth(result.output.inputBDD.root);
		float question_length = result.output.getQuestionLength();
		float fragmentation = result.output.getFragmentation(size);

        std::cout << "LS," << filename_train << "," << method.ToString() << "," << result.cost.total  << "," << initialization.ToString() << ","
            << size << "," << diagram.numberOfFeatures << "," << diagram.numberOfClasses << "," << diagram.data.size() + diagram.test_data.size() 
            << "," << train_acc << "," << test_acc << "," << elapsed.count() << "," << cc.convergence_time.back().count() << ","
            << depth << "," << question_length << "," << fragmentation << ",";

    }
    else {

        auto start = std::chrono::high_resolution_clock::now();

        int number_iterations = 5;
		if (ils_iterations.IsSet()) {
			number_iterations = ils_iterations;
		}

        for (int i = 0; i < number_iterations; i++) {
            result = BDD_solver.Solve();
            BDD_Output out = result.output;

            sm.iteratedLocalSearch = true;
			sm.saved_isFlipped = out.isFlipped;
			sm.saved_nodeFeatures = out.nodeFeatures;
			sm.saved_nodeThresholds = out.nodeThresholds;
			sm.saved_leftChild = out.leftChild;
			sm.saved_rightChild = out.rightChild;


            int num_perturbations = 0.15f * diagram.nodes.size();
			if (ils_perturbations.IsSet()) {
				num_perturbations = ils_perturbations;
			}


            for (int j = 0; j < num_perturbations; j++) {
                int randNode = Random::Uniform<int>(0, diagram.nodes.size() - 1);
                int randFeature = Random::Uniform<int>(0, diagram.numberOfFeatures - 1);
                float randSplit = diagram.data[Random::Uniform<int>(0, diagram.data.size() - 1)][randFeature];

                sm.saved_nodeFeatures[randNode] = randFeature;
                sm.saved_nodeThresholds[randNode] = randSplit;
            }
        }

        BDD_Output out = result.output;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        float train_acc = result.output.calculateAccuracy(out.inputBDD.root, diagram.data);
        float test_acc = result.output.calculateTestAccuracy(out.inputBDD.root, diagram.test_data);

        float size = 0;
        for (float threshold : result.output.nodeThresholds) {
            if (threshold != std::numeric_limits<float>::lowest() && threshold != std::numeric_limits<float>::max()) {
                size++;
            }
        }

        int depth = result.output.getDepth(result.output.inputBDD.root);
        float question_length = result.output.getQuestionLength();
        float fragmentation = result.output.getFragmentation(size);

        std::cout << "\n";
        std::cout << "LS," << filename_train << "," << method.ToString() << "," << result.cost.total << "," << initialization.ToString() << "," << size << "," << diagram.numberOfFeatures << "," << diagram.numberOfClasses << "," << diagram.data.size() + diagram.test_data.size() << "," << train_acc << "," << test_acc << "," << elapsed.count() << "," << cc.convergence_time.back().count()
            << "," << depth << "," << question_length << "," << fragmentation << ",";
    }

#if !defined(NDEBUG)
    std::cout << "\nThis code is running in DEBUG mode" << std::endl;
#endif
    return 0; 
}