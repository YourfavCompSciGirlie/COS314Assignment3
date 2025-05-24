import java.io.*;
import java.util.*;

import weka.core.Instances;
import weka.classifiers.Evaluation;

// === Data Model ===
class DataPoint {
    double open, high, low, close, adjClose;
    int output;

    public DataPoint(double open, double high, double low, double close, double adjClose, int output) {
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.adjClose = adjClose;
        this.output = output;
    }
}

// === Abstract Node ===
abstract class Node {
    abstract double evaluate(DataPoint dp);

    int classify(DataPoint dp) {
        return (int) evaluate(dp) > 0 ? 1 : 0; // Example classification
    }

    abstract String print();
}

// === Terminal Node ===
class TerminalNode extends Node {
    String feature;

    public TerminalNode(String feature) {
        this.feature = feature;
    }

    @Override
    double evaluate(DataPoint dp) {
        switch (feature) {
            case "Open":
                return dp.open;
            case "High":
                return dp.high;
            case "Low":
                return dp.low;
            case "Close":
                return dp.close;
            case "AdjClose":
                return dp.adjClose;
            default:
                return 0;
        }
    }

    @Override
    String print() {
        return feature;
    }
}

// === Constant Node ===
class ConstantNode extends Node {
    double value;

    public ConstantNode(double value) {
        this.value = value;
    }

    @Override
    double evaluate(DataPoint dp) {
        return value;
    }

    @Override
    String print() {
        return String.valueOf(value);
    }
}

// === Operator Node (e.g., +, -, >, <) ===
class OperatorNode extends Node {
    String operator;
    Node left, right;

    public OperatorNode(String operator, Node left, Node right) {
        this.operator = operator;
        this.left = left;
        this.right = right;
    }

    @Override
    double evaluate(DataPoint dp) {
        double l = left.evaluate(dp);
        double r = right.evaluate(dp);

        switch (operator) {
            case "+":
                return l + r;
            case "-":
                return l - r;
            case "*":
                return l * r;
            case "/":
                return (r == 0 ? 0 : l / r);
            case ">":
                return l > r ? 1 : 0;
            case "<":
                return l < r ? 1 : 0;
            default:
                return 0;
        }
    }

    @Override
    String print() {
        return "(" + left.print() + " " + operator + " " + right.print() + ")";
    }
}

// === If Node (ternary logic) ===
class IfNode extends Node {
    Node condition, thenBranch, elseBranch;

    public IfNode(Node condition, Node thenBranch, Node elseBranch) {
        this.condition = condition;
        this.thenBranch = thenBranch;
        this.elseBranch = elseBranch;
    }

    @Override
    double evaluate(DataPoint dp) {
        return condition.evaluate(dp) > 0 ? thenBranch.evaluate(dp) : elseBranch.evaluate(dp);
    }

    @Override
    String print() {
        return "if(" + condition.print() + ", " + thenBranch.print() + ", " + elseBranch.print() + ")";
    }
}

// === CSV Loader ===
class DataLoader {
    public static List<DataPoint> loadCSV(String filePath) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line = br.readLine(); // skip header

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            if (parts.length < 6)
                continue;
            double open = Double.parseDouble(parts[0]);
            double high = Double.parseDouble(parts[1]);
            double low = Double.parseDouble(parts[2]);
            double close = Double.parseDouble(parts[3]);
            double adjClose = Double.parseDouble(parts[4]);
            int output = Integer.parseInt(parts[5]);

            data.add(new DataPoint(open, high, low, close, adjClose, output));
        }
        br.close();
        return data;
    }
}

// === Main (for testing purposes) ===
public class GP_Classifier {
    private static int MAX_GENERATIONS = 50;
    private static int POPULATION_SIZE = 500;
    private static final int MAX_TREE_DEPTH = 5;
    private static final int TOURNAMENT_SIZE = 5;
    private static int SEED = 42;
    private static final boolean ELITISM = true;
    private static final int ELITISM_COUNT = 5;
    private static final double CROSSOVER_RATE = 0.7;
    private static final double MUTATION_RATE = 0.1;

    private static final String[] FEATURES = { "Open", "High", "Low", "Close", "AdjClose" };
    private static final String[] OPERATORS = { "+", "-", "*", "/", ">", "<" };

    public static void main(String[] args) throws IOException {
        // Load datasets

        System.out.println("===== Generic Programming Stock Price Classifier =====");
        System.out.println("Enter seed value: ");
        Scanner scanner = new Scanner(System.in);
        SEED = scanner.nextInt();

        System.out.println("Enter population size: ");
        POPULATION_SIZE = scanner.nextInt();

        System.out.println("Enter max generations: ");
        MAX_GENERATIONS = scanner.nextInt();
        scanner.close();

        List<DataPoint> trainingData = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_train.csv");
        List<DataPoint> testingData = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_test.csv");

        System.out.println("\nStarting evolution...");
        Node evolvedTree = evolveTree(trainingData);
        System.out.println("Evolution complete! \n");

        // Calculate training accuracy
        double trainingAccuracy = calculateAccuracy(evolvedTree, trainingData);
        System.out.println("Training Accuracy: " + (trainingAccuracy * 100.0) + "%");

        // Calculate testing accuracy
        double testingAccuracy = calculateAccuracy(evolvedTree, testingData);
        System.out.println("Testing Accuracy: " + (testingAccuracy * 100.0) + "%");

        System.out.println("\n Rule: " + evolvedTree.print() + "\n");

        // Output confusion matrix for test data
        int[] confusionMatrix = calculateConfusionMatrix(evolvedTree, testingData);
        System.out.println("Confusion Matrix (Test Data):");
        System.out.println("True Positive: " + confusionMatrix[0]);
        System.out.println("False Positive: " + confusionMatrix[1]);
        System.out.println("False Negative: " + confusionMatrix[2]);
        System.out.println("True Negative: " + confusionMatrix[3]);

        int[] confusionMatrixTrain = calculateConfusionMatrix(evolvedTree, trainingData);

        double f1Score = calculateF1Score(confusionMatrix);
        double f1ScoreTrain = calculateF1Score(confusionMatrixTrain);
        System.out.println("\nF1 Score (Test Data): " + f1Score);
        System.out.println("F1 Score (Train Data): " + f1ScoreTrain);

        System.out.println("\nRunning Weka GP Classifier...");

        // Load Weka datasets once
        Instances train = null;
        Instances test = null;
        try {
            weka.core.converters.ConverterUtils.DataSource trainSource = new weka.core.converters.ConverterUtils.DataSource("./Euro_USD_Stock/BTC_train.arff");
            weka.core.converters.ConverterUtils.DataSource testSource = new weka.core.converters.ConverterUtils.DataSource("./Euro_USD_Stock/BTC_test.arff");
            train = trainSource.getDataSet();
            test = testSource.getDataSet();
            if (train.classIndex() == -1)
                train.setClassIndex(train.numAttributes() - 1);
            if (test.classIndex() == -1)
                test.setClassIndex(test.numAttributes() - 1);
        } catch (Exception e) {
            System.err.println("Failed to load Weka datasets.");
            e.printStackTrace();
            return;
        }

        // Prepare file writer for output scores
        BufferedWriter writer = new BufferedWriter(new FileWriter("../wilxon_test/gp_f1_scores.txt", false)); // overwrite
                                                                                                                // old
                                                                                                                // file

        System.out.println("Running 10 GP trials for Wilcoxon test:");
        for (int i = 0; i < 10; i++) {
            int currentSeed = 42 + i; // Different seed each time
            try {
                double f1ScoreWeka = runGP(train, test, currentSeed);
                System.out.printf("Run %d (Seed %d) - F1 Score: %.4f%n", i + 1, currentSeed, f1ScoreWeka);
                writer.write(f1ScoreWeka + "\n");
            } catch (Exception e) {
                System.err.printf("Run %d failed due to error:%n", i + 1);
                e.printStackTrace();
            }
        }

        writer.close();
        System.out.println("All scores written to ../wilxon_test/gp_f1_scores.txt");
    }

    private static Node evolveTree(List<DataPoint> dataset) {
        // 1. Initialize population of random trees
        // 2. Evaluate fitness of each tree
        // 3. Evolve over generations using:
        // - Selection
        // - Crossover
        // - Mutation
        // 4. Return the best performing tree

        Random rand = new Random(SEED);
        List<Node> population = new ArrayList<>();

        // 1. Initial population
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population.add(randomTree(rand, MAX_TREE_DEPTH));
        }

        Node bestEver = null;
        double bestEverFitness = -1;

        // 2. Evolution loop
        for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
            // Calculate fitness for all individuals
            Map<Node, Double> fitnessMap = new HashMap<>();
            for (Node individual : population) {
                double fitness = fitness(individual, dataset);
                fitnessMap.put(individual, fitness);

                // Track best overall
                if (fitness > bestEverFitness) {
                    bestEverFitness = fitness;
                    bestEver = cloneTree(individual);
                }
            }

            // Sort population by fitness
            population.sort((a, b) -> Double.compare(fitnessMap.get(b), fitnessMap.get(a)));

            // Report progress
            // double avgFitness =
            // fitnessMap.values().stream().mapToDouble(Double::doubleValue).average().orElse(0);

            // Create new population
            List<Node> newPopulation = new ArrayList<>();

            // Apply elitism if enabled
            if (ELITISM) {
                for (int i = 0; i < Math.min(ELITISM_COUNT, population.size()); i++) {
                    newPopulation.add(cloneTree(population.get(i)));
                }
            }

            // Fill the rest of the population
            while (newPopulation.size() < POPULATION_SIZE) {
                Node parent1 = tournamentSelect(population, fitnessMap, rand);
                Node parent2 = tournamentSelect(population, fitnessMap, rand);

                Node child1, child2;
                if (rand.nextDouble() < CROSSOVER_RATE) {
                    Node[] children = crossover(parent1, parent2, rand);
                    child1 = children[0];
                    child2 = children[1];
                } else {
                    child1 = cloneTree(parent1);
                    child2 = cloneTree(parent2);
                }

                if (rand.nextDouble() < MUTATION_RATE) {
                    child1 = mutate(child1, rand);
                }
                if (rand.nextDouble() < MUTATION_RATE) {
                    child2 = mutate(child2, rand);
                }

                newPopulation.add(child1);
                if (newPopulation.size() < POPULATION_SIZE) {
                    newPopulation.add(child2);
                }
            }

            population = newPopulation;
        }

        return bestEver != null ? bestEver : population.get(0);

    }

    private static Node randomTree(Random rand, int maxDepth) {
        if (maxDepth == 0 || (maxDepth < 3 && rand.nextDouble() < 0.6)) {
            if (rand.nextDouble() < 0.7) {
                String feature = FEATURES[rand.nextInt(FEATURES.length)];
                return new TerminalNode(feature);
            } else {
                // Random constant, scaled appropriately for financial data
                double value = (rand.nextDouble() - 0.5) * 100;
                return new ConstantNode(value);
            }
        } else {
            // Decide what type of node to create
            double nodeType = rand.nextDouble();

            if (nodeType < 0.7) { // Create operator node
                Node left = randomTree(rand, maxDepth - 1);
                Node right = randomTree(rand, maxDepth - 1);
                String op = OPERATORS[rand.nextInt(OPERATORS.length)];
                return new OperatorNode(op, left, right);
            } else { // Create if node
                Node condition = randomTree(rand, maxDepth - 1);
                Node thenBranch = randomTree(rand, maxDepth - 1);
                Node elseBranch = randomTree(rand, maxDepth - 1);
                return new IfNode(condition, thenBranch, elseBranch);
            }
        }
    }

    private static Node tournamentSelect(List<Node> population, Map<Node, Double> fitnessMap, Random rand) {
        Node best = null;
        double bestFitness = -1;

        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Node candidate = population.get(rand.nextInt(population.size()));
            double fitness = fitnessMap.get(candidate);

            if (best == null || fitness > bestFitness) {
                best = candidate;
                bestFitness = fitness;
            }
        }

        return best;
    }

    private static double fitness(Node tree, List<DataPoint> dataset) {
        int correct = 0;
        for (DataPoint dp : dataset) {
            if (tree.classify(dp) == dp.output)
                correct++;
        }
        return correct / (double) dataset.size();
    }

    private static Node[] crossover(Node parent1, Node parent2, Random rand) {
        // Implement crossover logic
        Node[] result = new Node[2];
        result[0] = cloneTree(parent1);
        result[1] = cloneTree(parent2);

        // Get all nodes from each tree
        List<Node> nodesP1 = getAllNodes(result[0]);
        List<Node> nodesP2 = getAllNodes(result[1]);

        if (nodesP1.isEmpty() || nodesP2.isEmpty()) {
            return result;
        }

        // Select random crossover points
        Node crossPoint1 = nodesP1.get(rand.nextInt(nodesP1.size()));
        Node crossPoint2 = nodesP2.get(rand.nextInt(nodesP2.size()));

        // Replace node in first tree
        replaceNode(result[0], crossPoint1, cloneSubtree(crossPoint2));

        // Replace node in second tree (for second child)
        replaceNode(result[1], crossPoint2, cloneSubtree(crossPoint1));

        return result;
    }

    private static List<Node> getAllNodes(Node root) {
        List<Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        return nodes;
    }

    private static void collectNodes(Node node, List<Node> nodes) {
        if (node == null)
            return;

        nodes.add(node);

        if (node instanceof OperatorNode) {
            OperatorNode op = (OperatorNode) node;
            collectNodes(op.left, nodes);
            collectNodes(op.right, nodes);
        } else if (node instanceof IfNode) {
            IfNode ifn = (IfNode) node;
            collectNodes(ifn.condition, nodes);
            collectNodes(ifn.thenBranch, nodes);
            collectNodes(ifn.elseBranch, nodes);
        }
    }

    private static boolean replaceNode(Node tree, Node target, Node replacement) {
        if (tree instanceof OperatorNode) {
            OperatorNode op = (OperatorNode) tree;

            if (op.left == target) {
                op.left = replacement;
                return true;
            }
            if (op.right == target) {
                op.right = replacement;
                return true;
            }

            return replaceNode(op.left, target, replacement) ||
                    replaceNode(op.right, target, replacement);

        } else if (tree instanceof IfNode) {
            IfNode ifn = (IfNode) tree;

            if (ifn.condition == target) {
                ifn.condition = replacement;
                return true;
            }
            if (ifn.thenBranch == target) {
                ifn.thenBranch = replacement;
                return true;
            }
            if (ifn.elseBranch == target) {
                ifn.elseBranch = replacement;
                return true;
            }

            return replaceNode(ifn.condition, target, replacement) ||
                    replaceNode(ifn.thenBranch, target, replacement) ||
                    replaceNode(ifn.elseBranch, target, replacement);
        }

        return false;
    }

    private static Node mutate(Node tree, Random rand) { // grow mutation for exploration
        Node mutated = cloneTree(tree);
        List<Node> allNodes = getAllNodes(mutated);

        if (allNodes.isEmpty()) {
            return mutated;
        }

        // Select random node to mutate
        Node targetNode = allNodes.get(rand.nextInt(allNodes.size()));

        // Select mutation type
        double mutationType = rand.nextDouble();

        if (mutationType < 0.3) {
            // Replace with completely new random subtree
            Node replacement = randomTree(rand, 3); // Limit depth of new random tree
            replaceNode(mutated, targetNode, replacement);
        } else if (mutationType < 0.6) {
            // Point mutation - modify node but keep structure
            if (targetNode instanceof TerminalNode) {
                // Change the feature
                ((TerminalNode) targetNode).feature = FEATURES[rand.nextInt(FEATURES.length)];
            } else if (targetNode instanceof ConstantNode) {
                // Change the constant value
                ((ConstantNode) targetNode).value = (rand.nextDouble() - 0.5) * 100;
            } else if (targetNode instanceof OperatorNode) {
                // Change the operator
                ((OperatorNode) targetNode).operator = OPERATORS[rand.nextInt(OPERATORS.length)];
            }
        } else {
            // Grow mutation - add complexity
            if (targetNode instanceof TerminalNode || targetNode instanceof ConstantNode) {
                Node replacement;
                if (rand.nextBoolean()) {
                    // Replace with an operator node
                    replacement = new OperatorNode(
                            OPERATORS[rand.nextInt(OPERATORS.length)],
                            cloneTree(targetNode),
                            randomTree(rand, 2));
                } else {
                    // Replace with an if node
                    replacement = new IfNode(
                            randomTree(rand, 2),
                            cloneTree(targetNode),
                            randomTree(rand, 2));
                }
                replaceNode(mutated, targetNode, replacement);
            }
        }

        return mutated;
    }

    private static Node cloneSubtree(Node node) {
        return cloneTree(node);
    }

    private static Node cloneTree(Node tree) {
        if (tree instanceof TerminalNode) {
            return new TerminalNode(((TerminalNode) tree).feature);
        } else if (tree instanceof ConstantNode) {
            return new ConstantNode(((ConstantNode) tree).value);
        } else if (tree instanceof OperatorNode) {
            OperatorNode op = (OperatorNode) tree;
            return new OperatorNode(op.operator, cloneTree(op.left), cloneTree(op.right));
        } else if (tree instanceof IfNode) {
            IfNode ifn = (IfNode) tree;
            return new IfNode(cloneTree(ifn.condition), cloneTree(ifn.thenBranch), cloneTree(ifn.elseBranch));
        }
        return null;
    }

    private static double calculateAccuracy(Node tree, List<DataPoint> dataset) {
        int correct = 0;
        for (DataPoint dp : dataset) {
            if (tree.classify(dp) == dp.output) {
                correct++;
            }
        }
        return correct / (double) dataset.size();
    }

    private static int[] calculateConfusionMatrix(Node tree, List<DataPoint> dataset) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int trueNegatives = 0;

        for (DataPoint dp : dataset) {
            int prediction = tree.classify(dp);
            int actual = dp.output;

            if (prediction == 1 && actual == 1) {
                truePositives++;
            } else if (prediction == 1 && actual == 0) {
                falsePositives++;
            } else if (prediction == 0 && actual == 1) {
                falseNegatives++;
            } else if (prediction == 0 && actual == 0) {
                trueNegatives++;
            }
        }

        return new int[] { truePositives, falsePositives, falseNegatives, trueNegatives };
    }

    private static double calculateF1Score(int[] confusionMatrix) {
        int truePositives = confusionMatrix[0];
        int falsePositives = confusionMatrix[1];
        int falseNegatives = confusionMatrix[2];

        // Calculate precision and recall
        double precision = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falsePositives);
        double recall = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falseNegatives);

        // Calculate F1 score
        double f1Score = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

        return f1Score;
    }

    public static double runGP(Instances train, Instances test, int seed) throws Exception {
        weka.classifiers.Classifier gp = new weka.classifiers.functions.GaussianProcesses();
        gp.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(gp, test);

        double f1Score = eval.fMeasure(1); // assuming class "1" is the positive class
        System.out.printf("Seed %d Test F1: %.4f\n", seed, f1Score);
        return f1Score;
    }

}
