import java.io.*;
import java.util.*;

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

    abstract int classify(DataPoint dp);

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
    int classify(DataPoint dp) {
        return (int) evaluate(dp);
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
    int classify(DataPoint dp) {
        return (int) value;
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
    int classify(DataPoint dp) {
        return (int) evaluate(dp);
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
        return condition.evaluate(dp) != 0 ? thenBranch.evaluate(dp) : elseBranch.evaluate(dp);
    }

    @Override
    int classify(DataPoint dp) {
        return (int) evaluate(dp);
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
    private static final int MAX_GENERATIONS = 100;
    private static final int POPULATION_SIZE = 100;
    private static final int MAX_TREE_DEPTH = 5;
    private static final int SEED = 42;
    private static final double CROSSOVER_RATE = 0.7;
    private static final double MUTATION_RATE = 0.1;

    private static final String[] FEATURES = { "Open", "High", "Low", "Close", "AdjClose" };
    private static final String[] OPERATORS = { "+", "-", "*", "/", ">", "<" };

    public static void main(String[] args) throws IOException {
        List<DataPoint> dataset = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_train.csv");

        List<DataPoint> trainingData = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_train.csv");
        List<DataPoint> testingData = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_test.csv");

        Node evolvedTree = evolveTree(trainingData);

        int correct = 0;
        for (DataPoint dp : testingData) {
            if (evolvedTree.classify(dp) == dp.output) correct++;
        }


        System.out.println("Accuracy: " + (correct * 100.0 / dataset.size()) + "%");
       //  System.out.println("Rule: " + evolvedTree.print());
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
            population.add(randomTree(MAX_TREE_DEPTH));
        }

        Node best = null;
        double bestFitness = -1;

        // 2. Evolution loop
        for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
            List<Node> newPopulation = new ArrayList<>();

            while (newPopulation.size() < POPULATION_SIZE) {
                Node parent1 = selectTree(population, dataset);
                Node parent2 = selectTree(population, dataset);

                Node child;
                if (rand.nextDouble() < CROSSOVER_RATE) {
                    child = crossover(parent1, parent2);
                } else {
                    child = cloneTree(parent1);
                }

                if (rand.nextDouble() < MUTATION_RATE) {
                    child = mutate(child);
                }

                newPopulation.add(child);
            }

            // Evaluate best
            for (Node n : newPopulation) {
                double f = fitness(n, dataset);
                if (f > bestFitness) {
                    bestFitness = f;
                    best = n;
                }
            }

            population = newPopulation;
            // System.out.println("Gen " + gen + " | Best Fitness: " + bestFitness);
        }

        return best;
    }

    private static Node randomTree(int depth) {
        Random rand = new Random(SEED);
        if (depth == 0 || rand.nextDouble() < 0.3) {
            if (rand.nextBoolean()) {
                String feature = FEATURES[rand.nextInt(FEATURES.length)];
                return new TerminalNode(feature);
            } else {
                return new ConstantNode(rand.nextDouble() * 50000); // Random constant
            }
        } else {
            Node left = randomTree(depth - 1);
            Node right = randomTree(depth - 1);
            String op = OPERATORS[rand.nextInt(OPERATORS.length)];
            return new OperatorNode(op, left, right);
        }
    }

    private static Node selectTree(List<Node> population, List<DataPoint> dataset) {
        Node best = null;
        double bestFit = -1;
        Random rand = new Random(SEED);
        for (int i = 0; i < 5; i++) {
            Node candidate = population.get(rand.nextInt(population.size()));
            double fit = fitness(candidate, dataset);
            if (fit > bestFit) {
                best = candidate;
                bestFit = fit;
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

    private static Node crossover(Node parent1, Node parent2) {
        // Implement crossover logic
        Random rand = new Random(SEED);
        if (rand.nextDouble() > CROSSOVER_RATE) return parent1;

        return crossoverSubtree(parent1, parent2, MAX_TREE_DEPTH);
    }
    
    private static Node crossoverSubtree(Node a, Node b, int maxDepth) {
        Random rand = new Random(SEED);
        if (rand.nextDouble() < 0.1 || maxDepth == 0 || isLeaf(a) || isLeaf(b)) {
            return copyTree(rand.nextBoolean() ? a : b); // swap
        }

        if (a instanceof OperatorNode && b instanceof OperatorNode) {
            OperatorNode ao = (OperatorNode) a;
            OperatorNode bo = (OperatorNode) b;

            return new OperatorNode(ao.operator,
                crossoverSubtree(ao.left, bo.left, maxDepth - 1),
                crossoverSubtree(ao.right, bo.right, maxDepth - 1)
            );
        } else if (a instanceof IfNode && b instanceof IfNode) {
            IfNode ai = (IfNode) a;
            IfNode bi = (IfNode) b;

            return new IfNode(
                crossoverSubtree(ai.condition, bi.condition, maxDepth - 1),
                crossoverSubtree(ai.thenBranch, bi.thenBranch, maxDepth - 1),
                crossoverSubtree(ai.elseBranch, bi.elseBranch, maxDepth - 1)
            );
        }

        return rand.nextBoolean() ? copyTree(a) : copyTree(b);
    }

    private static Node mutate(Node tree) { // grow mutation for exploration
        // Implement mutation logic
        Random rand = new Random(SEED);
        if (rand.nextDouble() < 0.1) {
            return randomTree(MAX_TREE_DEPTH); // replace whole subtree
        }
        return mutateSubtree(tree, MAX_TREE_DEPTH);
    }    

    private static Node mutateSubtree(Node node, int depth) {
        if (depth == 0 || node instanceof TerminalNode || node instanceof ConstantNode) {
            return randomTree(1); // Replace with a small random subtree
        }

        Random rand = new Random(SEED);
        if (node instanceof OperatorNode) {
            OperatorNode op = (OperatorNode) node;
            if (rand.nextBoolean()) {
                op.left = mutateSubtree(op.left, depth - 1);
            } else {
                op.right = mutateSubtree(op.right, depth - 1);
            }
        } else if (node instanceof IfNode) {
            IfNode ifn = (IfNode) node;
            int choice = rand.nextInt(3);
            if (choice == 0) ifn.condition = mutateSubtree(ifn.condition, depth - 1);
            else if (choice == 1) ifn.thenBranch = mutateSubtree(ifn.thenBranch, depth - 1);
            else if (choice == 2) ifn.elseBranch = mutateSubtree(ifn.elseBranch, depth - 1);
        }

        return node;
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
    private static Node copyTree(Node node) {
        if (node instanceof TerminalNode) {
            return new TerminalNode(((TerminalNode) node).feature);
        } else if (node instanceof ConstantNode) {
            return new ConstantNode(((ConstantNode) node).value);
        } else if (node instanceof OperatorNode) {
            OperatorNode op = (OperatorNode) node;
            return new OperatorNode(op.operator, copyTree(op.left), copyTree(op.right));
        } else if (node instanceof IfNode) {
            IfNode ifn = (IfNode) node;
            return new IfNode(copyTree(ifn.condition), copyTree(ifn.thenBranch), copyTree(ifn.elseBranch));
        }

        return null;
    }

    private static boolean isLeaf(Node node) {
        return (node instanceof TerminalNode || node instanceof ConstantNode);
    }   
}
