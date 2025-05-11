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
            case "Open": return dp.open;
            case "High": return dp.high;
            case "Low": return dp.low;
            case "Close": return dp.close;
            case "AdjClose": return dp.adjClose;
            default: return 0;
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
            case "+": return l + r;
            case "-": return l - r;
            case "*": return l * r;
            case "/": return (r == 0 ? 0 : l / r);
            case ">": return l > r ? 1 : 0;
            case "<": return l < r ? 1 : 0;
            default: return 0;
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
            if (parts.length < 6) continue;
            double open = Double.parseDouble(parts[0]);
            double high = Double.parseDouble(parts[1]);
            double low = Double.parseDouble(parts[2]);
            double close = Double.parseDouble(parts[3]);
            double adjClose = Double.parseDouble(parts[4]);
            int output = Integer.parseInt(parts[5]);

            data.add(new DataPoint(open, high, low, close, adjClose, output));
        }

        return data;
    }
}

// === Main (for testing purposes) ===
public class GP_Classifier {
    public static void main(String[] args) throws IOException {
        List<DataPoint> dataset = DataLoader.loadCSV("./Euro_USD_STOCK/BTC_test.csv");

        // Create a test tree manually for now
        Node tree = new IfNode(
            new OperatorNode(">", new TerminalNode("Close"), new ConstantNode(30000)),
            new ConstantNode(1),
            new ConstantNode(0)
        );

        int correct = 0;
        for (DataPoint dp : dataset) {
            int prediction = tree.classify(dp);
            if (prediction == dp.output) correct++;
        }

        System.out.println("Accuracy: " + (correct * 100.0 / dataset.size()) + "%");
        System.out.println("Rule: " + tree.print());
    }
}
