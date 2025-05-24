# Genetic Programming Classifier

This project implements a **Genetic Programming (GP) Classifier** in Java to evolve mathematical expression trees capable of classifying financial data instances (e.g., stock market data) into binary labels (0.0 or 1.0). The core functionality includes genetic operators like subtree crossover and grow mutation.

---

### Project Structure

* `GP_Classifier.java`: Main class containing the genetic programming implementation.
* `Node`: Abstract class representing nodes in an expression tree.
* `FunctionNode`, `TerminalNode`: Tree node subclasses for operators and feature terminals.

---

### How It Works

1. **Initialization**: Generates a population of random expression trees.
2. **Evaluation**: Classifies training data and scores each tree using accuracy.
3. **Selection**: Selects parent trees via tournament selection.
4. **Crossover**: Combines two trees by exchanging random subtrees.
5. **Mutation**: Replaces a subtree using grow mutation.
6. **Evolution**: Repeats for multiple generations, preserving the best-performing tree.

---

### Key Features

* Tree-based representation of solutions.
* Genetic operators:

  * Subtree crossover
  * Grow mutation
* Fitness evaluation based on **classification accuracy**
* Supports feature-based terminals and arithmetic functions (`+`, `-`, `*`, `/`)
* Works with CSV datasets of financial data (e.g., Open, Close, High, Low)

---

### Example Use Case

This classifier can be used to predict binary outcomes such as:

* Will a stock price go up tomorrow? (`1.0`) or not (`0.0`)
* Is a transaction suspicious?

---

### Input Format

CSV file with the following columns:

```csv
Open,High,Low,Close,Volume,Label
132.34,135.12,131.20,134.75,1000000,1.0
...
```

* `Label`: Binary classification target (`1.0` or `0.0`)

---

### Configuration

You can modify these values in `GP_Classifier.java`:

```java
int POPULATION_SIZE = 100;
int NUM_GENERATIONS = 50;
int MAX_TREE_DEPTH = 5;
double CROSSOVER_RATE = 0.9;
double MUTATION_RATE = 0.1;
```

---

### How to Run

1. Compile:

   ```bash
   javac GP_Classifier.java
   ```

2. Run:

   ```bash
   java GP_Classifier 
   ```

3. Run the Executable JAR

   ```bash
   java -jar GPClassifier.jar
   ```

4. Run Weka tests

   ```bash
   javac -cp ".;weka.jar;mtj-1.0.4.jar;netlib-java-1.1.jar;native_ref-java-1.1.jar;native_system-java-1.1.jar" GP_Classifier.java
   java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;weka.jar;mtj-1.0.4.jar;netlib-java-1.1.jar native_ref-java-1.1.jar;native_system-java-1.1.jar" GP_Classifier

   ```
---


### Output

* Displays training and testing accuracy.
* Prints the best evolved tree and its fitness score.
* Displays F1 score and confusion matrix
* 

