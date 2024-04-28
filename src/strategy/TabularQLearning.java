package strategy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

/**
 * This class implements Q-Learning with a tabular representation of the Q-values.
 */
public class TabularQLearning extends QLearningStrategy {

    // Q-table to store state-action values
    private final HashMap<String, double[]> QTable;

    // Maze dimensions
    private final int sizeMazeX;
    private final int sizeMazeY;

    /**
     * Constructor for TabularQLearning.
     *
     * @param epsilon exploration rate
     * @param gamma discount factor
     * @param alpha learning rate
     * @param sizeMazeX width of the maze
     * @param sizeMazeY height of the maze
     */
    public TabularQLearning(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY, int nbWalls) {
        super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

        this.sizeMazeX = sizeMazeX;
        this.sizeMazeY = sizeMazeY;

        // Initialize Q-table 
        QTable = new HashMap<>();
    }

    /**
     * Encodes the current PacmanGame state into a string representation.
     *
     * @param state the current PacmanGame state
     * @return the encoded state string
     */
    private String encodeState(PacmanGame state) {
        StringBuilder stateBuilder = new StringBuilder();

        for (int i = 0; i < sizeMazeX; i++) {
            for (int j = 0; j < sizeMazeY; j++) {
                char symbol;
                if (state.isWallAtPosition(i, j)) {
                    symbol = '0';
                } else if (state.isCapsuleAtPosition(i, j)) {
                    symbol = '1';
                } else if (state.isPacmanAtPosition(i, j)) {
                    symbol = '2';
                } else if (state.isGhostAtPosition(i, j)) {
                    symbol = '3';
                } else if (state.isGumAtPosition(i, j)) {
                    symbol = '4';
                } else {
                    symbol = '5';
                }
                stateBuilder.append(symbol);
            }
        }

        return stateBuilder.toString();
    }

    /**
     * Chooses an action based on the epsilon-greedy strategy.
     *
     * @param state the current PacmanGame state
     * @return the chosen AgentAction
     */
    @Override
    public AgentAction chooseAction(PacmanGame state) {
        Random random = new Random();

        // Exploration: choose a random legal action with epsilon probability
        if (random.nextDouble() < current_epsilon) {
            ArrayList<AgentAction> legalActions = state.getLegalPacmanActions();
            return legalActions.get(random.nextInt(legalActions.size()));
        } else {
            // Exploitation: choose the action with the highest Q-value
            String currentState = encodeState(state);
            double[] qValues = QTable.getOrDefault(currentState, new double[AgentAction.NUM_ACTIONS]);

            int bestActionIndex = 0;
            for (int i = 1; i < qValues.length; i++) {
                if (qValues[i] > qValues[bestActionIndex]) {
                    bestActionIndex = i;
                }
            }

            return new AgentAction(bestActionIndex);
        }
    }

    /**
     * Updates the Q-value for the chosen action in the current state.
     *
     * @param state the current PacmanGame state
     * @param nextState the next PacmanGame state
     * @param action the chosen AgentAction
     * @param reward the reward received for taking the action
     * @param isFinalState whether the next state is terminal
     */
    @Override
    public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
        String currentState = encodeState(state);
        String nextStateEncoded = encodeState(nextState);

        double[] qValuesCurrent = QTable.getOrDefault(currentState, new double[AgentAction.NUM_ACTIONS]);
        double[] qValuesNext = QTable.getOrDefault(nextStateEncoded, new double[AgentAction.NUM_ACTIONS]);

        double maxQNext = Double.NEGATIVE_INFINITY;
        for (double value : qValuesNext) {
            maxQNext = Math.max(maxQNext, value);
        }

        // Update Q-value using the Bellman equation
        if (isFinalState) {
            maxQNext = 0;
        }

        int actionIndex = action.get_idAction();
        qValuesCurrent[actionIndex] = (1 - learningRate) * qValuesCurrent[actionIndex] + 
                                       learningRate * (reward + gamma * maxQNext);

        QTable.put(currentState, qValuesCurrent);
    }

    /**
     * This method is not used in Tabular Q-Learning, as it leverages a lookup table.
     * 
     * @param trainExamples list of training examples (unused)
     */
    @Override
    public void learn(ArrayList<TrainExample> trainExamples) {
        // Not applicable for Tabular Q-Learning
    }
}
