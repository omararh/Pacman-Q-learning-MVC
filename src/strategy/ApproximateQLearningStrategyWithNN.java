package strategy;

import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;
import neuralNetwork.TrainExample;


import java.util.ArrayList;

public class ApproximateQLearningStrategyWithNN extends QLearningStrategy {

	private NeuralNetWorkDL4J nn;
	private int nEpochs;
	private int batchSize;

	public ApproximateQLearningStrategyWithNN(double epsilon, double gamma, double alpha, int nEpochs, int batchSize, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;


		this.nn=new NeuralNetWorkDL4J(alpha,0,7,1);
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		// Get the legal actions (available moves) for Pacman in the current state
		ArrayList<AgentAction> legalActions = getLegalActions(state);
	
		// Choose an action based on epsilon-greedy strategy
		if (Math.random() < current_epsilon) {
			// Exploration: choose a random legal action
			return legalActions.get((int) Math.floor(Math.random() * legalActions.size()));
		} else {
			// Exploitation: choose the action with the highest Q-value
			return getActionWithMaxQValue(legalActions, state);
		}
	}
	
	// Helper method to retrieve legal actions 
	private ArrayList<AgentAction> getLegalActions(PacmanGame state) {
		ArrayList<AgentAction> legalActions = new ArrayList<>();
		Maze maze = state.getMaze();
	
		for (int i = 0; i < 4; i++) {
			AgentAction action = new AgentAction(i);
			if (!maze.isWall(state.pacman.get_position().getX() + action.get_vx(),
					state.pacman.get_position().getY() + action.get_vy())) {
				legalActions.add(action);
			}
		}
	
		return legalActions;
	}
	
	// Helper method to find the action with the highest Q-value 
	private AgentAction getActionWithMaxQValue(ArrayList<AgentAction> legalActions, PacmanGame state) {
		double maxQValue = Double.NEGATIVE_INFINITY;
		AgentAction chosenAction = null;
	
		for (AgentAction action : legalActions) {
			double[] features = extractFeatures(state, action); // Extract features for the current action
			double qValue = this.nn.predict(features)[0];
	
			if (qValue > maxQValue) {
				maxQValue = qValue;
				chosenAction = action;
			}
		}
	
		return chosenAction;
	}
	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		double[] targetQ=new double[1];

		if(isFinalState) {

			targetQ[0] = reward;

		} else {

			double maxQnext = getMaxQNext(nextState);
			targetQ[0] = reward + this.gamma*maxQnext;

		}

		double[] features = extractFeatures(state, action);
		this.trainExamples.add(new TrainExample(features,targetQ));
	}


	public double getMaxQNext(PacmanGame game) {

		PositionAgent nextPos = game.pacman._position;
		Maze maze = game.getMaze();

		double maxQvalue = -99999;

		for(int i =0; i < 4; i++) {


			AgentAction action = new AgentAction(i);
			if(!maze.isWall(nextPos.getX() + action.get_vx(),
					nextPos.getY() + action.get_vy())) {

				double[] features = extractFeatures( game, action);
				double qValue = this.nn.predict(features)[0];;

				if(qValue > maxQvalue) {

					maxQvalue = qValue;

				}

			}

		}

		return maxQvalue;

	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		if (trainExamples.isEmpty()) {
			System.out.println("No training examples available, skipping learning phase.");
			return;
		}
		nn.fit(trainExamples, nEpochs, batchSize,this.learningRate);
	}


	private double[] extractFeatures(PacmanGame state, AgentAction action) {
		double[] features = new double[7];
		features[0] = 1.0; // Bias term (common practice in neural networks)
	
		int nextX = state.pacman._position.getX() + action.get_vx();
		int nextY = state.pacman._position.getY() + action.get_vy();
		Maze maze = state.getMaze();
	
		// Food feature
		features[1] = isFoodAtPosition(maze, nextX, nextY) ? 1.0 : 0.0;
	
		// Capsule feature
		features[2] = isCapsuleAtPosition(maze, nextX, nextY) ? 1.0 : 0.0;
	
		// Ghost count feature (depending on invincibility)
		features[state.getNbTourInvincible() > 1 ? 3 : 4] = 
				countGhostsAround(nextX, nextY, state);
	
		// Distance to closest food feature
		features[5] = nbCoupsProchainePacgomme(state, nextX, nextY);
	
		// Invincibility duration feature
		features[6] = state.getNbTourInvincible();
	
		return features;
	}
	
	// Helper method to check if there's food at a position (optional)
	private boolean isFoodAtPosition(Maze maze, int x, int y) {
		return maze.isFood(x, y);
	}
	
	// Helper method to check if there's a capsule at a position (optional)
	private boolean isCapsuleAtPosition(Maze maze, int x, int y) {
		return maze.isCapsule(x, y);
	}
	
}
