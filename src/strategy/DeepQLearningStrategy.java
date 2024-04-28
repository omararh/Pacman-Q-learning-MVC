package strategy;

import java.util.ArrayList;

import agent.AgentAction;

import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;



public class DeepQLearningStrategy extends QLearningStrategy {



	int nEpochs;
	int batchSize;

	int range;

	NeuralNetWorkDL4J nn;
	int sizeState;

	boolean modeAllMaze;


	public DeepQLearningStrategy(double epsilon, double gamma, double alpha, int range, int nEpochs, int batchSize,
                             int sizeMazeX, int sizeMazeY, boolean modeAllMaze, int nbWalls) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);  // Call to parent constructor

		this.modeAllMaze = modeAllMaze;
		System.out.println("Number of walls: " + nbWalls);  // More descriptive message

		// Calculate state size based on mode
		this.sizeState = calculateStateSize(modeAllMaze, sizeMazeX, sizeMazeY, nbWalls, range);

		System.out.println("Size of neural network input: " + this.sizeState);

		this.nn = new NeuralNetWorkDL4J(alpha, 0, sizeState, 4);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
		this.range = range;
	}

	// Helper method to calculate state size based on mode
	private int calculateStateSize(boolean modeAllMaze, int sizeMazeX, int sizeMazeY, int nbWalls, int range) {
		if (modeAllMaze) {
			return (sizeMazeX * sizeMazeY * 4) - nbWalls;
		} else {
			return range * range * 4;
		}
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		// Get the encoded state representation from the current PacmanGame
		double[] encodedState = getEncodedState(state);
	
		// Predict Q-values for the current state using the neural network
		double[] predictedQValues = this.nn.predict(encodedState);
	
		// Calculate the maximum Q-value for the next state (consider refactoring into a separate method if used elsewhere)
		double maxQValueNextState = calculateMaxQValueNextState(nextState, isFinalState);
	
		// Update the Q-value for the chosen action based on the reward and next state's max Q-value
		predictedQValues[action.get_idAction()] = reward + gamma * maxQValueNextState;
	
		// Create a training example from the current state, predicted Q-values, and updated Q-value for the chosen action
		TrainExample trainExample = new TrainExample(encodedState, predictedQValues);
	
		// Add the training example to the list for later training
		this.trainExamples.add(trainExample);
	}
	
	// Helper method to calculate the maximum Q-value for the next state (optional)
	private double calculateMaxQValueNextState(PacmanGame nextState, boolean isFinalState) {
		if (isFinalState) {
			return 0.0; // Terminal state has zero future reward
		} else {
			return getMaxQNext(nextState);
		}
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

	// Helper method to retrieve legal actions (optional)
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

	// Helper method to find the action with the highest Q-value (optional)
	private AgentAction getActionWithMaxQValue(ArrayList<AgentAction> legalActions, PacmanGame state) {
		double maxQValue = Double.NEGATIVE_INFINITY;  // Use a more appropriate negative infinity constant
		AgentAction chosenAction = null;

		for (AgentAction action : legalActions) {
			double[] encodedState = getEncodedState(state);
			double[] output = this.nn.predict(encodedState);
			if (output[action.get_idAction()] > maxQValue) {
				maxQValue = output[action.get_idAction()];
				chosenAction = action;
			}
		}

		return chosenAction;
	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		nn.fit(trainExamples, nEpochs, batchSize,this.learningRate);

	}

	public double[] getEncodedState(PacmanGame game){
		double[] result=new double[sizeState];
		int iWall=0;
		int iFood=range*range;
		int iGhost=range*range*2;

		int x = game.pacman._position.getX();
		int y = game.pacman._position.getY();
		Maze maze=game.getMaze();

		for(int i=x-(range/2);i<=x+(range/2);i++) {
			for(int j=y-(range/2);j<=x+(range/2);j++) {

				//ajout des murs
				if(maze.isWall(i, j)) {
					result[iWall]=1;
					result[iFood]=0;
					result[iGhost]=0;
				}else{
					result[iWall]=0;

					//ajout des foods & capsules
					double isFood = maze.isFood(i, j)? (double)1 : (double)0;
					result[iFood]=maze.isCapsule(i, j)? -1 : isFood;

					//ajout des fantômes
					boolean ghostSeen=false;
					for(PositionAgent gPos:game.getPostionFantom()) {
						if(gPos.getX()==i && gPos.getY()==j) {
							ghostSeen=true;
							if(game.isGhostsScarred()) result[iGhost]=-1;
							else result[iGhost]=1;
							break;
						}
					}
					if(!ghostSeen)result[iGhost]=0;
				}
				iWall++;
				iFood++;
				iGhost++;
			}
		}
		String affichage=" result :";
		for(int i=0;i<result.length;i++) {
			affichage+=result[i]+" ";
		}
		System.out.println(affichage);
		return result;
	}

	public double getMaxQNext(PacmanGame game){
		double maxQvalue = -99999;
		double[] nextQValue = this.nn.predict(getEncodedState(game));

		for(int i =0; i < 4; i++) {
			if(nextQValue[i]>maxQvalue) maxQvalue=nextQValue[i];
		}

		return maxQvalue;
	}


}