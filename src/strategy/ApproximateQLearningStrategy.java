package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public class ApproximateQLearningStrategy extends QLearningStrategy {
	private double[] weights;
	private Random rand = new Random();

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		int numberOfFeatures = 6;
		this.weights = new double[numberOfFeatures];

		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = rand.nextDouble() * 2 - 1;
		}
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		ArrayList<AgentAction> legalActions = new ArrayList<AgentAction>();
		Maze maze = state.getMaze(); 
		AgentAction actionChoosen = new AgentAction(0); 

		for(int i =0; i < 4; i++) {
			AgentAction action = new AgentAction(i);
			if(!maze.isWall(state.pacman.get_position().getX() + action.get_vx(),
					state.pacman.get_position().getY() + action.get_vy())) {
				legalActions.add(action);
			}
		}

		if(Math.random() < this.current_epsilon){
			actionChoosen = legalActions.get((int) Math.floor(Math.random() * legalActions.size()));
		} else {
			double maxQvalue = -9999;

			int trouve = 1;

			for(AgentAction action : legalActions) {

				double[] features = extractFeatures(state, action);
				double qValue = perceptron(weights, features);

				if(qValue > maxQvalue) {

					maxQvalue = qValue;
					actionChoosen = action;
					trouve = 1;

				} else if(qValue == maxQvalue) {
					trouve += 1;

					if(Math.floor(trouve*Math.random())== 0) {
						maxQvalue = qValue;
						actionChoosen = action;
					}

				}

			}

		}

		return actionChoosen;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {


		double targetQ;

		if(isFinalState) {

			targetQ = reward;

		} else {

			double maxQnext = getMaxQNext(nextState);

			targetQ = reward + this.gamma*maxQnext;

		}

		double[] features = extractFeatures(state, action); 
		double qValue = perceptron(weights, features); 
		for(int i =0; i < this.weights.length; i ++) {
			this.weights[i] = this.weights[i] - 2*this.learningRate*features[i]*(qValue - targetQ);

		}


	}
	public double perceptron(double[] weights, double[] features) {

		double results = 0;

		for(int i =0; i < weights.length; i++) {
			results += weights[i]*features[i];
		}

		return results;
	}

	public double getMaxQNext(PacmanGame game) {
		PositionAgent nextPos = game.pacman._position;
		Maze maze = game.getMaze();
		double maxQvalue = -99999;
		for(int i =0; i < 4; i++) {
			AgentAction action = new AgentAction(i);
			if(!maze.isWall(nextPos.getX() + action.get_vx(),
					nextPos.getY() + action.get_vy())) {

				double[] features = extractFeatures(game, action);
				double qValue = perceptron(weights, features);

				if(qValue > maxQvalue) {

					maxQvalue = qValue;

				}
			}
		}
		return maxQvalue;
	}

	private double[] extractFeatures(PacmanGame state, AgentAction action) {


		double[] features = new double[6];

		features[0] = 1;

		Maze maze = state.getMaze();

		int x = state.pacman._position.getX();
		int y = state.pacman._position.getY();

		int new_x = x + action.get_vx();
		int new_y = y + action.get_vy();

	

		if(maze.isFood(new_x, y + action.get_vy())) {
			features[1] = 1; // Si mange une pacgomme
		}

		if(maze.isCapsule(new_x, new_y)) {
			features[2]=1; 
		}
		if(state.getNbTourInvincible()>1) { 
			features[3]=countGhostsAround(new_x,new_y,state);
		}else{ 
			features[4]=countGhostsAround(new_x,new_y,state);
		}
		features[5]=nbCoupsProchainePacgomme(state,new_x,new_y); 

		return features;

	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
	}
}
