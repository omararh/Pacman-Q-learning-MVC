package strategy.utils;
import agent.AgentAction;


public class Node {

    public int x;
    public int y;

    //public int heuristicCost;
    public int effectiveCost;

    //public int globalCost;

    public int id;

    public Node parent;
    public AgentAction action;

    public Node(Node node, AgentAction action, int x,int y, int effectiveCost) {

        this.parent = node;
        this.action = action;
        this.x = x;
        this.y = y;
        this.effectiveCost = effectiveCost;
        //this.heuristicCost = heuristicCost;

        //this.globalCost = effectiveCost + heuristicCost;
        this.id = x*100+ y;
    }

}
