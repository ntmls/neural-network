import { NetworkConnection } from "./NetworkConnection";
import {
  BiasNode,
  HiddenNode,
  InputNode,
  NetworkNode,
  OutputNode,
} from "./NetworkNode";
import { ReadonlyNetworkNode } from "./ReadonlyNetworkNode";
import {
  ActivationFunction,
  SigmoidActivationFunction,
} from "./SigmoidActivationFunction.spec";

const BIAS_NODE_INDEX = 0;

export class NeuralNetwork {
  private _nodeCount: number;
  private _nodes: NetworkNode[];
  private _inputCount: number = 0;
  private _inputNodes: InputNode[];
  private _outputCount: number = 0;
  private _outputNodes: OutputNode[];
  private _connectionCount: number = 0;
  private _connections: NetworkConnection[];
  private _calculationOrder: NetworkNode[];
  learningRate = 0.05;

  constructor() {
    const biasNode = new BiasNode(0);
    biasNode.label = "Bias";
    this._nodes = [biasNode];
    this._nodeCount = 1; // start with a bias node
    this._connections = [];
    this._outputNodes = [];
    this._inputNodes = [];
    this._calculationOrder = [];
  }

  get inputCount(): number {
    return this._inputCount;
  }

  get inputValues(): number[] {
    return this._inputNodes.map((x) => x.value);
  }

  get outputCount(): number {
    return this._outputCount;
  }

  get outputValues(): number[] {
    return this._outputNodes.map((x) => x.value);
  }

  get nodeCount(): number {
    return this._nodeCount;
  }

  get connectionCount(): number {
    return this._connectionCount;
  }

  get weights(): number[] {
    const result = new Array<number>(this._connectionCount);
    for (let i = 0; i < this._connectionCount; i++) {
      result[i] = this._connections[i].weight;
    }
    return result;
  }

  addInput(label = ""): number {
    const nodeIndex = this._nodeCount;
    this._inputCount += 1;
    this._nodeCount += 1;
    const node = new InputNode(nodeIndex);
    node.label = label;
    this._inputNodes.push(node);
    this._nodes.push(node);
    return nodeIndex;
  }

  addHiddenNode(
    label = "",
    activationFunction: ActivationFunction = new SigmoidActivationFunction()
  ): number {
    const nodeIndex = this._nodeCount;
    this._nodeCount += 1;
    const node = new HiddenNode(nodeIndex, activationFunction);
    node.label = label;
    this._nodes.push(node);
    this.connectNodes(BIAS_NODE_INDEX, nodeIndex);
    return nodeIndex;
  }

  addOutputNode(
    label = "",
    activationFunction: ActivationFunction = new SigmoidActivationFunction()
  ): number {
    const nodeIndex = this._nodeCount;
    this._nodeCount += 1;
    this._outputCount += 1;
    const node = new OutputNode(nodeIndex, activationFunction);
    node.label = label;
    this._nodes.push(node);
    this._outputNodes.push(node);
    this.connectNodes(BIAS_NODE_INDEX, nodeIndex);
    return nodeIndex;
  }

  connectNodes(fromNodeIndex: number, toNodeIndex: number): void {
    this._connectionCount += 1;
    const fromNode = this._nodes[fromNodeIndex];
    const toNode = this._nodes[toNodeIndex];

    if (!fromNode) throw new Error(`Invalkid fromNodeIndex ${fromNodeIndex}`);
    if (!toNode) throw new Error(`Invalkid toNodeIndex ${toNodeIndex}`);

    const connection = new NetworkConnection(fromNode, toNode);
    toNode.appendIncommingConnection(connection);
    fromNode.appendOutgoingConnection(connection);
    this._connections.push(connection);
  }

  initializeWeights(weights: number[]) {
    const result = new Array<number>(this._connectionCount);
    for (let i = 0; i < this._connectionCount; i++) {
      this._connections[i].weight = weights[i];
    }
  }

  nodeAt(nodeIndex: number): ReadonlyNetworkNode {
    return new ReadonlyNetworkNode(this._nodes[nodeIndex]);
  }

  calculate(inputs: number[]): void {
    this.determineCalculateOrder();

    // clear nodes
    for (const node of this._nodes) {
      node.resetForCalculation();
    }

    // set inputs
    for (let i = 0; i < this.inputCount; i++) {
      this._inputNodes[i].initializeValue(inputs[i]);
    }

    // calculate nodes
    for (const node of this._calculationOrder) {
      node.calculateValue();
    }
  }

  //https://youtu.be/-zI1bldB8to

  updateWeights(targetValues: number[]) {
    // set the target value of the outptus
    for (let i = 0; i < this._outputCount; i++) {
      this._outputNodes[i].targetValue = targetValues[i];
    }

    for (let i = this._calculationOrder.length - 1; i >= 0; i--) {
      this._calculationOrder[i].determineSlope();
    }

    for (let i = this._calculationOrder.length - 1; i >= 0; i--) {
      this._calculationOrder[i].updateWeights(this.learningRate);
    }
  }

  private determineCalculateOrder() {
    // reset the order of all nodes
    for (const node of this._nodes) {
      node.resetOrder();
    }

    // start with the inputs and propogate the order down to the outputs
    for (const input of this._inputNodes) {
      input.applyOrder(0);
    }

    this._calculationOrder = this._nodes
      .filter((x) => {
        if (x instanceof HiddenNode) return true;
        if (x instanceof OutputNode) return true;
        return false;
      })
      .sort((x) => x.order);
  }
}
