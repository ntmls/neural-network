import { NeuralNetwork } from "./NeuralNetwork";
import { TrainingSet } from "./TrainingSet";

describe("NeuralNetwork", () => {
  it("Should create a new neural network.", () => {
    const network = new NeuralNetwork();
    expect(network).toBeDefined();
    expect(network.nodeCount).toBe(1); // 1 because it should always havea bias node.
    expect(network.inputCount).toBe(0);
    expect(network.outputCount).toBe(0);
    expect(network.connectionCount).toBe(0);
    expect(network.outputValues).toStrictEqual([]);
    expect(network.weights).toStrictEqual([]);
  });

  it("Should add an input node.", () => {
    const network = new NeuralNetwork();
    const nodeId = network.addInput();
    expect(nodeId).toBe(1);
    expect(network.nodeCount).toBe(2);
    expect(network.inputCount).toBe(1);
    expect(network.connectionCount).toBe(0);
  });

  it("Should add a hidden node.", () => {
    const network = new NeuralNetwork();
    const nodeId = network.addHiddenNode();
    expect(nodeId).toBe(1);
    expect(network.nodeCount).toBe(2);
    expect(network.connectionCount).toBe(1);
  });

  it("Should add an output node.", () => {
    const network = new NeuralNetwork();
    const nodeId = network.addOutputNode();
    expect(nodeId).toBe(1);
    expect(network.nodeCount).toBe(2);
    expect(network.outputCount).toBe(1);
    expect(network.connectionCount).toBe(1);
  });

  it("Should connect input node to hidden node.", () => {
    const network = new NeuralNetwork();
    const inputNodeId = network.addInput();
    const hiddenNodeId = network.addHiddenNode();
    network.connectNodes(inputNodeId, hiddenNodeId);
    expect(network.connectionCount).toBe(2);
  });

  it("Should connect hidden node to output node.", () => {
    const network = new NeuralNetwork();
    const inputNodeId = network.addHiddenNode();
    const outputNodeId = network.addOutputNode();
    network.connectNodes(inputNodeId, outputNodeId);
    expect(network.connectionCount).toBe(3); // two bias connections and one that we just added.
  });

  it("Should be able to give weights an initiate value.", () => {
    const network = createXorNetwork();
    const weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    network.initializeWeights(weights);
    expect(network.weights).toStrictEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  it("EXAMPLE: Should create XOR network.", () => {
    const network = createXorNetwork();
    expect(network.nodeCount).toBe(5);
    expect(network.connectionCount).toBe(7);
    assertNodeAttributes(network, 0, "Bias", 0, 2, 0, 1);
    assertNodeAttributes(network, 1, "Input 1", 0, 2, 0, 0);
    assertNodeAttributes(network, 2, "Input 2", 0, 2, 0, 0);
    assertNodeAttributes(network, 3, "Hidden", 3, 1, 0, 0);
    assertNodeAttributes(network, 4, "Output", 4, 0, 0, 0);
    expect(network.weights).toStrictEqual([0, 0, 0, 0, 0, 0, 0]);
  });

  it("EXAMPLE: Should calculate XOR network with initial weights.", () => {
    const network = createXorNetwork();
    network.calculate([1, 1]);
    assertNodeAttributes(network, 0, "Bias", 0, 2, 0, 1);
    assertNodeAttributes(network, 1, "Input 1", 0, 2, 0, 1);
    assertNodeAttributes(network, 2, "Input 2", 0, 2, 0, 1);
    assertNodeAttributes(network, 3, "Hidden", 3, 1, 1, 0.5);
    assertNodeAttributes(network, 4, "Output", 4, 0, 2, 0.5);
    expect(network.outputValues).toStrictEqual([0.5]);
  });

  it("EXAMPLE: Should calculate XOR network with initial weights and update weights once.", () => {
    const network = createXorNetwork();
    network.calculate([1, 1]);
    assertNodeAttributes(network, 0, "Bias", 0, 2, 0, 1);
    assertNodeAttributes(network, 1, "Input 1", 0, 2, 0, 1);
    assertNodeAttributes(network, 2, "Input 2", 0, 2, 0, 1);
    assertNodeAttributes(network, 3, "Hidden", 3, 1, 1, 0.5);
    assertNodeAttributes(network, 4, "Output", 4, 0, 2, 0.5);

    network.updateWeights([0]);

    expect(network.nodeAt(4).slope).toBe(0.125);
    expect(network.nodeAt(3).slope).toBe(0);
  });

  it("EXAMPLE: Should calculate XOR network with given weights.", () => {
    const network = createXorNetwork();
    network.initializeWeights([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    network.calculate([1, 1]);
    assertNodeAttributes(network, 0, "Bias", 0, 2, 0, 1);
    assertNodeAttributes(network, 1, "Input 1", 0, 2, 0, 1);
    assertNodeAttributes(network, 2, "Input 2", 0, 2, 0, 1);
    assertNodeAttributes(network, 3, "Hidden", 3, 1, 1, 0.8176);
    assertNodeAttributes(network, 4, "Output", 4, 0, 2, 0.8709);
  });
});

it("EXAMPLE: Should train XOR network given training set.", () => {
  const network = createXorNetwork();
  network.learningRate = 0.1;
  network.initializeWeights([
    -0.555, 0.419, -0.604, -0.628, 0.318, -0.146, 0.045,
  ]);
  /*
  let weights: number[] = [];
  for(let i = 0; i < 7; i++) {
    weights.push(Math.random() * 2 - 1 );
  }
  network.initializeWeights(weights);
  */
  // console.log(network.weights);

  const trainingSet: TrainingSet = {
    items: [
      { inputs: [1, 1], outputs: [0] },
      { inputs: [1, 0], outputs: [1] },
      { inputs: [0, 1], outputs: [1] },
      { inputs: [0, 0], outputs: [0] },
    ],
  };

  // let out = "";
  for (let i = 0; i < 50000; i++) {
    for (const item of trainingSet.items) {
      network.calculate(item.inputs);
      network.updateWeights(item.outputs);
      /*
      const outputs = network.outputValues;
      let sum = 0;
      for (let o = 0; o < network.outputCount; o++) {
        sum += Math.abs(outputs[o] - item.outputs[o]);
      }
      if (i % 1000 === 0) {
        const avergeError = sum / network.outputCount;
        // console.log(network.inputValues);
        // console.log(network.outputValues);
        // console.log(network.weights);
        out += "\n" + avergeError.toString();
      }
      */
    }
  }

  // console.log(out);
  // console.log(network.weights);

  expect(network.weights).toStrictEqual([
    3.0990467601063725, 10.238024812047714, -7.9811239383425905,
    -7.981116868731012, -6.753654183105628, -6.753773972470266,
    -14.655297301364296,
  ]);
});

function createXorNetwork() {
  const network = new NeuralNetwork();
  const inputNodeId1 = network.addInput("Input 1");
  const inputNodeId2 = network.addInput("Input 2");
  const hiddenNodeId = network.addHiddenNode("Hidden");
  const outpuNodeId = network.addOutputNode("Output");
  network.connectNodes(inputNodeId1, hiddenNodeId);
  network.connectNodes(inputNodeId2, hiddenNodeId);
  network.connectNodes(inputNodeId1, outpuNodeId);
  network.connectNodes(inputNodeId2, outpuNodeId);
  network.connectNodes(hiddenNodeId, outpuNodeId);
  return network;
}

function assertNodeAttributes(
  network: NeuralNetwork,
  index: number,
  label: string,
  incommingCount: number,
  outgoingCount: number,
  order: number,
  value: number
) {
  let node = network.nodeAt(index);
  expect(node.label).toBe(label);
  expect(node.incomming.length).toBe(incommingCount);
  expect(node.outgoing.length).toBe(outgoingCount);
  expect(node.order).toBe(order);
  expect(node.value).toBeCloseTo(value, 4);
}
