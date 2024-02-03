import { NetworkConnection } from "./NetworkConnection";
import { ActivationFunction } from "./SigmoidActivationFunction.spec";

export abstract class NetworkNode {
  label = "";
  slope = 0;
  protected _value = 0;
  protected _order = 0;
  protected _incomming: NetworkConnection[];
  protected _outgoing: NetworkConnection[];

  constructor(readonly index: number) {
    this._incomming = [];
    this._outgoing = [];
  }

  abstract calculateValue(): void;
  abstract determineSlope(): void;
  abstract resetForCalculation(): void;

  get value(): number {
    return this._value;
  }

  get incomming(): readonly NetworkConnection[] {
    return this._incomming;
  }

  get outgoing(): readonly NetworkConnection[] {
    return this._outgoing;
  }

  get order(): number {
    return this._order;
  }

  resetOrder() {
    this._order = 0;
  }

  applyOrder(order: number) {
    if (order > this._order || order === 0) {
      this._order = order;
      for (const out of this._outgoing) {
        out.toNode.applyOrder(this._order + 1);
      }
    }
  }

  appendOutgoingConnection(connection: NetworkConnection) {
    this._outgoing.push(connection);
  }

  appendIncommingConnection(connection: NetworkConnection) {
    this._incomming.push(connection);
  }

  updateWeights(learningRate: number): void {
    for (const connection of this._incomming) {
      connection.weight -= learningRate * this.slope * connection.fromNode.value;
    }
  }
}

export class BiasNode extends NetworkNode {
  resetForCalculation(): void {}
  determineSlope(): void {
    throw new Error("Method not implemented.");
  }

  calculateValue(): void {}

  constructor(index: number) {
    super(index);
    this._value = 1.0;
  }
}

export class InputNode extends NetworkNode {
  resetForCalculation(): void {}
  determineSlope(): void {
    throw new Error("Method not implemented.");
  }
  calculateValue(): void {}

  initializeValue(imputValue: number) {
    this._value = imputValue;
  }
}

export class HiddenNode extends NetworkNode {
  constructor(index: number, readonly activation: ActivationFunction) {
    super(index);
  }
  calculateValue(): void {
    for (const connection of this._incomming) {
      this._value += connection.calculateValue();
    }
    this._value = this.activation.activationAt(this._value);
  }

  determineSlope(): void {
    let sum = 0;
    for (const connection of this._outgoing) {
      sum += connection.toNode.slope * connection.weight;
    }
    this.slope = this.activation.slopeAt(this.value) * sum;
  }

  resetForCalculation(): void {
    this.slope = 0;
    this._value = 0;
  }
}

export class OutputNode extends NetworkNode {
  targetValue: number = 0;
  private _costFunction: CostFunction;
  constructor(index: number, readonly activation: ActivationFunction) {
    super(index);
    this._costFunction = new CostFunction();
  }
  calculateValue(): void {
    for (const connection of this._incomming) {
      this._value += connection.calculateValue();
    }
    this._value = this.activation.activationAt(this._value);
  }

  determineSlope(): void {
    const costSlope = this._costFunction.slope(this._value, this.targetValue);
    this.slope = costSlope * this.activation.slopeAt(this._value);
  }

  resetForCalculation(): void {
    this.slope = 0;
    this._value = 0;
  }
}

export class CostFunction {
  slope(predicted: number, actual: number) {
    return predicted - actual; // half scquared error (squared error derivative is 2(v - t))
  }
}
