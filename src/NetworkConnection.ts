import { NetworkNode } from "./NetworkNode";


export class NetworkConnection {
  constructor(private _fromNode: NetworkNode, private _toNode: NetworkNode) { }
  public weight: number = 0;
  get fromNode(): NetworkNode {
    return this._fromNode;
  }
  get toNode(): NetworkNode {
    return this._toNode;
  }
  calculateValue(): number {
    return this.fromNode.value * this.weight;
  }
}
