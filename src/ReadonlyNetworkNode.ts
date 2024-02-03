import { NetworkNode } from "./NetworkNode";
import { ReadonlyNodeConneciton } from "./ReadonlyNodeConneciton";


export class ReadonlyNetworkNode {
  constructor(private node: NetworkNode) { }
  get index(): number {
    return this.node.index;
  }
  get label(): string {
    return this.node.label;
  }
  get order(): number {
    return this.node.order;
  }
  get value(): number {
    return this.node.value;
  }
  get slope(): number {
    return this.node.slope
  }
  get incomming(): ReadonlyNodeConneciton[] {
    return this.node.incomming.map(
      (x) => new ReadonlyNodeConneciton(x.weight, x.fromNode.index)
    );
  }
  get outgoing(): ReadonlyNodeConneciton[] {
    return this.node.outgoing.map(
      (x) => new ReadonlyNodeConneciton(x.weight, x.toNode.index)
    );
  }
}
