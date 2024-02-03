describe("Sigmoid Activation  Function", () => {
  it;
  it.each`
    input   | expected
    ${-1.0} | ${0.2689}
    ${0.0}  | ${0.5}
    ${1.0}  | ${0.7311}
  `(
    "Should produce an activation for a given input value",
    (test: { input: number; expected: number }) => {
      const activation = new SigmoidActivationFunction();
      const actual = activation.activationAt(test.input);
      expect(actual).toBeCloseTo(test.expected, 4);
    }
  )
  it;
  it.each`
    input   | expected
    ${-1.0} | ${0.1966}
    ${0.0}  | ${0.25}
    ${1.0}  | ${0.1966}
  `(
    "Should produce a slope for a given input value",
    (test: { input: number; expected: number }) => {
      const activation = new SigmoidActivationFunction();
      const actual = activation.slopeAt(activation.activationAt(test.input));
      expect(actual).toBeCloseTo(test.expected, 4);
    }
  );
});



export interface ActivationFunction {
  activationAt(value: number): number;
  slopeAt(value: number): number;
}

export class SigmoidActivationFunction implements ActivationFunction {
  activationAt(value: number): number {
    return 1 / (1 + Math.exp(-value));
  }
  slopeAt(value: number): number {
    // const f = this.activationAt(value);
    return value * (1 - value);
  }
}
