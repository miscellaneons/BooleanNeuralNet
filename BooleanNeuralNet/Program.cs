
using BooleanNeuralNet.Math;

// Test backpropogation

Vector input = new Vector() {1, 2, 3};
Matrix matrix = Matrix.Random(3, BooleanEnumeration.Or);
//Matrix matrix = Matrix.Identity(3, BooleanEnumeration.Or, 2.0);
//Matrix matrix = new Matrix(BooleanEnumeration.And) { new Vector() { 1.0, 1.0, 1.0 }, new Vector() { 1.0, 1.0, 1.0 }, new Vector() { 1.0, 1.0, 1.0 } };

var product = matrix * input;

for (int i = 0; i < 3000; i++)
{
	matrix.Learn(product, input, .01);
	product = matrix * input;
}

product = matrix * input;

for (int i = 0; i < product.Count; i++)
{
	Console.WriteLine(product[i]);
}

Console.WriteLine("Finished");

