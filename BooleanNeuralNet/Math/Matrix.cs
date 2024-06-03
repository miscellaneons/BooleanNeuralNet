using math = System.Math;

namespace BooleanNeuralNet.Math
{
	public class Matrix : List<Vector>
	{

		/// <summary>
		/// do boolean operator instead of matrix multiplication
		/// </summary>
		public Boolean IsBoolean { get; set; } = true;

		/// <summary>
		/// boolean operator to perform <see cref="BooleanEnumeration"/>
		/// </summary>
		public Double Maximum { get; set; } = 1.0;

		public Matrix(BooleanEnumeration? booleanEnumeration = null, double max = 1.0)
		{
			if (booleanEnumeration == null)
				IsBoolean = false;
			else
				Maximum = 0.5 * max * (int)booleanEnumeration;
		}

		public static Matrix Identity(int size, BooleanEnumeration? booleanEnumeration = null, double max = 1.0)
		{
			Matrix matrix = new Matrix(booleanEnumeration, max);
			for (int i = 0; i < size; i++)
			{
				Vector row = new Vector();
				for (int j = 0; j < size; j++)
					row.Add(i == j ? 1.0 : 0.0);
				matrix.Add(row);
			}
			return matrix;
		}

		public static Matrix Random(int size, BooleanEnumeration? booleanEnumeration = null, double max = 1.0)
		{
			Random random = new Random();
			Matrix matrix = new Matrix(booleanEnumeration, max);
			for (int i = 0; i < size; i++)
			{
				Vector row = new Vector();
				for (int j = 0; j < size; j++)
					row.Add(random.NextDouble());
				matrix.Add(row);
			}
			return matrix;
		}

		/// <summary>
		/// backpropogation
		/// </summary>
		/// <param name="actualOut"> output of the neural network </param>
		/// <param name="expectedOut"> the desired output </param>
		/// <param name="learningRate"> a scaling constant </param>
		public void Learn(Vector actualOut, Vector expectedOut, double learningRate = 0.1)
		{
			for (int i = 0; i < this.Count; i++)
			{
				double error = actualOut[i] - expectedOut[i];
				var column = this[i];
				for (int j = 0; j < column.Count; j++)
				{
					column[j] -= learningRate * error * math.Sign(column[j]);
				}
			}
		}

		/// <summary>
		/// do multiplication or boolean operation
		/// </summary>
		/// <param name="layer"> the layer </param>
		/// <param name="inputVector"> the input vector </param>
		/// <returns> the output vector </returns>
		public static Vector operator *(Matrix layer, Vector inputVector)
		{
			Vector product = new Vector();

			if (layer is null || inputVector is null)
				return product;
			
			for (int i = 0; i < layer.Count; i++)
			{
				var column = layer[i];
				Double value = layer.IsBoolean ? 1.0 : 0.0;
				Double scale = 1.0;
				

				for (int j = 0; j < column.Count; j++)
				{
					Double item = column[j];
					if (layer.IsBoolean)
					{
						Double derivative = value;
						if (i != j)
							scale *= layer.Maximum;
						value *= layer.Maximum - inputVector[j] * item;
						for (int k = 0; k < column.Count; k++) 
							if (k != j)
							    derivative *= layer.Maximum - inputVector[k] * item;
					}
					else
						value += inputVector[j] * item;
				}

				if (layer.IsBoolean)
				{
					value = (scale - value) / (scale > 0.0 ? scale : 1.0);

					value = math.Abs(value);
				}

				product.Add(value);
			}

			return product;
		}
	}
}
