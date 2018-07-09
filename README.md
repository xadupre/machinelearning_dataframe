# DataFrame for ML.net

### Example 1: inner API

This example relies on the inner API, mostly used
inside components of ML.net.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrame.ReadCsv(iris, sep: '\t',
                           dtypes: new DataKind?[] { DataKind.R4 });

// We add a transform to concatenate two features in one vector columns.
var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);

// We create training data by mapping roles to columns.
var trainingData = env.CreateExamples(conc, "Feature", label: "Label");

// We create a trainer, here a One Versus Rest with a logistic regression as inner model.
var trainer = env.CreateTrainer("ova{p=lr}");

using (var ch = env.Start("test"))
{
    // We train the model.
    var pred = trainer.Train(env, ch, trainingData);

    // We compute the prediction (here with the same training data but it should not be the same).
    var scorer = trainer.GetScorer(pred, trainingData, env, null);

    // We store the predictions on a file.
    DataFrame.ViewToCsv(env, scorer, "iris_predictions.txt");

    // Or we could put the predictions into a dataframe.
    var dfout = DataFrame.ReadView(scorer);

    // And access one value...
    var v = dfout.iloc[0, 7];
    Console.WriteLine("PredictedLabel: {0}", v);
}
```

The current interface of
[DataFrame](https://github.com/xadupre/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs)
is not rich. It will improve in the future.

### Example 2: EntryPoints API

This is the same example based on
[Iris Classification](https://github.com/dotnet/machinelearning-samples/tree/master/samples/getting-started/MulticlassClassification_Iris)
but using the new class DataFrame. It is not necessary anymore
to create a class specific to the data used to train. It is a
little bit less efficient for predictions as two consecutive
calls to method ``Predict`` on generic data requires
some the pipeline to build new iterators at every call.
This extra work can be saved when the prediction instance is known.

```CSharp
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrame.ReadCsv(iris, sep: '\t',
                           dtypes: new DataKind?[] { DataKind.R4 });

var importData = df.EPTextLoader(iris, sep: '\t', header: true);
var learningPipeline = new GenericLearningPipeline();
learningPipeline.Add(importData);
learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
var predictor = learningPipeline.Train();
var predictions = predictor.Predict(df);

var dfout = DataFrame.ReadView(predictions);

// And access one value...
var v = dfout.iloc[0, 7];
Console.WriteLine("{0}: {1}", vdf.Schema.GetColumnName(7), v.iloc[0, 7]);
```

### Example 3: DataFrame in C#

The class ``DataFrame`` replicates some functionalities
datascientist are used to in others languages such as
*Python* or *R*. It is possible to do basic operations
on columns:

```CSharp
var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
var df = DataFrame.ReadStr(text);
df["AA+BB"] = df["AA"] + df["BB"];
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB
0,1,text,1
1,1.1,text2,2.1
```

Or:

```CSharp
df["AA2"] = df["AA"] + 10;
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
1,1.1,text2,2.1,11
```

The next instructions change one value
based on a condition.

```CSharp
df.loc[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed";
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
1,1.1,changed,2.1,11
```

A specific set of columns or rows can be extracted:

```CSharp
var view = df[df.ALL, new [] {"AA", "CC"}];
Console.WriteLine(view.ToString());
```

```
AA,CC
0,text
1,changed
```

The dataframe also allows basic filtering:

```CSharp
var view = df[df["AA"] == 0];
Console.WriteLine(view.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
```
