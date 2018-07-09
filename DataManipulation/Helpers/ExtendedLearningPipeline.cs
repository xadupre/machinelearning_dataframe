// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Transforms = Microsoft.ML.Transforms;


namespace Scikit.ML.DataFrame
{
    class GenericScorerPipelineStep : ILearningPipelineDataStep
    {
        public GenericScorerPipelineStep(Var<IDataView> data, Var<ITransformModel> model)
        {
            Data = data;
            Model = model;
        }

        public Var<IDataView> Data { get; }
        public Var<ITransformModel> Model { get; }
    }

    /// <summary>
    /// Extends <see cref="LearningPipeline"/> class to work with IDataView.
    /// </summary>
    /// <example>
    /// <para/>
    /// For example,<para/>
    /// <code>
    /// var pipeline = new ExtendedLearningPipeline();
    /// pipeline.Add(new TextLoader &lt;SentimentData&gt; (dataPath, separator: ","));
    /// pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
    /// pipeline.Add(new FastTreeBinaryClassifier());
    /// 
    /// var model = pipeline.Train&lt;SentimentData, SentimentPrediction&gt;();
    /// </code>
    /// </example>
    public class GenericLearningPipeline : LearningPipeline
    {
        private readonly int? _seed;
        private readonly int _conc;

        /// <summary>
        /// Construct an empty <see cref="ExtendedLearningPipeline"/> object.
        /// </summary>
        public GenericLearningPipeline(int? seed = null, int conc = 0) : base()
        {
            _seed = seed;
            _conc = conc;
        }

        /// <summary>
        /// Train the model using the ML components in the pipeline.
        /// </summary>
        public ExtendedPredictionModel Train()
        {
            using (var environment = new TlcEnvironment(seed: _seed, conc: _conc))
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineStep step = null;
                List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
                List<Var<ITransformModel>> transformModels = new List<Var<ITransformModel>>();
                Var<ITransformModel> lastTransformModel = null;

                foreach (ILearningPipelineItem currentItem in this)
                {
                    if (currentItem is ILearningPipelineLoader loader)
                        loaders.Add(loader);

                    step = currentItem.ApplyStep(step, experiment);
                    if (step is ILearningPipelineDataStep dataStep && dataStep.Model != null)
                        transformModels.Add(dataStep.Model);
                    else if (step is ILearningPipelinePredictorStep predictorDataStep)
                    {
                        if (lastTransformModel != null)
                            transformModels.Insert(0, lastTransformModel);

                        Var<IPredictorModel> predictorModel;
                        if (transformModels.Count != 0)
                        {
                            var localModelInput = new Transforms.ManyHeterogeneousModelCombiner
                            {
                                PredictorModel = predictorDataStep.Model,
                                TransformModels = new ArrayVar<ITransformModel>(transformModels.ToArray())
                            };
                            var localModelOutput = experiment.Add(localModelInput);
                            predictorModel = localModelOutput.PredictorModel;
                        }
                        else
                            predictorModel = predictorDataStep.Model;

                        var scorer = new Transforms.Scorer
                        {
                            PredictorModel = predictorModel
                        };

                        var scorerOutput = experiment.Add(scorer);
                        lastTransformModel = scorerOutput.ScoringTransform;
                        step = new GenericScorerPipelineStep(scorerOutput.ScoredData, scorerOutput.ScoringTransform);
                        transformModels.Clear();
                    }
                }

                if (transformModels.Count > 0)
                {
                    if (lastTransformModel != null)
                        transformModels.Insert(0, lastTransformModel);

                    var modelInput = new Transforms.ModelCombiner
                    {
                        Models = new ArrayVar<ITransformModel>(transformModels.ToArray())
                    };

                    var modelOutput = experiment.Add(modelInput);
                    lastTransformModel = modelOutput.OutputModel;
                }

                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                {
                    loader.SetInput(environment, experiment);
                }
                experiment.Run();

                ITransformModel model = experiment.GetOutput(lastTransformModel);
                using (var memoryStream = new MemoryStream())
                {
                    model.Save(environment, memoryStream);
                    memoryStream.Position = 0;
                    return new ExtendedPredictionModel(memoryStream);
                }
            }
        }

        /// <summary>
        /// Executes a pipeline and returns the resulting data.
        /// </summary>
        /// <returns>
        /// The IDataView that was returned by the pipeline.
        /// </returns>
        internal IDataView Execute(IHostEnvironment environment)
        {
            Experiment experiment = environment.CreateExperiment();
            ILearningPipelineStep step = null;
            List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
            foreach (ILearningPipelineItem currentItem in this)
            {
                if (currentItem is ILearningPipelineLoader loader)
                    loaders.Add(loader);

                step = currentItem.ApplyStep(step, experiment);
            }

            if (!(step is ILearningPipelineDataStep endDataStep))
            {
                throw new InvalidOperationException($"{nameof(LearningPipeline)}.{nameof(Execute)} must have a Data step as the last step.");
            }

            experiment.Compile();
            foreach (ILearningPipelineLoader loader in loaders)
            {
                loader.SetInput(environment, experiment);
            }
            experiment.Run();

            return experiment.GetOutput(endDataStep.Data);
        }
    }

    public class ExtendedPredictionModel
    {
        private readonly TransformModel _predictorModel;
        private readonly IHostEnvironment _env;

        public ExtendedPredictionModel(Stream stream)
        {
            _env = new TlcEnvironment();
            _predictorModel = new TransformModel(_env, stream);
        }

        public TransformModel PredictorModel
        {
            get { return _predictorModel; }
        }

        /// <summary>
        /// Returns labels that correspond to indices of the score array in the case of 
        /// multi-class classification problem.
        /// </summary>
        /// <param name="names">Label to score mapping</param>
        /// <param name="scoreColumnName">Name of the score column</param>
        /// <returns></returns>
        public bool TryGetScoreLabelNames(out string[] names, string scoreColumnName = DefaultColumnNames.Score)
        {
            names = null;
            ISchema schema = _predictorModel.OutputSchema;
            int colIndex;
            if (!schema.TryGetColumnIndex(scoreColumnName, out colIndex))
                return false;

            int expectedLabelCount = schema.GetColumnType(colIndex).ValueCount;
            if (!schema.HasSlotNames(colIndex, expectedLabelCount))
                return false;

            VBuffer<DvText> labels = new VBuffer<DvText>();
            schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colIndex, ref labels);

            if (labels.Length != expectedLabelCount)
                return false;

            names = new string[expectedLabelCount];
            int index = 0;
            foreach (var label in labels.DenseValues())
                names[index++] = label.ToString();

            return true;
        }

        /// <summary>
        /// Read model from file asynchronously.
        /// </summary>
        /// <param name="path">Path to the file</param>
        /// <returns>Model</returns>
        public static Task<ExtendedPredictionModel> ReadAsync(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return ReadAsync(stream);
            }
        }

        /// <summary>
        /// Read model from stream asynchronously.
        /// </summary>
        /// <param name="stream">Stream with model</param>
        /// <returns>Model</returns>
        public static Task<ExtendedPredictionModel> ReadAsync(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            return Task.FromResult(new ExtendedPredictionModel(stream));
        }

        /// <summary>
        /// Run prediction on top of IDataView.
        /// </summary>
        /// <param name="input">Incoming IDataView</param>
        /// <returns>IDataView which contains predictions</returns>
        public IDataView Predict(IDataView input) => _predictorModel.Apply(_env, input);

        /// <summary>
        /// Save model to file.
        /// </summary>
        /// <param name="path">File to save model</param>
        /// <returns></returns>
        public Task WriteAsync(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
            {
                return WriteAsync(stream);
            }
        }

        /// <summary>
        /// Save model to stream.
        /// </summary>
        /// <param name="stream">Stream to save model.</param>
        /// <returns></returns>
        public Task WriteAsync(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            _predictorModel.Save(_env, stream);
            return Task.CompletedTask;
        }
    }
}

