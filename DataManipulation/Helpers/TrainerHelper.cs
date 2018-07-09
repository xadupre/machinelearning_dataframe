// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Calibration;


namespace Scikit.ML.DataFrame
{
    /// <summary>
    /// Extended interface for trainers.
    /// </summary>
    public interface ITrainerExtended
    {
        /// <summary>
        /// Returns the inner trainer.
        /// </summary>
        ITrainer Trainer { get; }

        /// <summary>
        /// Returns the loading name of the trainer.
        /// </summary>
        string LoadName { get; }

        /// <summary>
        /// Trains a model.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="ch">channel</param>
        /// <param name="data">traing data</param>
        /// <param name="validData">validation data</param>
        /// <param name="calibrator">calibrator</param>
        /// <param name="maxCalibrationExamples">number of examples used to calibrate</param>
        /// <param name="cacheData">cache training data</param>
        /// <param name="inpPredictor">for continuous training, initial state</param>
        /// <returns>predictor</returns>
        IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, RoleMappedData validData = null,
                                SubComponent<ICalibratorTrainer, SignatureCalibrator> calibrator = null, int maxCalibrationExamples = 0,
                                bool? cacheData = null, IPredictor inpPredictor = null);

        /// <summary>
        /// Creates a scorer to compute the predictions of a trainer.
        /// </summary>
        /// <param name="predictor">predictor</param>
        /// <param name="data">data</param>
        /// <param name="env">host</param>
        /// <param name="trainSchema">training schema</param>
        /// <returns>scorer</returns>
        IDataScorerTransform GetScorer(IPredictor predictor, RoleMappedData data, IHostEnvironment env, RoleMappedSchema trainSchema = null);
    }

    /// <summary>
    /// Wrapper for a trainer with
    /// extra functionalities.
    /// </summary>
    public class ExtendedTrainer : ITrainerExtended, ITrainer
    {
        ITrainer _trainer;
        string _loadName;

        #region ITrainer API

        public string LoadName => _loadName;
        public ITrainer Trainer => _trainer;
        public PredictionKind PredictionKind => _trainer.PredictionKind;
        public IPredictor CreatePredictor() { return _trainer.CreatePredictor(); }

        #endregion

        public ExtendedTrainer(ITrainer trainer, string loadName)
        {
            _loadName = loadName;
            _trainer = trainer;
        }

        /// <summary>
        /// Create a trainer.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="settings">trainer description as a string such <pre>ova{p=lr}</pre></param>
        /// <param name="extraArgs">additional arguments</param>
        public static ITrainerExtended CreateTrainer(IHostEnvironment env, string settings, params object[] extraArgs)
        {
            var sc = SubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            var inst = sc.CreateInstance(env, extraArgs);
            return new ExtendedTrainer(inst, sc.Kind);
        }

        /// <summary>
        /// Trains a model.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="ch">channel</param>
        /// <param name="data">traing data</param>
        /// <param name="validData">validation data</param>
        /// <param name="calibrator">calibrator</param>
        /// <param name="maxCalibrationExamples">number of examples used to calibrate</param>
        /// <param name="cacheData">cache training data</param>
        /// <param name="inpPredictor">for continuous training, initial state</param>
        /// <returns>predictor</returns>
        public IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, RoleMappedData validData = null,
                                SubComponent<ICalibratorTrainer, SignatureCalibrator> calibrator = null, int maxCalibrationExamples = 0,
                                bool? cacheData = null, IPredictor inpPredictor = null)
        {
            return TrainUtils.Train(env, ch, data, Trainer, LoadName, validData, calibrator, maxCalibrationExamples,
                                    cacheData, inpPredictor);
        }

        /// <summary>
        /// Creates a scorer to compute the predictions of a trainer.
        /// </summary>
        /// <param name="predictor">predictor</param>
        /// <param name="data">data</param>
        /// <param name="env">host</param>
        /// <param name="trainSchema">training schema</param>
        /// <returns>scorer</returns>
        public IDataScorerTransform GetScorer(IPredictor predictor, RoleMappedData data, IHostEnvironment env, RoleMappedSchema trainSchema = null)
        {
            return ScoreUtils.GetScorer(predictor, data, env, trainSchema);
        }
    }

    public static class TrainerHelper
    {
        /// <summary>
        /// Create a trainer.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="settings">trainer description as a string such <pre>ova{p=lr}</pre></param>
        /// <param name="extraArgs">additional arguments</param>
        public static ITrainerExtended CreateTrainer(this IHostEnvironment env, string settings, params object[] extraArgs)
        {
            return ExtendedTrainer.CreateTrainer(env, settings, extraArgs);
        }
    }
}
