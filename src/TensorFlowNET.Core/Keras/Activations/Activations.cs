using Newtonsoft.Json;
using System.Reflection;
using System.Runtime.Versioning;
using Tensorflow.Keras.Saving.Common;

namespace Tensorflow.Keras
{
    [JsonConverter(typeof(CustomizedActivationJsonConverter))]
    public interface IActivation
    {
        string Name { get; set; }
        /// <summary>
        /// The parameters are `features` and `name`.
        /// </summary>
        Func<Tensor, string, Tensor> ActivationFunction { get; set; }

        Tensor Apply(Tensor input, string name = null);
    }

    public interface IActivationsApi
    {
        IActivation GetActivationFromName(string name);
        
        IActivation Linear { get; }

        IActivation Relu { get; }

        IActivation Sigmoid { get; }

        IActivation Softmax { get; }

        IActivation Tanh { get; }

        IActivation Mish { get; }
    }
}
