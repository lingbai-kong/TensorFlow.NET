using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public class Activation : IActivation
    {
        public string Name { get; set; }
        /// <summary>
        /// The parameters are `features` and `name`.
        /// </summary>
        public Func<Tensor, string, Tensor> ActivationFunction { get; set; }

        public Tensor Apply(Tensor input, string name = null) => ActivationFunction(input, name);

        public static implicit operator Activation(Func<Tensor, string, Tensor> func)
        {
            return new Activation()
            {
                Name = func.GetMethodInfo().Name,
                ActivationFunction = func
            };
        }

        public static implicit operator Activation(string name)
        {
            IActivation activation = keras.activations.GetActivationFromName(name);
            return new Activation()
            {
                Name = activation.Name,
                ActivationFunction = activation.ActivationFunction
            }; ;
        }
    }
    public class Activations : IActivationsApi
    {
        private static Dictionary<string, Activation> _nameActivationMap;

        private static Activation _linear = new Activation()
        {
            Name = "linear",
            ActivationFunction = (features, name) => features
        };
        private static Activation _relu = new Activation()
        {
            Name = "relu",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Relu", name, new ExecuteOpArgs(features))
        };
        private static Activation _sigmoid = new Activation()
        {
            Name = "sigmoid",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Sigmoid", name, new ExecuteOpArgs(features))
        };
        private static Activation _softmax = new Activation()
        {
            Name = "softmax",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Softmax", name, new ExecuteOpArgs(features))
        };
        private static Activation _tanh = new Activation()
        {
            Name = "tanh",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(features))
        };
        private static Activation _mish = new Activation()
        {
            Name = "mish",
            ActivationFunction = (features, name) => features * tf.math.tanh(tf.math.softplus(features))
        };

        /// <summary>
        /// Register the name-activation mapping in this static class.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="activation"></param>
        private static void RegisterActivation(Activation activation)
        {
            _nameActivationMap[activation.Name] = activation;
        }

        static Activations()
        {
            _nameActivationMap = new Dictionary<string, Activation>();

            RegisterActivation(_relu);
            RegisterActivation(_linear);
            RegisterActivation(_sigmoid);
            RegisterActivation(_softmax);
            RegisterActivation(_tanh);
            RegisterActivation(_mish);
        }

        public IActivation Linear => _linear;

        public IActivation Relu => _relu;

        public IActivation Sigmoid => _sigmoid;

        public IActivation Softmax => _softmax;

        public IActivation Tanh => _tanh;

        public IActivation Mish => _mish;

        public IActivation GetActivationFromName(string name)
        {
            if (name == null)
            {
                return Linear;
            }
            if (!_nameActivationMap.TryGetValue(name, out var res))
            {
                throw new Exception($"Activation {name} not found");
            }
            else
            {
                return res;
            }
        }
    }
}
