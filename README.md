Lightweight Trained Neural Network
==================================

[![Build Status][build-img]][build-link] [![Scan Status][scan-img]][scan-link]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597221.svg)](https://doi.org/10.5281/zenodo.597221)

[build-img]: https://travis-ci.org/lwtnn/lwtnn.svg?branch=master
[build-link]: https://travis-ci.org/lwtnn/lwtnn
[scan-img]: https://scan.coverity.com/projects/9285/badge.svg
[scan-link]: https://scan.coverity.com/projects/lwtnn-lwtnn

What is this?
-------------

The code comes in two parts:

 1. A set of scripts to convert saved neural networks to a JSON format
 2. A set of classes which are used to reconstruct a neural network in a C++ framework from the JSON format.

The main design principles are:

 - **Minimal dependencies:** The core class only depends on
   C++11 and [Eigen][eg]. The JSON parser to read in NNs also requires
   boost [PropertyTree][pt]. The converter requires Python3 and `h5py`[^1].
 - **Easy to extend:** Should cover 95% of deep network architectures we
   would realistically consider.
 - **Hard to break:** The NN constructor checks the input NN for
   consistency and fails loudly if anything goes wrong.

We also include converters from several popular formats to the `lwtnn`
JSON format. Currently the following formats are supported:
 - [AGILEPack][ap]
 - [Keras][kr] (most popular, see below)

[^1]: Conversion requiring `h5py` and `python3` can be done in any offline environment.

[eg]: http://eigen.tuxfamily.org
[pt]: http://www.boost.org/doc/libs/1_59_0/doc/html/property_tree.html
[ap]: https://github.com/lukedeo/AGILEPack
[kr]: http://keras.io/

Quick Start With Model API
--------------------------

The following instructions apply to the model/functional API in Keras. To see the instructions relevant to the sequential API, go to [Quick Start With sequential API][seqQuickStart].

After running `make`, there are some required steps:

 1. Save your network output file

    Make sure you saved your architecture and weights file from Keras,
    and created your input variable file. See
    [the lwtnn Keras Converter wiki page][weightsInputs] for the
    _correct_ procedure in doing all of this[^2].

    Then
    ```
    lwtnn/converters/kerasfunc2json.py architecture.json weights.h5 inputs.json > neural_net.json
    ```

[^2]: Helpful hint: if you do `lwtnn/converters/kerasfunc2json.py
    architecture.json weights.h5` it creates a skeleton of an input
    file for you, which can be used in the above command!

 2. Test your saved output file

    A good idea is to test your converted network:
    ```
    ./lwtnn-test-lightweight-graph neural_net.json
    ```

    A basic regression test is performed with a bunch of random
    numbers. This test just ensures that lwtnn can in fact read your NN.

[weightsInputs]: https://github.com/lwtnn/lwtnn/wiki/Keras-Converter
[seqQuickStart]: https://github.com/lwtnn/lwtnn/wiki/Quick-Start-With-Sequential-API

 3. Apply your saved neural network within C++ code

    ```C++
// Include several headers. See the files for more documentation.
// First include the class that does the computation
#include "lwtnn/LightweightGraph.hh"
// Then include the json parsing functions
#include "lwtnn/parse_json.hh"

...

// get your saved JSON file as an std::istream object
std::ifstream input("path-to-file.json");
// build the graph
LightweightGraph graph(parse_json_graph(input));

...

// fill a map of input nodes
std::map<std::string, std::map<std::string, double> > inputs;
inputs["input_node"] = {{"value", value}, {"value_2", value_2}};
inputs["another_input_node"] = {{"another_value", another_value}};
// compute the output values
std::map<std::string, double> outputs = graph.compute(inputs);
```

After the constructor for the class `LightweightNeuralNetwork` is constructed,
it has one method, `compute`, which
takes a `map<string, double>` as an input and returns a `map` of named
outputs (of the same type). It's fine to give `compute` a map with
more arguments than the NN requires, but if some argument is _missing_
it will throw an `NNEvaluationException`.

All inputs and outputs are stored in `std::map`s to prevent bugs with
incorrectly ordered inputs and outputs. The strings used as keys in
the map are specified by the network configuration.


### Supported Layers ###

In particular, the following layers are supported as implemented in the
Keras sequential and functional models:

|               | K sequential | K functional  |
|---------------|--------------|---------------|
| Dense         |  yes         |  yes          |
| Normalization | See Note 1   | See Note 1    |
| Maxout        |  yes         |  yes          |
| Highway       |  yes         |  yes          |
| LSTM          |  yes         |  yes          |
| GRU           |  yes         |  yes          |
| Embedding     | sorta        | [issue][ghie] |
| Concatenate   |  no          |  yes          |

**Note 1:** Normalization layers (i.e. Batch Normalization) are only
supported for Keras 1.0.8 and higher.

[ghie]: https://github.com/lwtnn/lwtnn/issues/39
[ghkeras2]: https://github.com/lwtnn/lwtnn/issues/40

#### Supported Activation Functions ####

| Function      | Implemented? |
|---------------|--------------|
| ReLU          | Yes          |
| Sigmoid       | Yes          |
| Hard Sigmoid  | Yes          |
| Tanh          | Yes          |
| Softmax       | Yes          |
| ELU           | No           |

The converter scripts can be found in `converters/`. Run them with
`-h` for more information.


Have problems?
--------------

For more in-depth documentation please see the [`lwtnn` wiki][lwtnnwiki].

If you find a bug in this code, or have any ideas, criticisms,
etc, please email me at `dguest@cern.ch`.

[lwtnnwiki]: https://github.com/lwtnn/lwtnn/wiki
