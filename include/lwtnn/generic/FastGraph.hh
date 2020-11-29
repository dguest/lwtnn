#ifndef LWTNN_UTILS_FAST_GRAPH_H
#define LWTNN_UTILS_FAST_GRAPH_H

#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

#include <Eigen/Dense>

namespace lwt {

  struct InputOrder;

namespace generic {

  template<typename T> class Graph;
  template<typename T> class FastInputPreprocessor;
  template<typename T> class FastInputVectorPreprocessor;

  template<typename T>
  using NodeVec = std::vector<VectorX<T>>;
  template<typename T>
  using SeqNodeVec = std::vector<MatrixX<T>>;

  namespace internal {
    struct SourceIndices
    {
      std::vector<size_t> scalar;
      std::vector<size_t> sequence;
    };
  }

  // Graph class
  template<typename T>
  class FastGraph
  {
  public:

    // In cases where the graph has multiple outputs, we have to
    // define a "default" output, so that calling "compute" with no
    // output specified doesn't lead to ambiguity.
    FastGraph(const GraphConfig& config, const InputOrder& order,
              std::string default_output = "");

    ~FastGraph();
    FastGraph(FastGraph&) = delete;
    FastGraph& operator=(FastGraph&) = delete;

    // The simpler "compute" function
    VectorX<T> compute(const NodeVec<T>&, const SeqNodeVec<T>& = {}) const;

    // the other "compute" which allows you to select an arbitrary output
    VectorX<T> compute(const NodeVec<T>&, const SeqNodeVec<T>&, size_t) const;

  private:
    typedef FastInputPreprocessor<T> IP;
    typedef FastInputVectorPreprocessor<T> IVP;
    typedef std::vector<IP> Preprocs;
    typedef std::vector<IVP> VecPreprocs;

    Graph<T> m_graph;
    Preprocs m_preprocs;
    VecPreprocs m_vec_preprocs;
    size_t m_default_output;
    // the mapping from a node in the network to a user input node
    internal::SourceIndices m_input_indices;
  };
} // namespace generic
} // namespace lwt

#include "FastGraph.tcc"

#endif
