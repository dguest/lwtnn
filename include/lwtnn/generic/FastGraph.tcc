#include "lwtnn/generic/FastGraph.hh"
#include "lwtnn/generic/FastInputPreprocessor.hh"
#include "lwtnn/generic/Graph.hh"
#include "lwtnn/generic/eigen_typedefs.hh"
#include "lwtnn/InputOrder.hh"

#include <Eigen/Dense>

namespace lwt {
namespace generic {

  namespace internal {
    template<typename T>
    using FIP = FastInputPreprocessor<T>;
    template<typename T>
    using FIVP = FastInputVectorPreprocessor<T>;
    template<typename T>
    using FastPreprocs = std::vector<FIP<T>>;
    template<typename T>
    using FastVecPreprocs = std::vector<FIVP<T>>;
  }

namespace internal {

  // this is used internally to ensure that we only look up map inputs
  // when the network asks for them.
  template<typename T>
  class LazySource: public ISource<T>
  {
  public:
    LazySource(const NodeVec<T>&, const SeqNodeVec<T>&,
               const FastPreprocs<T>&, const FastVecPreprocs<T>&,
               const SourceIndices& input_indices);
    virtual VectorX<T> at(size_t index) const override;
    virtual MatrixX<T> matrix_at(size_t index) const override;
  private:
    const NodeVec<T>& m_nodes;
    const SeqNodeVec<T>& m_seqs;
    const FastPreprocs<T>& m_preprocs;
    const FastVecPreprocs<T>& m_vec_preprocs;
    const SourceIndices& m_input_indices;
  };

  template<typename T>
  LazySource<T>::LazySource(const NodeVec<T>& n,
                            const SeqNodeVec<T>& s,
                            const FastPreprocs<T>& p,
                            const FastVecPreprocs<T>& v,
                            const SourceIndices& i):
    m_nodes(n), m_seqs(s), m_preprocs(p), m_vec_preprocs(v),
    m_input_indices(i)
  {
  }
  template<typename T>
  VectorX<T> LazySource<T>::at(size_t index) const
  {
    const auto& preproc = m_preprocs.at(index);
    size_t source_index = m_input_indices.scalar.at(index);
    if (source_index >= m_nodes.size()) {
      throw NNEvaluationException(
        "The NN needs an input VectorXd at position "
        + std::to_string(source_index) + " but only "
        + std::to_string(m_nodes.size()) + " inputs were given");
    }
    return preproc(m_nodes.at(source_index));
  }
  template<typename T>
  MatrixX<T> LazySource<T>::matrix_at(size_t index) const
  {
    const auto& preproc = m_vec_preprocs.at(index);
    size_t source_index = m_input_indices.sequence.at(index);
    if (source_index >= m_seqs.size()) {
      throw NNEvaluationException(
        "The NN needs an input MatrixXd at position "
        + std::to_string(source_index) + " but only "
        + std::to_string(m_seqs.size()) + " inputs were given");
    }
    return preproc(m_seqs.at(source_index));
  }

  // utility functions
  //
  // Build a mapping from the inputs in the saved network to the
  // inputs that the user is going to hand us.
  std::vector<size_t> get_node_indices(
    const order_t& order,
    const std::vector<lwt::InputNodeConfig>& inputs);

} // namespace internal

  // ______________________________________________________________________
  // Fast Graph

  template<typename T>
  FastGraph<T>::FastGraph(const GraphConfig& config, const InputOrder& order,
                          std::string default_output):
    m_graph(config.nodes, config.layers)
  {
    using namespace internal;

    m_input_indices.scalar = get_node_indices(
      order.scalar, config.inputs);

    m_input_indices.sequence = get_node_indices(
      order.sequence, config.input_sequences);

    for (size_t i = 0; i < config.inputs.size(); i++) {
      const lwt::InputNodeConfig& node = config.inputs.at(i);
      size_t input_node = m_input_indices.scalar.at(i);
      std::vector<std::string> varorder = order.scalar.at(input_node).second;
      m_preprocs.emplace_back(node.variables, varorder);
    }
    for (size_t i = 0; i < config.input_sequences.size(); i++) {
      const lwt::InputNodeConfig& node = config.input_sequences.at(i);
      size_t input_node = m_input_indices.sequence.at(i);
      std::vector<std::string> varorder = order.sequence.at(input_node).second;
      m_vec_preprocs.emplace_back(node.variables, varorder);
    }
    if (default_output.size() > 0) {
      if (!config.outputs.count(default_output)) {
        throw NNConfigurationException("no output node" + default_output);
      }
      m_default_output = config.outputs.at(default_output).node_index;
    } else if (config.outputs.size() == 1) {
      m_default_output = config.outputs.begin()->second.node_index;
    } else {
      throw NNConfigurationException("you must specify a default output");
    }
  }

  template<typename T>
  FastGraph<T>::~FastGraph() {
  }

  template<typename T>
  VectorX<T> FastGraph<T>::compute(const NodeVec<T>& nodes,
                                   const SeqNodeVec<T>& seq) const {
    return compute(nodes, seq, m_default_output);
  }
  template<typename T>
  VectorX<T> FastGraph<T>::compute(const NodeVec<T>& nodes,
                                   const SeqNodeVec<T>& seq,
                                   size_t idx) const {
    using namespace internal;
    LazySource<T> source(nodes, seq, m_preprocs, m_vec_preprocs,
                         m_input_indices);
    return m_graph.compute(source, idx);
  }

} // namespace generic
} // namespace lwt
