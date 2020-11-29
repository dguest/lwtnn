#ifndef LWTNN_GENERIC_FAST_INPUT_PREPROCESSOR_TCC
#define LWTNN_GENERIC_FAST_INPUT_PREPROCESSOR_TCC

#include "lwtnn/generic/FastInputPreprocessor.hh"
#include "lwtnn/Exceptions.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

namespace lwt {
namespace generic {
namespace internal {
  std::vector<size_t> get_value_indices(
    const std::vector<std::string>& order,
    const std::vector<lwt::Input>& inputs);
} // end internal namespace

  // ______________________________________________________________________
  // FastInput preprocessors

  // simple feed-forwared version
  template<typename T>
  FastInputPreprocessor<T>::FastInputPreprocessor(
    const std::vector<Input>& inputs,
    const std::vector<std::string>& order):
    m_offsets(inputs.size()),
    m_scales(inputs.size())
  {
    size_t in_num = 0;
    for (const auto& input: inputs) {
      m_offsets(in_num) = input.offset;
      m_scales(in_num) = input.scale;
      in_num++;
    }
    m_indices = internal::get_value_indices(order, inputs);
  }

  template<typename T>
  VectorX<T> FastInputPreprocessor<T>::operator()(const VectorX<T>& in) const {
    VectorX<T> invec(m_indices.size());
    size_t input_number = 0;
    for (size_t index: m_indices) {
      if (static_cast<int>(index) >= in.rows()) {
        throw NNEvaluationException(
          "index " + std::to_string(index) + " is out of range, scalar "
          "input only has " + std::to_string(in.rows()) + " entries");
      }
      invec(input_number) = in(index);
      input_number++;
    }
    return (invec + m_offsets).cwiseProduct(m_scales);
  }


  // Input vector preprocessor
  template<typename T>
  FastInputVectorPreprocessor<T>::FastInputVectorPreprocessor(
    const std::vector<Input>& inputs,
    const std::vector<std::string>& order):
    m_offsets(inputs.size()),
    m_scales(inputs.size())
  {
    size_t in_num = 0;
    for (const auto& input: inputs) {
      m_offsets(in_num) = input.offset;
      m_scales(in_num) = input.scale;
      in_num++;
    }
    // require at least one input at configuration, since we require
    // at least one for evaluation
    if (in_num == 0) {
      throw NNConfigurationException("need at least one input");
    }
    m_indices = internal::get_value_indices(order, inputs);
  }

  template<typename T>
  MatrixX<T> FastInputVectorPreprocessor<T>::operator()(const MatrixX<T>& in)
    const {
    using namespace Eigen;
    size_t n_cols = in.cols();
    MatrixX<T> inmat(m_indices.size(), n_cols);
    size_t in_num = 0;
    for (size_t index: m_indices) {
      if (static_cast<int>(index) >= in.rows()) {
        throw NNEvaluationException(
          "index " + std::to_string(index) + " is out of range, sequence "
          "input only has " + std::to_string(in.rows()) + " entries");
      }
      inmat.row(in_num) = in.row(index);
      in_num++;
    }
    if (n_cols == 0) {
      return MatrixX<T>(m_indices.size(), 0);
    }
    return m_scales.asDiagonal() * (inmat.colwise() + m_offsets);
  }

} // end generic namespace
} // end lwt namespace

#endif
