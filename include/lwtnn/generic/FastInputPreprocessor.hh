#ifndef LWTNN_FAST_INPUT_PREPROCESSOR_HH
#define LWTNN_FAST_INPUT_PREPROCESSOR_HH

#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

#include <Eigen/Dense>
#include <vector>

namespace lwt {

namespace generic {

  // ______________________________________________________________________
  // input preprocessor (handles normalization and packing into Eigen)

  template<typename T>
  class FastInputPreprocessor
  {
  public:
    FastInputPreprocessor(const std::vector<Input>& inputs,
                          const std::vector<std::string>& order);
    VectorX<T> operator()(const VectorX<T>&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<size_t> m_indices;
  };

  template<typename T>
  class FastInputVectorPreprocessor
  {
  public:
    FastInputVectorPreprocessor(const std::vector<Input>& inputs,
                                const std::vector<std::string>& order);
    MatrixX<T> operator()(const MatrixX<T>&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<size_t> m_indices;
  };
}

}

#include "FastInputPreprocessor.tcc"

#endif
