#include "lwtnn/generic/FastInputPreprocessor.hh"

namespace lwt {
namespace generic {
namespace internal {
  std::vector<size_t> get_value_indices(
    const std::vector<std::string>& order,
    const std::vector<lwt::Input>& inputs)
  {
    std::map<std::string, size_t> order_indices;
    for (size_t i = 0; i < order.size(); i++) {
      order_indices[order.at(i)] = i;
    }
    std::vector<size_t> value_indices;
    for (const lwt::Input& input: inputs) {
      if (!order_indices.count(input.name)) {
        throw lwt::NNConfigurationException("Missing input " + input.name);
      }
      value_indices.push_back(order_indices.at(input.name));
    }
    return value_indices;
  }
} // end internal namespace
} // end generic namespace
} // end lwt namespace

