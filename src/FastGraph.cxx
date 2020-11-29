#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/InputOrder.hh"
#include "lwtnn/Exceptions.hh"

namespace lwt {
namespace generic {
namespace internal {

  // utility functions
  //
  // Build a mapping from the inputs in the saved network to the
  // inputs that the user is going to hand us.
  std::vector<size_t> get_node_indices(
    const order_t& order,
    const std::vector<lwt::InputNodeConfig>& inputs)
  {
    std::map<std::string, size_t> order_indices;
    for (size_t i = 0; i < order.size(); i++) {
      order_indices[order.at(i).first] = i;
    }
    std::vector<size_t> node_indices;
    for (const lwt::InputNodeConfig& input: inputs) {
      if (!order_indices.count(input.name)) {
        throw NNConfigurationException("Missing input " + input.name);
      }
      node_indices.push_back(order_indices.at(input.name));
    }
    return node_indices;
  }
}
}
}
