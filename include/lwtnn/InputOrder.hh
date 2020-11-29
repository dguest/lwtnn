#ifndef LWTNN_INPUT_ORDER_H
#define LWTNN_INPUT_ORDER_H

#include <vector>
#include <string>

namespace lwt {

  // the user should specify what inputs they are going to feed to
  // the network. This is different from the ordering that the
  // network uses internally: some variables that are passed in
  // might be dropped or reorganized.
  typedef std::vector<
    std::pair<std::string, std::vector<std::string>>
    > order_t;

  struct InputOrder
  {
    order_t scalar;
    order_t sequence;
  };

}

#endif
