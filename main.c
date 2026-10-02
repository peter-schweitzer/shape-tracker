#include <stdlib.h>

#include "fmt.h"
#include "num_types.h"
#include "tensor.h"

int main(void) {
  tensor_t* t = make_tensor(16);
  print_tensor(t);

  reshape(t, 2, DIMS(4, 4));
  print_mat(t);

  reshape(t, 3, DIMS(2, 4, 2));
  print_tensor(t);

  destroy_tensor(t);

  /* ======================================================================== */

  MAKE_TENSOR_FROM_DIMS(a, 3, 4);
  print_mat(a);

  MAKE_TENSOR_FROM_DIMS(b, 4, 2)
  print_mat(b);

  tensor_t* c = mat_mul(a, b);
  print_mat(c);

  destroy_tensor(a);
  destroy_tensor(b);
  destroy_tensor(c);

  return 0;
}
