#include <stdlib.h>

#include "fmt.h"
#include "num_types.h"
#include "tensor.h"
#include "tensor_view.h"

int main(void) {
  {
    tensor_t* t = make_tensor(16);
    print_tensor(t);

    reshape_tensor(t, 2, U_DIMS(4, 4));
    print_tensor_mat(t);

    reshape_tensor(t, 3, U_DIMS(2, 4, 2));
    print_tensor(t);

    destroy_tensor(t);
  }

  {
    MAKE_TENSOR_FROM_DIMS(a, 3, 4);
    print_tensor_mat(a);

    MAKE_TENSOR_FROM_DIMS(b, 4, 2)
    print_tensor_mat(b);

    tensor_t* c = mat_mul(a, b);
    print_tensor_mat(c);

    destroy_tensor(a);
    destroy_tensor(b);
    destroy_tensor(c);
  }

  {
    MAKE_TENSOR_FROM_DIMS(a, 16, 16);
    print_tensor_mat(a);

    MAKE_TENSOR_FROM_DIMS(b, 32, 32);
    tensor_view_t* b_v = slice(b, I_DIMS(0), I_DIMS(-1, -1), U_DIMS(2, 2));
    print_view_mat(b_v);

    destroy_tensor(a);

    destroy_view(b_v);
    destroy_tensor(b);
  }

  return 0;
}
