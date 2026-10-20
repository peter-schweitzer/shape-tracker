#include <stdlib.h>

#include "fmt.h"
#include "num_types.h"
#include "ops.h"
#include "param.h"
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
    print_tensor(a);
    param_t* p_a = make_param_from_tensor(a);
    print_mat(p_a);

    MAKE_TENSOR_FROM_DIMS(b, 4, 2);
    param_t* p_b = make_param_from_tensor(b);
    print_mat(p_b);

    param_t* p_c = op_matmul(p_a, p_b);
    print_mat(p_c);

    destroy_param(p_a);
    destroy_param(p_b);
    destroy_param(p_c);
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

  {
    MAKE_TENSOR_FROM_DIMS(t, 4, 5);
    print_tensor_mat(t);

    tensor_view_t* v = slice(t, I_DIMS(1, 2), I_DIMS(-1, -1), U_DIMS(2, 2));
    print_view_mat(v);

    transpose_view(v);
    print_view_mat(v);

    destroy_tensor(t);
  }
  return 0;
}
