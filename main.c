#include "fmt.h"
#include "ops.h"
#include "tensor.h"
#include "voodoo.h"

// #define PART_1
// #define PART_2
// #define PART_3
// #define PART_4
#define PART_5

int main(void) {
#ifdef PART_1
  {
    tensor_t* t = make_tensor(16);
    print(t);

    op_reshape(t, 2, U_DIMS(4, 4));
    print_mat(t);

    op_reshape(t, 3, U_DIMS(2, 4, 2));
    print(t);

    destroy_tensor(t);
  }
#endif

#ifdef PART_2
  {
    MAKE_TENSOR_FROM_DIMS(a, 3, 4);
    print_mat(a);

    MAKE_TENSOR_FROM_DIMS(b, 4, 2);
    print_mat(b);

    tensor_t* c = op_matmul(a, b);
    print_mat(c);

    destroy_tensor(c);
    destroy_tensor(b);
    destroy_tensor(a);
  }
#endif

#ifdef PART_3
  {
    MAKE_TENSOR_FROM_DIMS(a, 16, 16);
    print_mat(a);

    MAKE_TENSOR_FROM_DIMS(b, 32, 32);
    tensor_t* b_s = op_slice(b, I_DIMS(0), I_DIMS(-1, -1), U_DIMS(2, 2));
    print_mat(b_s);

    destroy_tensor(b_s);
    destroy_tensor(b);
    destroy_tensor(a);
  }
#endif

#ifdef PART_4
  {
    MAKE_TENSOR_FROM_DIMS(t, 5, 5);
    print_mat(t);

    tensor_t* s = op_slice(t, I_DIMS(1, 1), I_DIMS(-2, -2), U_DIMS(2, 1));
    destroy_tensor(t);

    print(s);
    print_mat(s);

    tensor_t* c = make_contiguous(s);
    destroy_tensor(s);

    print(c);

    tensor_t* c_T = op_transpose(c);
    destroy_tensor(c);

    print_mat(c_T);
    destroy_tensor(c_T);
  }
#endif

#ifdef PART_5
  {
    MAKE_TENSOR_FROM_DIMS(t, 5, 5);
    print_mat(t);

    tensor_t* s = op_slice(t, I_DIMS(1, 1), I_DIMS(-2, -2), U_DIMS(1, 2));
    destroy_tensor(t);

    print(s);
    print_mat(s);
  }
#endif

  return 0;
}
