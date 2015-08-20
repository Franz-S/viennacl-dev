#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/linalg/sum.hpp"

#include <iomanip>
#include <stdlib.h>
#include <fstream>
#define FILENAME "file.txt"


int cmp(const void *x, const void *y)//the comparing for the sort func
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

template<class T, class F>
void init_random(viennacl::matrix<T, F> & M)
{
    std::vector<T> cM(M.internal_size());
    for (std::size_t i = 0; i < M.size1(); ++i)
        for (std::size_t j = 0; j < M.size2(); ++j)
            cM[F::mem_index(i, j, M.internal_size1(), M.internal_size2())] = T(rand())/T(RAND_MAX);
    viennacl::fast_copy(&cM[0],&cM[0] + cM.size(),M);
}

template<class T>
void init_random(viennacl::vector<T> & x)
{
    std::vector<T> cx(x.internal_size());
    for (std::size_t i = 0; i < cx.size(); ++i)
        cx[i] = T(rand())/T(RAND_MAX);
    viennacl::fast_copy(&cx[0], &cx[0] + cx.size(), x.begin());
}

template<class T>
void bench(size_t BLAS_N, std::string const & prefix,int bereich)
{
    std::fstream output;

    using viennacl::linalg::inner_prod;
    using viennacl::linalg::norm_1;
    using viennacl::linalg::norm_2;
    using viennacl::linalg::norm_inf;
    using viennacl::linalg::index_norm_inf;
    using viennacl::linalg::max;
    using viennacl::linalg::min;
    using viennacl::linalg::sum;
    using viennacl::linalg::plane_rotation;
    using viennacl::scalar_vector;
    using viennacl::linalg::element_fabs;
    using viennacl::linalg::inclusive_scan;
    using viennacl::linalg::exclusive_scan;
    using viennacl::scalar_matrix;
    using viennacl::linalg::matrix_diagonal_assign;
    using viennacl::linalg::matrix_diag_from_vector;
    using viennacl::linalg::matrix_diag_to_vector;
    using viennacl::linalg::prod;
    using viennacl::linalg::outer_prod;
    using viennacl::identity_matrix;
    using viennacl::diag;

#define A_SIZE 40//the number of iterations for the function

    viennacl::tools::timer timer ;
    double time_spent_a[A_SIZE];
    double time_spent=0;
    output.open(FILENAME,std::ios::in|std::ios::out|std::ios::ate);

#define BENCHMARK_OP(OPERATION, NAME, PERF, INDEX) \
  if(BLAS_N==0)\
  {\
    output << std::left << std::setw(11) << prefix + NAME <<",";\
  }\
  else{\
  OPERATION; \
  viennacl::backend::finish();\
  timer.start(); \
  for(int i=0;i<A_SIZE;++i) \
  { \
    time_spent_a[i] = timer.get(); \
    OPERATION; \
    viennacl::backend::finish(); \
    time_spent_a[i] = timer.get()-time_spent_a[i]; \
  } \
  qsort(time_spent_a, sizeof(time_spent_a)/sizeof(time_spent_a[0]), sizeof(time_spent_a[0]), cmp);\
  time_spent=time_spent_a[(long)A_SIZE/3]; \
  output << std::left << std::setw(11) << PERF<<" " ; \
  }

#define MATRIX_OP \
        BENCHMARK_OP(A=trans(B),                             "trans",       std::setprecision(3) << double(2*(BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A=B*b,                                  "A=B*b",       std::setprecision(3) << double((BLAS_N+2*BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A=B*b+C*c,                              "A=B*b+C*c",   std::setprecision(3) << double((2*BLAS_N+3*BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A+=B*b+C*c,                             "A+=B*b+C*c",  std::setprecision(3) << double((2*BLAS_N+4*BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A = scalar_matrix<T>(BLAS_N,BLAS_N,a),  "A[i][i]=a",   std::setprecision(3) << double((BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A=identity_matrix<T>(BLAS_N),           "A.diag=1",    std::setprecision(3) << double((BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A=diag(x,0),                            "A.diag=x",    std::setprecision(3) << double((BLAS_N+BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(x=diag(A),                              "x=A.diag",    std::setprecision(3) << double((BLAS_N+BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A=element_fabs(B),                      "A=fabs(B)",   std::setprecision(3) << double(2*(BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(y=prod(A,x),                            "y=prod(A.x)", std::setprecision(3) << double((2*BLAS_N+BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\
        BENCHMARK_OP(A+= a * outer_prod(x,y),                "A.rank1",     std::setprecision(3) << double((2*BLAS_N+2*BLAS_N*BLAS_N)*sizeof(T))/time_spent * 1e-9, "GB/s")\

    switch(bereich)
    {
    case 0://For Testing
    {
        T a = (T)2.4;
        viennacl::vector<T> x(BLAS_N);
        viennacl::vector<T> y(BLAS_N);
        viennacl::vector<T> z(BLAS_N);
        BENCHMARK_OP(a=max(x),                       "max",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a=min(x),                       "min",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a=sum(x),                       "sum",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(inclusive_scan(x,y),            "in.scan", std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(exclusive_scan(x,y),            "ex.scan", std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        init_random(x);
        init_random(y);
        init_random(z);
        break;
    }
    case 1:
    {
        //Vector vector
        viennacl::scalar<T> s(0);
        T a = (T)2.4;
        T b = (T)3.8;

        long j;

        viennacl::vector<T> x(BLAS_N);
        viennacl::vector<T> y(BLAS_N);
        viennacl::vector<T> z(BLAS_N);

        //viennacl::vector_range<viennacl::vector<T> > x_small(x, viennacl::range(1, 10)); how to make a ranged vector

        init_random(x);
        init_random(y);
        init_random(z);

        BENCHMARK_OP(y = a*x,                        "y=a*x",   std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(y = x,                          "y=x",     std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(y = a * x+ b * z,               "y=ax+bz", std::setprecision(3) << double(3*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(y += a * x+ b * z,              "y+=ax+bz",std::setprecision(3) << double(4*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(y = scalar_vector<T>(BLAS_N,a), "y[i]=a",  std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(swap(x,y),                      "swap",    std::setprecision(3) << double(4*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(y=element_fabs(x),              "fabs(x)", std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(s=inner_prod(x,y),              "dot",     std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a = norm_1(x),                  "L1",      std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a = norm_2(x),                  "L2",      std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a = norm_inf(x),                "Linf",    std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(j=index_norm_inf(x),            "index.L", std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a=max(x),                       "max",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a=min(x),                       "min",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(a=sum(x),                       "sum",     std::setprecision(3) << double(1*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(plane_rotation(x, y, a, b),     "pla.rot", std::setprecision(3) << double(4*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(inclusive_scan(x,y),            "in.scan", std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        BENCHMARK_OP(exclusive_scan(x,y),            "ex.scan", std::setprecision(3) << double(2*BLAS_N*sizeof(T))/time_spent * 1e-9, "GB/s")
        j++;
        break;
    }
    case 2://Matrix vector column mayor
    {

        viennacl::matrix<T,viennacl::column_major> A(BLAS_N, BLAS_N);
        viennacl::matrix<T,viennacl::column_major> B(BLAS_N, BLAS_N);
        viennacl::matrix<T,viennacl::column_major> C(BLAS_N, BLAS_N);
        viennacl::vector<T> x(BLAS_N);
        viennacl::vector<T> y(BLAS_N);
        T a = (T)2.4;
        T b = (T)3.8;
        T c = (T)3.8;

        init_random(A);
        init_random(B);
        init_random(C);
        init_random(x);
        init_random(y);
        MATRIX_OP
        break;
    }
    case 3://Matric vector with row mayor
    {

        viennacl::matrix<T,viennacl::row_major> A(BLAS_N, BLAS_N);
        viennacl::matrix<T,viennacl::row_major> B(BLAS_N, BLAS_N);
        viennacl::matrix<T,viennacl::row_major> C(BLAS_N, BLAS_N);
        viennacl::vector<T> x(BLAS_N);
        viennacl::vector<T> y(BLAS_N);

        T a = (T)2.4;
        T b = (T)3.8;
        T c = (T)3.8;

        init_random(A);
        init_random(B);
        init_random(C);
        init_random(x);
        init_random(y);
        MATRIX_OP
        break;
    }
    }
    output.close();

}
int main(int argc, char *argv[])
{
    //argv[1]-->vector or matrix size
    //argv[2]-->if the operations are(0:testing,1:Vector,2:Matrix coloum mayor,3:Matrix row mayor)
    std::fstream output;
    std::size_t BLAS_N = atoi(argv[1]);
    if(BLAS_N==0)
    {
        output.open(FILENAME,std::ios::out);
        output << std::left << std::setw(11) << "N" <<",";
        output.close();
    }

    output.open(FILENAME,std::ios::in|std::ios::out|std::ios::ate);

    if(BLAS_N!=0) output<< std::setprecision(3) << std::left << std::setw(11) << double(BLAS_N) <<" ";
    output.close();
    bench<float>(BLAS_N, "s.",atoi(argv[2]));
    bench<double>(BLAS_N, "d.",atoi(argv[2]));
    output.open(FILENAME,std::ios::in|std::ios::out|std::ios::ate);
    output << std::endl;
    output.close();
}
