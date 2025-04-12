#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <pthread.h>
#include <omp.h>

int main()
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("A ");
            int a = 1;
            #pragma omp task depend (inout: a)
            {
                #pragma omp parallel for
                for(int i=0; i<100; i++){
                    printf("race ");
                    fflush(stdout);
                }
                a = 4;
            }
            
            #pragma omp task depend (in: a)
            {
                printf("a value is %d \n", a);
                #pragma omp parallel for 
                for(int i=0; i<100; i++){
                    printf("car! ");
                    fflush(stdout);
                }
            }

        }
    }
    printf("\n");
    return 0;
}


// void* thread_function(void* arg) {
//     int coreID = 0; // Specify the core ID to bind (e.g., core 0)

//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(coreID, &cpuset);

//     pthread_t thread = pthread_self();
//     pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

//     // Your MPI communication code here
//     int cores = sched_getcpu();
//     printf("threads function, core id %d \n", cores);
//     return nullptr;
// }

// int main(int argc, char* argv[]) {
//     int nprocs, myrank;
//     int provided;
//     MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE,&provided);
//     MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
//     MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

//     if (provided != MPI_THREAD_MULTIPLE) {
//         std::cerr << "This program requires MPI_THREAD_SINGLE support." << std::endl;
//         MPI_Finalize();
//         return 1;
//     }

//     // pthread_t first_thread;
//     // pthread_create(&first_thread, nullptr, thread_function, nullptr);

//     // // Set OpenMP environment variables to control affinity
//     // // Avoid binding OpenMP threads to the same core as the first thread
//     // setenv("OMP_PROC_BIND", "true", 1);
//     // setenv("OMP_PLACES", "cores", 1);
//     // omp_set_num_threads(4);
//     #pragma omp parallel
//     {
//         // Your OpenMP parallel code here
//         int tid = omp_get_thread_num();
//         int nthreads = omp_get_num_threads();
//         int cores = sched_getcpu();
//         printf("myrank %d nthreads %d tid %d cores %d \n",myrank, nthreads, tid, cores);
//     }

//     // pthread_join(first_thread, nullptr);

//     MPI_Finalize();
//     return 0;
// }
