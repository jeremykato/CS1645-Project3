#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <pthread.h>

#define POINT(x, y) (x + (y * (total_n)))

int total_n, total_m, total_grid_size, n, m;

double *t_appx, *t_exact;
double x_min, x_max, y_min, y_max, delta_change, x_size, y_size, global_max_change;

pthread_barrier_t barrier;
pthread_mutex_t mutex;

struct jacobi_params {
    int t_num;
    int n_offset;
    int m_offset;
};

void print_matrix(double* matrix, int total_n, int total_m) {
    for (int j = (total_m - 1); j >= 0; j--) {
        for (int i = 0; i < total_n; i++) {
            printf("%2.6f\t", matrix[POINT(i, j)]);
        }
        printf("\n");
    }
    printf("\n");
}

double get_max_error(double* appx, double* exact, int total_n, int total_m) {
    
    double max_error = 0.0, curr_error = 0.0;

    for (int i = 0; i < total_n; i++) {
        for (int j = 0; j < total_m; j++) {
            curr_error = fabs(appx[POINT(i, j)] - exact[POINT(i, j)]);
            if (curr_error > max_error) {
                max_error = curr_error;
            }
        }
    }

    return max_error;

}

void* jacobi_iteration(void* v_param) {

    struct jacobi_params *params = (struct jacobi_params*) v_param;
    double max_change = 1.0; // once this is less than the delta_change, we finish


    int n_offset = params->n_offset, m_offset = params->m_offset;
    int t_num = params->t_num;
    
    printf("%d: Got params.\n", t_num);

    // the way our jacobi iteration runs, we need a separate data structure to put our
    // updated values into
    double *t_appx_new = calloc(n * m, sizeof(double));

    if (t_appx_new == NULL) {
        printf("calloc failed: could not acquire memory.\n");
        exit(-1);
    }

    printf("%d:\tn_off: %d\tm_off: %d\tn: %d\tm: %d\n", t_num, n_offset, m_offset, n, m);
    pthread_barrier_wait(&barrier);
    // loop while the max change is fewer than our specified delta
    while (global_max_change > delta_change) {

        max_change = 0.0;
        for (int j = m_offset; j < (m_offset + m); j++) {
            for (int i = n_offset; i < (n_offset + n); i++) {

                printf("%d: looking at global (%d, %d), local (%d, %d)\n", t_num, i, j, (i - n_offset), (j - m_offset));

                double old = t_appx[POINT(i, j)];
                t_appx[POINT(i, j)] = 0.25 * (t_appx[POINT((i - 1), j)] + t_appx[POINT((i + 1), j)] + t_appx[POINT(i, (j - 1))] + t_appx[POINT(i, (j + 1))]);
                double change = fabs(t_appx[POINT(i, j)] - old);

                if (change > max_change) {
                    max_change = change;
                }
            }
        }

        pthread_mutex_lock(&mutex);
        printf("%d: Got lock, my change val is %2.12f.\n", t_num, max_change);
        if (global_max_change < max_change) {
            global_max_change = max_change;
        }
        pthread_mutex_unlock(&mutex);

        pthread_barrier_wait(&barrier);
    }

    // if we reach here, we're done!
    return 0;

}

int main(int argc, char *argv[]) {

    if (argc != 10) {
        printf("Error: 9 arguments are necessary.\n");
        printf("program-name <X-Min> <X-Max> <Y-Min> <Y-Max> <Grid Width Per Thread> <Grid Height Per Thread>\n"\
               "             <Convergence-Value> <Threads in X Direction> <Threads in Y Direction>\n");
        exit(-1);
    }

    x_min = atof(argv[1]), x_max = atof(argv[2]);
    y_min = atof(argv[3]), y_max = atof(argv[4]);

    x_size = fabs(x_max - x_min);
    y_size = fabs(y_max - y_min);
    
    // per thread
    n = atoi(argv[5]);
    m = atoi(argv[6]); 
    delta_change = atof(argv[7]);

    // threads
    const int   threads_x = atoi(argv[8]),
                threads_y = atoi(argv[9]), 
                total_threads = threads_x * threads_y;

    // grid sizes
    total_n = (threads_x * n) + 2;
    total_m = (threads_y * m) + 2;
    total_grid_size = total_n * total_m;

    // temperature grid for the approximated solution
    // and the exact solution
    t_appx  = (double*) calloc(total_grid_size, sizeof(double));
    t_exact = (double*) calloc(total_grid_size, sizeof(double));

    if (t_appx == NULL || t_exact == NULL) {
        printf("calloc failed: could not acquire memory.\n");
        exit(-1);
    }

    // boundary conditions for the domain of interest
    double x, y;
    for (int i = 0; i < total_n; i++) {
        printf("Setting up point x=%d, y=0 and y=%d\n", i, total_m - 1);
        x = (((double) i / (double) (total_n - 1)) * x_size) + x_min;       // x
        t_appx[POINT(i, 0)] = x;                                        // T(x, 0) = x
        t_appx[POINT(i, (total_m - 1))] = x * exp(1);                   // T(x, 1) = x * e
    }

    for (int i = 0; i < total_m; i++) {
        printf("Setting up point x=0 and x=%d, y=%d\n", total_n - 1, i);
        y = (((double) i / (double) (total_m - 1)) * y_size) + y_min;   // y
        t_appx[POINT(0, i)] = 0;                                        // T(0, y) = 0
        t_appx[POINT((total_n - 1), i)] = 2 * exp(y);                   // T(2, y) = 2 * e^y
    }

    // pthreads go here
    pthread_t *threads = calloc(total_threads, sizeof(pthread_t));
    struct jacobi_params *params = calloc(total_threads, sizeof(struct jacobi_params));
    pthread_barrier_init(&barrier, NULL, (unsigned int) total_threads);
    pthread_mutex_init(&mutex, NULL);
    global_max_change = 1.0;
    
    for (int i = 0; i < total_threads; i++) {
        params[i].t_num = i;
        params[i].n_offset = n * (i % threads_y) + 1;
        params[i].m_offset = m * (i / threads_y) + 1;
        pthread_create((threads + i), NULL, jacobi_iteration, (void*) (params + i));
    }

    for (int i = 0; i < total_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // compute the exact values along the domain
    // as a note, it would be faster to simply calculate each x and y value, store them,
    // and then multiply them together to avoid having to recalculate each value between
    // x/y min and max, but this will do for now (and for the exact answer, we worry about 
    // performance slightly less)
    for (int i = 0; i < total_n; i++) {
        x = (((double) i / (double) (total_n - 1)) * x_size) + x_min; // x
        for (int j = 0; j < total_m; j++) {
            y = (((double) j / (double) (total_m - 1)) * y_size) + y_min; // y
            printf("(%2.6f, %2.6f)\n", x, y);
            t_exact[POINT(i, j)] = x * exp(y);
        }
    }

    //printf("Iterations taken:\t%d\n", iterations);
    printf("Done!\n");
    printf("Max error: \t\t%2.6f\n", get_max_error(t_appx, t_exact, total_n, total_m));

    // printf("Point\t\tExact\t\tAppx.\n");
    // // print out the results
    // for (int j = 0; j < m; j++) {

    //     y = (((double) j / (double) (m - 1)) * y_size) + y_min; // y

    //     for (int i = 0; i < n; i++) {

    //         x = (((double) i / (double) (n - 1)) * x_size) + x_min; // x


    //         printf("(%1.2f, %1.2f)\t%2.6f\t\t%2.6f\n", x, y, 
    //             t_exact[POINT(i, j)], t_appx[POINT(i, j)]);
            
    //     }

    // }

    print_matrix(t_appx, total_n, total_m);
    print_matrix(t_exact, total_n, total_m);
}